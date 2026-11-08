"""Gated read-only NAR expert: YuE2's hybrid AR / NAR mask rebuilt on a pure RWKV-7 stack (docs/VAE_PLAN.md, revision 2026-10-27).

One interleaved sequence A_0 [F_0,1..blk] A_1 [F_1,1..blk] ... (A_i = AR block token that predicts block i, = inp(block i-1);
F_i,j = noisy frame j of block i at flow time t_i, only for the blocks in `nar`). Mask cells:
  A->A causal          normal RWKV over the A tokens; their token shift skips the F rows (A_i reads A_{i-1})
  A->F blocked         write gate: on F rows w = 1 (w_raw = -1e4), k = 0, a = 0 -> the WKV update leaves S exactly unchanged
  F->A full            F_i rows read S (after A_i) with their own r in every layer
  F->F bidirectional   forward shift F_i,j <- F_i,j-1 (F_i,1 <- A_i) + backward shift F_i,j <- F_i,j+1 (zero-init mix)
NAR expert = AR weights + zero-init LoRA on r / out / FFN + own adaLN(t) norms, so at init F rows run the AR weights.
`iso=True` drops the in-block shifts (F_i,j <- A_i only, no backward) = ablation without F->F.
`lattn=True` adds a local attention per layer on F rows only (zero-init out): the blk frames of one group attend to the clean
frames of block i-1, the group's extra memory (GatedCursorLM: the 24 cursor letters) and each other. A rows never enter it,
so A stays bit-exact; F->F without blending inputs through the shift, letters and seams at frame resolution.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .rwkv import wkv7, wkv7_ref, init_gate
from .renderer import t_embed


def layout(nb, blk, nar, iso=False, reps=1):
    """-> dict of LongTensors over the T = nb + blk * reps * len(nar) interleaved positions: isf [T] bool, prev / nxt [T] shift
    sources (-1 = zero), a_pos [nb] positions of A_i, f_pos [K, blk] positions of F group k (block nar[k // reps]; reps groups
    per block = independent noise draws that all read the same state and never each other)"""
    nar = sorted(set(int(i) for i in nar)); isf, prev, nxt, a_pos, f_pos, f_blk = [], [], [], [], [], []
    for i in range(nb):
        p = len(isf); isf.append(False); prev.append(a_pos[-1] if a_pos else -1); nxt.append(-1); a_pos.append(p)
        if i in nar:
            for _ in range(reps):
                q0 = len(isf); fp = list(range(q0, q0 + blk)); f_pos.append(fp); f_blk.append(i)
                for j, q in enumerate(fp):
                    isf.append(True); prev.append(p if (iso or j == 0) else q - 1); nxt.append(-1 if (iso or j == blk - 1) else q + 1)
    L = lambda x: torch.tensor(x, dtype=torch.long)
    return dict(isf=torch.tensor(isf), prev=L(prev), nxt=L(nxt), a_pos=L(a_pos), f_pos=L(f_pos).view(-1, blk), f_blk=L(f_blk), nar=nar)


def gather(x, idx):
    """x [B,T,C], idx [T] (-1 = zero) -> x[:, idx]"""
    return F.pad(x, (0, 0, 1, 0))[:, idx.to(x.device) + 1]


class LoRA(nn.Module):
    def __init__(s, din, dout, r):
        super().__init__(); s.a = nn.Linear(din, r, bias=False); s.b = nn.Linear(r, dout, bias=False); nn.init.zeros_(s.b.weight)
    def forward(s, x): return s.b(s.a(x))


class AdaLN(nn.Module):
    """LayerNorm with shift / scale from the flow-time embedding (zero-init: plain LN at step 0)"""
    def __init__(s, d):
        super().__init__(); s.ln = nn.LayerNorm(d, elementwise_affine=False); s.mod = nn.Linear(d, 2 * d)
        nn.init.zeros_(s.mod.weight); nn.init.zeros_(s.mod.bias)
    def forward(s, x, c):
        sh, sc = s.mod(F.silu(c)).chunk(2, -1); return s.ln(x) * (1 + sc) + sh


class GTimeMix(nn.Module):
    def __init__(s, d, head=64, lora=32):
        super().__init__(); s.h = d // head; s.n = head
        s.mix = nn.Parameter(torch.full((6, d), 0.5))                    # token-shift interpolation for r,w,k,v,a,g
        s.mix_b = nn.Parameter(torch.zeros(2, d))                        # F rows only: backward shift for r, g
        s.r, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.w1, s.w2 = nn.Linear(d, 64, bias=False), nn.Linear(64, d)
        s.a1, s.a2 = nn.Linear(d, 64, bias=False), nn.Linear(64, d)
        s.g1, s.g2 = nn.Linear(d, 128, bias=False), nn.Linear(128, d, bias=False)
        s.k_k = nn.Parameter(torch.full((d,), 0.85)); s.ln_x = nn.GroupNorm(s.h, d, eps=64e-5)
        s.r_f, s.o_f = LoRA(d, d, lora), LoRA(d, d, lora)               # NAR expert deltas
        nn.init.zeros_(s.o.weight); nn.init.constant_(s.w2.bias, 0.5); init_gate(s.g1, s.g2)
        for m in (s.w2, s.a2): nn.init.zeros_(m.weight)

    def forward(s, x, lay):
        B, T, C = x.shape; H, N = s.h, s.n; f = lay["isf"].to(x.device)[None, :, None]
        dx = gather(x, lay["prev"]) - x; db = gather(x, lay["nxt"]) - x
        xr, xw, xk, xv, xa, xg = (x + dx * s.mix[i] for i in range(6))
        xr = xr + f * db * s.mix_b[0]; xg = xg + f * db * s.mix_b[1]
        r = s.r(xr) + f * s.r_f(xr); k, v = s.k(xk), s.v(xv)
        w_raw = s.w2(torch.tanh(s.w1(xw))).float()
        a = torch.sigmoid(s.a2(s.a1(xa))).float(); g = s.g2(torch.sigmoid(s.g1(xg)))
        kk = F.normalize((k * s.k_k).float().view(B, T, H, N), dim=-1)
        k = k.float() * (1 + (a - 1) * 0.5)
        w_raw = torch.where(f, torch.full_like(w_raw, -1e4), w_raw)                       # write gate on F rows: w = 1,
        k = torch.where(f, torch.zeros_like(k), k); a = torch.where(f, torch.zeros_like(a), a)  # k = 0, a = 0 -> S unchanged
        r, k, v, w_raw, a = (t.float().view(B, T, H, N) for t in (r, k, v, w_raw, a))
        y = (wkv7 if x.is_cuda else wkv7_ref)(r, w_raw, k, v, kk, a).float().reshape(B * T, C)
        y = s.ln_x(y).view(B, T, C).to(x.dtype) * g
        return s.o(y) + f * s.o_f(y)

    def step_a(s, x, xp, S):
        """one A token (fp32 recurrence = wkv7_ref): x [B,C] ln1 output, xp [B,C] previous A's (zeros at A_0), S [B,H,N,N]
        -> out [B,C], S after this token"""
        B, C = x.shape; H, N = s.h, s.n; dx = xp - x
        xr, xw, xk, xv, xa, xg = (x + dx * s.mix[i] for i in range(6))
        r, k, v = s.r(xr), s.k(xk), s.v(xv)
        w_raw = s.w2(torch.tanh(s.w1(xw))).float(); a = torch.sigmoid(s.a2(s.a1(xa))).float(); g = s.g2(torch.sigmoid(s.g1(xg)))
        kk = F.normalize((k * s.k_k).float().view(B, H, N), dim=-1)
        k = (k.float() * (1 + (a - 1) * 0.5)).view(B, H, N); a = a.view(B, H, N)
        w = torch.exp(-math.exp(-0.5) * torch.sigmoid(w_raw)).view(B, H, N)
        sa = (S * kk.unsqueeze(-2)).sum(-1)
        S = S * w.unsqueeze(-2) - sa.unsqueeze(-1) * (kk * a).unsqueeze(-2) + v.float().view(B, H, N).unsqueeze(-1) * k.unsqueeze(-2)
        y = (S * r.float().view(B, H, N).unsqueeze(-2)).sum(-1).reshape(B, C)
        return s.o(s.ln_x(y).to(x.dtype) * g), S

    def step_f(s, x, xp, xn, S):
        """F rows of one group: x [B,n,C] ln1_f outputs, xp / xn [B,n,C] shift sources, S [B,H,N,N] state after A_i. With
        w = 1, k = 0, a = 0 the update is the identity, so y = S r and only r / g are needed"""
        B, n, C = x.shape; H, N = s.h, s.n; dx, db = xp - x, xn - x
        xr = x + dx * s.mix[0] + db * s.mix_b[0]; xg = x + dx * s.mix[5] + db * s.mix_b[1]
        r = (s.r(xr) + s.r_f(xr)).float().view(B, n, H, N); g = s.g2(torch.sigmoid(s.g1(xg)))
        y = fread(S, r).reshape(B * n, C)
        y = s.ln_x(y).view(B, n, C).to(x.dtype) * g
        return s.o(y) + s.o_f(y)


class GChannelMix(nn.Module):
    def __init__(s, d, mult=4, lora=32):
        super().__init__(); s.mix = nn.Parameter(torch.full((d,), 0.5)); s.mix_b = nn.Parameter(torch.zeros(d))
        s.k = nn.Linear(d, d * mult, bias=False); s.v = nn.Linear(d * mult, d, bias=False); nn.init.zeros_(s.v.weight)
        s.k_f, s.v_f = LoRA(d, d * mult, lora), LoRA(d * mult, d, lora)
    def forward(s, x, lay):
        f = lay["isf"].to(x.device)[None, :, None]
        xx = x + (gather(x, lay["prev"]) - x) * s.mix + f * (gather(x, lay["nxt"]) - x) * s.mix_b
        h = torch.relu(s.k(xx) + f * s.k_f(xx)) ** 2
        return s.v(h) + f * s.v_f(h)

    def step(s, x, xp, xn=None):
        """A token (xn None) or F rows (xn = next-frame source): x / xp / xn same shape"""
        if xn is None: return s.v(torch.relu(s.k(x + (xp - x) * s.mix)) ** 2)
        xx = x + (xp - x) * s.mix + (xn - x) * s.mix_b; h = torch.relu(s.k(xx) + s.k_f(xx)) ** 2
        return s.v(h) + s.v_f(h)


class LocalAttn(nn.Module):
    """F rows only: the blk frames of one NAR group attend bidirectionally to [memory, the group's own frames] (zero-init out).
    Narrow (da wide, 4 * d * da params) so the arm stays near the gated baseline's size; F rows already carry t via ln1_f."""
    def __init__(s, d, head=64, da=256):
        super().__init__(); s.h = da // head; s.ln_q, s.ln_m = nn.LayerNorm(d), nn.LayerNorm(d)
        s.q, s.kv, s.o = nn.Linear(d, da, bias=False), nn.Linear(d, 2 * da, bias=False), nn.Linear(da, d, bias=False); nn.init.zeros_(s.o.weight)

    def forward(s, xf, mem, c=None):
        """xf [B,G,blk,d] frames, mem [B,G,L,d] -> [B,G,blk,d]"""
        B, G, n, _ = xf.shape; H = s.h; xq = s.ln_q(xf)
        kv = s.kv(torch.cat([s.ln_m(mem).to(xq.dtype), xq], 2)); L = kv.shape[2]
        q = s.q(xq).reshape(B * G, n, H, -1).transpose(1, 2)
        k, v = kv.reshape(B * G, L, 2, H, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))
        return s.o(y.transpose(1, 2).reshape(B, G, n, -1))


class GBlock(nn.Module):
    def __init__(s, d, head=64, lora=32, lattn=False):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d); s.ln1_f, s.ln2_f = AdaLN(d), AdaLN(d)
        s.tm = GTimeMix(d, head, lora); s.ffn = GChannelMix(d, lora=lora); s.la = LocalAttn(d, head) if lattn else None
    def forward(s, x, lay, c, mem=None):
        f = lay["isf"].to(x.device)[None, :, None]
        x = x + s.tm(torch.where(f, s.ln1_f(x, c), s.ln1(x)), lay)
        if s.la is not None and mem is not None:                                     # F rows only: index_add leaves A rows untouched
            B = x.shape[0]; G, n = lay["f_pos"].shape; fp = lay["f_pos"].reshape(-1).to(x.device)
            y = s.la(x[:, fp].view(B, G, n, -1), mem)
            x = x.index_add(1, fp, y.reshape(B, G * n, -1).to(x.dtype))
        return x + s.ffn(torch.where(f, s.ln2_f(x, c), s.ln2(x)), lay)

    def step_a(s, x, cl):
        """one A token x [B,d]; cl = this layer's cache (S, p1 / p2 = previous A's ln1 / ln2 outputs), updated in place"""
        u = s.ln1(x); y, cl["S"] = s.tm.step_a(u, cl["p1"], cl["S"]); cl["p1"] = u; x = x + y
        u = s.ln2(x); x = x + s.ffn.step(u, cl["p2"]); cl["p2"] = u
        return x

    def step_f(s, x, c, mem, cl, iso):
        """F rows of the block after the cached A token: x [B,n,d], c [B,1,d], mem [B,L,d] or None"""
        u = s.ln1_f(x, c); x = x + s.tm.step_f(u, *_fshift(u, cl["p1"], iso), cl["S"])
        if s.la is not None and mem is not None: x = x + s.la(x[:, None], mem[:, None])[:, 0].to(x.dtype)
        u = s.ln2_f(x, c)
        return x + s.ffn.step(u, *_fshift(u, cl["p2"], iso))


def fread(S, r):
    """read-only state read of F rows: S [B,H,N,N] (value i, key j), r [B,n,H,N] -> y [B,n,H,N], y_i = sum_j S_ij r_j.
    One batched matmul in >= fp32 (fp64 in the CPU tests; autocast off: it would drop it to bf16) instead of materialising
    S * r as [B,n,H,N,N]."""
    dt = torch.promote_types(torch.promote_types(S.dtype, r.dtype), torch.float32)
    with torch.autocast(S.device.type, enabled=False):
        return torch.matmul(r.to(dt).transpose(1, 2), S.to(dt).transpose(-1, -2)).transpose(1, 2)


def _fshift(u, pa, iso):
    """shift sources of one F group as in layout(): u [B,n,C], pa [B,C] = A_i's row -> prev, next (zeros = no source)"""
    if iso: return pa[:, None].to(u.dtype).expand_as(u), torch.zeros_like(u)
    return torch.cat([pa[:, None].to(u.dtype), u[:, :-1]], 1), torch.cat([u[:, 1:], torch.zeros_like(u[:, :1])], 1)


class GatedNARLM(nn.Module):
    """Block-flow LM whose flow head is the backbone itself (NAR expert rows), not a separate MLP."""
    def __init__(s, latent=64, blk=4, d=1024, n=12, head=64, lora=32, lattn=False):
        super().__init__(); s.latent, s.blk, s.lattn = latent, blk, lattn; D = s.D = latent * blk
        s.inp = nn.Linear(D, d); s.bos = nn.Parameter(torch.zeros(d)); s.kind = nn.Embedding(2, d)    # A / F
        s.fin = nn.Linear(latent, d); s.fpos = nn.Parameter(torch.randn(blk, d) * 0.02)
        s.tm = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList(GBlock(d, head, lora, lattn) for _ in range(n)); s.norm = nn.LayerNorm(d)
        s.norm_f = AdaLN(d); s.out = nn.Linear(d, latent); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
        if lattn: s.cin = nn.Linear(latent, d); s.cpos = nn.Parameter(torch.randn(blk, d) * 0.02)   # clean frames of block i-1
        s.ckpt = False

    def forward(s, z, xt, t, lay, ca=None, cf=None, mx=None):
        """z [B,NB,D] clean blocks (only z[:, :-1] is read); xt [B,K,blk,latent] noisy frames of blocks lay["nar"]; t [B,K];
        ca [B,NB,d] extra A input (cursor window), cf [B,K,d] extra F input (advance emb), mx [B,K,L,d] extra local-attention
        memory per group (lattn) -> h [B,NB,d], v [B,K,blk,latent]"""
        B, NB, _ = z.shape; K = xt.shape[1]; T = lay["isf"].numel(); d = s.bos.numel()
        a = torch.cat([s.bos.expand(B, 1, -1), s.inp(z[:, :-1])], 1) + s.kind.weight[0]
        if ca is not None: a = a + ca
        ct = s.tm(t_embed(t.reshape(-1).float(), 256)).view(B, K, 1, d)
        fr = s.fin(xt) + s.fpos + s.kind.weight[1] + ct
        if cf is not None: fr = fr + cf[:, :, None]
        x = z.new_zeros(B, T, d); x[:, lay["a_pos"]] = a.to(x.dtype); x[:, lay["f_pos"].reshape(-1)] = fr.reshape(B, K * s.blk, d).to(x.dtype)
        c = z.new_zeros(B, T, d); c[:, lay["f_pos"].reshape(-1)] = ct.expand(B, K, s.blk, d).reshape(B, K * s.blk, d).to(x.dtype)
        mem = None
        if s.lattn and K:                                                                   # block i-1's frames (zeros for i = 0)
            zp = F.pad(z.view(B, NB, s.blk, s.latent), (0, 0, 0, 0, 1, 0))[:, lay["f_blk"].to(z.device)]
            mem = s.cin(zp) + s.cpos
            if mx is not None: mem = torch.cat([mx.to(mem.dtype), mem], 2)
        for b in s.blocks:                                                                  # ckpt: recompute each layer in backward
            x = torch.utils.checkpoint.checkpoint(b, x, lay, c, mem, use_reentrant=False) if s.ckpt and s.training else b(x, lay, c, mem)
        h = s.norm(x[:, lay["a_pos"]])
        fp = lay["f_pos"].reshape(-1)
        v = s.out(s.norm_f(x[:, fp], c[:, fp])).view(B, K, s.blk, s.latent)
        return h, v

    # incremental inference, exact w.r.t. forward (A never reads F, F never writes S): one step_a per block caches every
    # layer's state after A_i; each flow step is then a blk-row step_f against it (no re-run of the prefix)
    def cache_init(s, B, device):
        d = s.bos.numel(); tm = s.blocks[0].tm; z = lambda: torch.zeros(B, d, device=device)
        return [dict(S=torch.zeros(B, tm.h, tm.n, tm.n, device=device), p1=z(), p2=z()) for _ in s.blocks]

    def step_a(s, a, cache):
        """a [B,d] input of the next A token (bos / inp(previous block) + kind + extras) -> h [B,d]; cache advanced"""
        x = a
        for b, cl in zip(s.blocks, cache): x = b.step_a(x, cl)
        return s.norm(x)

    def step_f(s, xt, t, cache, zprev=None, cf=None, mx=None, iso=False):
        """noisy frames of the block whose A token is the last in the cache: xt [B,blk,latent], t [B], zprev [B,D] clean
        block i-1 (lattn; None = zeros at i = 0), cf [B,d], mx [B,L,d] -> v [B,blk,latent]"""
        B = xt.shape[0]; ct = s.tm(t_embed(t.float(), 256))[:, None]
        x = s.fin(xt) + s.fpos + s.kind.weight[1] + ct
        if cf is not None: x = x + cf[:, None]
        mem = None
        if s.lattn:
            zp = xt.new_zeros(B, s.blk, s.latent) if zprev is None else zprev.view(B, s.blk, s.latent)
            mem = s.cin(zp) + s.cpos
            if mx is not None: mem = torch.cat([mx.to(mem.dtype), mem], 1)
        for b, cl in zip(s.blocks, cache): x = b.step_f(x, ct, mem, cl, iso)
        return s.out(s.norm_f(x, ct))


class GatedCursorLM(GatedNARLM):
    """CursorLM (vae/cursor_lm.py, cursor mode) with the gated NAR rows as flow head instead of the MLP FlowHead. A_i input =
    inp(block i-1) + cursor window i; each F group of block i adds window i + advance embedding a_i (= CursorLM's head condition)."""
    mode = "cursor"

    def __init__(s, latent=64, blk=4, d=1024, n=16, head=64, lora=64, wb=4, wf=20, max_adv=24, iso=False, lattn=False):
        super().__init__(latent, blk, d, n, head, lora, lattn); s.wb, s.wf, s.max_adv, s.iso = wb, wf, max_adv, iso
        from .cursor_lm import N_CTC
        s.wl = nn.Embedding(N_CTC + 2, 64); s.wpos = nn.Parameter(torch.randn(wb + wf, 64) * 0.02); s.win = nn.Linear((wb + wf) * 64, d)
        s.adv = nn.Linear(d, max_adv); s.adv_emb = nn.Embedding(max_adv + 1, d)                    # id max_adv = null
        if lattn: s.lmem = nn.Linear(64, d)                                 # the window's letters as local-attention memory

    def hidden(s, z, lyr):
        """A stream only (exact: A never reads F) -> h [B,NB,d]"""
        B, NB, _ = z.shape
        return s(z, z.new_zeros(B, 0, s.blk, s.latent), z.new_zeros(B, 0), layout(NB, s.blk, [], s.iso),
                 ca=s.win((s.wl(lyr) + s.wpos).flatten(2)))[0]

    def velocity(s, z, lyr, a_c, xt, t, nar, reps=1):
        """z [B,NB,D] context, lyr [B,NB,W] windows, a_c [B,NB] advance ids; xt [B,G,D] noisy blocks and t [B,G] for the
        G = len(nar) * reps groups (block-major, nar sorted) -> h [B,NB,d], v [B,G,D]"""
        B, NB, _ = z.shape; nar = sorted(int(i) for i in nar); G = xt.shape[1]; assert G == len(nar) * reps
        le = s.wl(lyr) + s.wpos; ca = s.win(le.flatten(2)); idx = torch.tensor(nar, device=z.device).repeat_interleave(reps)
        cf = ca[:, idx] + s.adv_emb(a_c[:, idx]); mx = s.lmem(le[:, idx]) if s.lattn else None
        h, v = s(z, xt.view(B, G, s.blk, s.latent), t, layout(NB, s.blk, nar, s.iso, reps), ca=ca, cf=cf, mx=mx)
        return h, v.reshape(B, G, -1)

    def step_hidden(s, zprev, lyr_i, cache):
        """incremental A token of block i: zprev [B,D] block i-1 (None at i = 0), lyr_i [B,W] window i -> h_i [B,d]"""
        B = lyr_i.shape[0]; ca = s.win((s.wl(lyr_i) + s.wpos).flatten(1))
        a = (s.bos.expand(B, -1) if zprev is None else s.inp(zprev)) + s.kind.weight[0] + ca
        return s.step_a(a, cache)

    def step_velocity(s, xt, t, lyr_i, a_i, cache, zprev=None):
        """incremental flow step of block i after step_hidden(.., lyr_i, cache): xt [B,D], t [B], a_i [B] advance ids -> v [B,D]"""
        B = xt.shape[0]; le = s.wl(lyr_i) + s.wpos; cf = s.win(le.flatten(1)) + s.adv_emb(a_i)
        v = s.step_f(xt.view(B, s.blk, s.latent), t, cache, zprev, cf, s.lmem(le) if s.lattn else None, s.iso)
        return v.reshape(B, -1)
