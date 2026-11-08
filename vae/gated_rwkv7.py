"""LiuBai-H: the gated read-only NAR expert of vae/gated_nar.py on the official RWKV-7 x070 block, so the pretrained g1k-3b
(32 x 2560, models/rwkv-g1k-3b-temp) loads as is (docs/VAE_PLAN.md, revision 2026-10-27 and the 3B port).

Sequence: [text prefix] A_0 [F_0,1..blk] A_1 [F_1,1..blk] ... (text = RWKV-vocab tokens, causal, for captions / native lyrics
later; A_i = AR block token = inp(block i-1) + cursor window i; F_i,j = noisy frame j of block i at flow time t). Mask cells as
in gated_nar.py:
  A->A causal       the official block over text + A; token shift skips F rows
  A->F blocked      F rows: decay logit -1e4 (w = 1 in the clampw kernel), k = 0, a = 0 -> the WKV state passes F unchanged
  F->A full         F rows read S with their own r in every layer
  F->F              forward shift F_i,j <- F_i,j-1 (F_i,1 <- A_i) + zero-init backward shift F_i,j <- F_i,j+1 (r, g, FFN);
                    lattn=True adds a zero-init local attention of the blk frames on [cursor letters, clean frames of block
                    i-1, each other] per layer (F rows only)
Official block math (tools/RWKV-LM/RWKV-v7/rwkv_v7_demo.py): value residual from layer 0, decay w0 + LoRA, k_a scaling, bonus
(r * k * r_k) v, GroupNorm ln_x eps 64e-5. The bonus is a per-row self term, so F rows keep their own k there (no leak).
NAR expert = pretrained weights + zero-init deltas on receptance / output / FFN key / value (lora > 0: rank-lora LoRA;
lora = 0: full-rank delta matrices = an untied full copy of those four weights per layer, +2.1B params at 3B) + adaLN(t) norms
initialised from the pretrained ln1 / ln2 / ln_out (zero-init modulation): at step 0 an F row runs the pretrained 3B.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .rwkv import wkv7, wkv7_ref
from .renderer import t_embed
from .gated_nar import layout as _layout, gather, LoRA, LocalAttn, _fshift, fread

N_HEAD = 64


class FullDelta(nn.Linear):
    """zero-init full-rank delta (lora = 0): W_F = W + delta for the F rows"""
    def __init__(s, din, dout):
        super().__init__(din, dout, bias=False); nn.init.zeros_(s.weight)


def delta(din, dout, r): return LoRA(din, dout, r) if r > 0 else FullDelta(din, dout)


def layout(nb, blk, nar, iso=False, reps=1, prefix=0):
    """gated_nar.layout with `prefix` causal text positions in front (A_0's shift reads the last text token); adds t_pos"""
    L = _layout(nb, blk, nar, iso, reps); P = int(prefix)
    if not P:
        L["t_pos"] = torch.zeros(0, dtype=torch.long); return L
    sh = lambda t: torch.where(t >= 0, t + P, t)
    prev = sh(L["prev"]); prev[L["a_pos"][0]] = P - 1
    return dict(isf=torch.cat([torch.zeros(P, dtype=torch.bool), L["isf"]]), prev=torch.cat([torch.arange(-1, P - 1), prev]),
                nxt=torch.cat([torch.full((P,), -1), sh(L["nxt"])]), a_pos=L["a_pos"] + P, f_pos=L["f_pos"] + P,
                f_blk=L["f_blk"], nar=L["nar"], t_pos=torch.arange(P))


class AdaLN(nn.Module):
    """LayerNorm (affine copied from a pretrained LN) then shift / scale from the flow-time embedding (zero-init)"""
    def __init__(s, C, dc):
        super().__init__(); s.weight, s.bias = nn.Parameter(torch.ones(C)), nn.Parameter(torch.zeros(C)); s.mod = nn.Linear(dc, 2 * C)
        nn.init.zeros_(s.mod.weight); nn.init.zeros_(s.mod.bias)
    def forward(s, x, c):
        sh, sc = s.mod(c).chunk(2, -1); return F.layer_norm(x, x.shape[-1:], s.weight, s.bias) * (1 + sc) + sh


class TMix(nn.Module):
    def __init__(s, C, layer_id, dims=(96, 96, 64, 320), lora=64):
        super().__init__(); s.layer_id = layer_id; H = s.H = C // N_HEAD; Dw, Da, Dv, Dg = dims; P = lambda *sh: nn.Parameter(torch.zeros(*sh))
        s.x_r, s.x_w, s.x_k, s.x_v, s.x_a, s.x_g = (P(1, 1, C) for _ in range(6))
        s.w0, s.w1, s.w2 = P(1, 1, C), P(C, Dw), P(Dw, C)
        s.a0, s.a1, s.a2 = P(1, 1, C), P(C, Da), P(Da, C)
        s.v0, s.v1, s.v2 = P(1, 1, C), P(C, Dv), P(Dv, C)
        s.g1, s.g2 = P(C, Dg), P(Dg, C)
        s.k_k, s.k_a, s.r_k = P(1, 1, C), P(1, 1, C), P(H, N_HEAD)
        s.receptance, s.key, s.value, s.output = (nn.Linear(C, C, bias=False) for _ in range(4))
        s.ln_x = nn.GroupNorm(H, C, eps=64e-5)
        s.mix_b = P(2, C)                                                   # NAR: backward shift for r, g (F rows only)
        s.r_f, s.o_f = delta(C, C, lora), delta(C, C, lora)

    def forward(s, x, v_first, lay, f):
        B, T, C = x.shape; H, N = s.H, N_HEAD
        dx = gather(x, lay["prev"]) - x; db = f * (gather(x, lay["nxt"]) - x)
        xr = x + dx * s.x_r + db * s.mix_b[0]; xg = x + dx * s.x_g + db * s.mix_b[1]
        xw, xk, xv, xa = x + dx * s.x_w, x + dx * s.x_k, x + dx * s.x_v, x + dx * s.x_a
        r = s.receptance(xr) + f * s.r_f(xr)
        w = s.w0 + torch.tanh(xw @ s.w1) @ s.w2
        k, v = s.key(xk), s.value(xv)
        if s.layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(s.v0 + (xv @ s.v1) @ s.v2)
        a = torch.sigmoid(s.a0 + (xa @ s.a1) @ s.a2)
        g = torch.sigmoid(xg @ s.g1) @ s.g2
        hv = lambda t: t.to(torch.promote_types(t.dtype, torch.float32)).view(B, T, H, N)      # fp32 (fp64 in the CPU tests)
        kk = F.normalize(hv(k * s.k_k), dim=-1)
        k = k * (1 + (a - 1) * s.k_a)
        wg = torch.where(f, torch.full_like(w, -1e4), w)                                     # write gate on F rows
        kg = torch.where(f, torch.zeros_like(k), k); ag = torch.where(f, torch.zeros_like(a), a)
        y = (wkv7 if x.is_cuda else wkv7_ref)(hv(r), hv(wg), hv(kg), hv(v), kk, hv(ag))
        y = y.to(torch.promote_types(y.dtype, torch.float32)).reshape(B * T, C)
        y = F.group_norm(y, H, s.ln_x.weight.to(y.dtype), s.ln_x.bias.to(y.dtype), s.ln_x.eps).view(B, T, C)   # fp32 (bf16 params)
        y = y + ((hv(r) * hv(k) * s.r_k).sum(-1, keepdim=True) * hv(v)).view(B, T, C)
        y = (y.to(x.dtype) * g)
        return s.output(y) + f * s.o_f(y), v_first

    def step(s, x, xp, xn, v_first, S, isf):
        """incremental: x [B,n,C] normed rows, xp / xn [B,n,C] shift sources (xn ignored for A), S [B,H,N,N] state.
        A (isf False, n = 1): one recurrence step (= wkv7_ref), S advanced. F rows: S read only (w = 1, k = 0, a = 0 in forward).
        -> out [B,n,C], v_first, S"""
        B, n, C = x.shape; H, N = s.H, N_HEAD; dx = xp - x; db = (xn - x) if isf else torch.zeros_like(x)
        xr = x + dx * s.x_r + db * s.mix_b[0]; xg = x + dx * s.x_g + db * s.mix_b[1]
        xw, xk, xv, xa = x + dx * s.x_w, x + dx * s.x_k, x + dx * s.x_v, x + dx * s.x_a
        r = s.receptance(xr) + (s.r_f(xr) if isf else 0)
        w = s.w0 + torch.tanh(xw @ s.w1) @ s.w2
        k, v = s.key(xk), s.value(xv)
        if s.layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(s.v0 + (xv @ s.v1) @ s.v2)
        a = torch.sigmoid(s.a0 + (xa @ s.a1) @ s.a2)
        g = torch.sigmoid(xg @ s.g1) @ s.g2
        hv = lambda t: t.to(torch.promote_types(t.dtype, torch.float32)).view(B, n, H, N)
        kk = F.normalize(hv(k * s.k_k), dim=-1)
        k = k * (1 + (a - 1) * s.k_a); R, Kh, V = hv(r), hv(k), hv(v); S = S.to(R.dtype)
        if isf: y = fread(S, R)
        else:
            wd = torch.exp(-math.exp(-0.5) * torch.sigmoid(hv(w)[:, 0])); k0, a0 = kk[:, 0], hv(a)[:, 0]
            sa = (S * k0.unsqueeze(-2)).sum(-1)
            S = S * wd.unsqueeze(-2) - sa.unsqueeze(-1) * (k0 * a0).unsqueeze(-2) + V[:, 0].unsqueeze(-1) * Kh[:, 0].unsqueeze(-2)
            y = (S * R[:, 0].unsqueeze(-2)).sum(-1)[:, None]
        y = y.reshape(B * n, C)
        y = F.group_norm(y, H, s.ln_x.weight.to(y.dtype), s.ln_x.bias.to(y.dtype), s.ln_x.eps).view(B, n, C)
        y = y + ((R * Kh * s.r_k).sum(-1, keepdim=True) * V).view(B, n, C)
        y = y.to(x.dtype) * g
        return s.output(y) + (s.o_f(y) if isf else 0), v_first, S


class CMix(nn.Module):
    def __init__(s, C, lora=64):
        super().__init__(); s.x_k = nn.Parameter(torch.zeros(1, 1, C)); s.mix_b = nn.Parameter(torch.zeros(C))
        s.key, s.value = nn.Linear(C, 4 * C, bias=False), nn.Linear(4 * C, C, bias=False)
        s.k_f, s.v_f = delta(C, 4 * C, lora), delta(4 * C, C, lora)
    def forward(s, x, lay, f):
        xx = x + (gather(x, lay["prev"]) - x) * s.x_k + f * (gather(x, lay["nxt"]) - x) * s.mix_b
        h = torch.relu(s.key(xx) + f * s.k_f(xx)) ** 2
        return s.value(h) + f * s.v_f(h)

    def step(s, x, xp, xn, isf):
        if not isf: return s.value(torch.relu(s.key(x + (xp - x) * s.x_k)) ** 2)
        xx = x + (xp - x) * s.x_k + (xn - x) * s.mix_b; h = torch.relu(s.key(xx) + s.k_f(xx)) ** 2
        return s.value(h) + s.v_f(h)


class Block(nn.Module):
    def __init__(s, C, layer_id, dims, lora, dc, lattn, da):
        super().__init__(); s.layer_id = layer_id; s.ckpt = False
        if layer_id == 0: s.ln0 = nn.LayerNorm(C)
        s.ln1, s.ln2 = nn.LayerNorm(C), nn.LayerNorm(C); s.ln1_f, s.ln2_f = AdaLN(C, dc), AdaLN(C, dc)
        s.att = TMix(C, layer_id, dims, lora); s.ffn = CMix(C, lora); s.la = LocalAttn(C, N_HEAD, da) if lattn else None

    def forward(s, x, v_first, lay, c, mem=None):          # per-layer activation checkpointing inside the (FSDP-wrapped) block
        if s.ckpt and s.training: return torch.utils.checkpoint.checkpoint(s._fwd, x, v_first, lay, c, mem, use_reentrant=False)
        return s._fwd(x, v_first, lay, c, mem)

    def _fwd(s, x, v_first, lay, c, mem):
        f = lay["isf"][None, :, None]
        if s.layer_id == 0: x = s.ln0(x)
        dx, v_first = s.att(torch.where(f, s.ln1_f(x, c), s.ln1(x)), v_first, lay, f); x = x + dx
        if s.la is not None and mem is not None:                                     # F rows only: A rows untouched
            B = x.shape[0]; G, n = lay["f_pos"].shape; fp = lay["f_pos"].reshape(-1)
            x = x.index_add(1, fp, s.la(x[:, fp].view(B, G, n, -1), mem).reshape(B, G * n, -1).to(x.dtype))
        return x + s.ffn(torch.where(f, s.ln2_f(x, c), s.ln2(x)), lay, f), v_first

    def step_a(s, x, v_first, cl):
        """one causal token (text or A) x [B,1,C]; cl = this layer's cache (S, p1 / p2 = previous token's ln1 / ln2 outputs)"""
        if s.layer_id == 0: x = s.ln0(x)
        u = s.ln1(x); dx, v_first, cl["S"] = s.att.step(u, cl["p1"][:, None].to(u.dtype), None, v_first, cl["S"], False)
        cl["p1"] = u[:, 0]; x = x + dx
        u = s.ln2(x); x = x + s.ffn.step(u, cl["p2"][:, None].to(u.dtype), None, False); cl["p2"] = u[:, 0]
        return x, v_first

    def step_f(s, x, v_first, c, mem, cl, iso):
        """F rows of the block after the last cached A token: x [B,blk,C], c [B,1,dc], mem [B,L,C] or None"""
        if s.layer_id == 0: x = s.ln0(x)
        u = s.ln1_f(x, c); dx, v_first, _ = s.att.step(u, *_fshift(u, cl["p1"], iso), v_first, cl["S"], True); x = x + dx
        if s.la is not None and mem is not None: x = x + s.la(x[:, None], mem[:, None])[:, 0].to(x.dtype)
        u = s.ln2_f(x, c)
        return x + s.ffn.step(u, *_fshift(u, cl["p2"], iso), True), v_first


class LiuBaiH(nn.Module):
    """g1k-3b backbone + audio A / F rows + lyric cursor (GatedCursorLM API: hidden / velocity)"""
    mode = "cursor"

    def __init__(s, n_layer=32, C=2560, vocab=65536, dims=(96, 96, 64, 320), latent=64, blk=4, lora=64, dc=256, wb=4, wf=20,
                 max_adv=24, iso=False, lattn=False, da=512):
        super().__init__(); s.latent, s.blk, s.wb, s.wf, s.max_adv, s.iso, s.lattn, s.C = latent, blk, wb, wf, max_adv, iso, lattn, C
        s.emb = nn.Embedding(vocab, C); s.ln_out = nn.LayerNorm(C); s.head = nn.Linear(C, vocab, bias=False)
        s.blocks = nn.ModuleList(Block(C, i, dims, lora, dc, lattn, da) for i in range(n_layer))
        s.inp = nn.Linear(latent * blk, C); s.bos = nn.Parameter(torch.zeros(C)); s.kind = nn.Embedding(2, C)   # audio A / F rows
        s.fin = nn.Linear(latent, C); s.fpos = nn.Parameter(torch.randn(blk, C) * 0.02)
        s.tm = nn.Sequential(nn.Linear(256, dc), nn.SiLU(), nn.Linear(dc, dc)); s.tin = nn.Linear(dc, C)
        s.norm_f = AdaLN(C, dc); s.out = nn.Linear(C, latent); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
        from .cursor_lm import N_CTC
        s.wl = nn.Embedding(N_CTC + 2, 64); s.wpos = nn.Parameter(torch.randn(wb + wf, 64) * 0.02); s.win = nn.Linear((wb + wf) * 64, C)
        s.adv = nn.Linear(C, max_adv); s.adv_emb = nn.Embedding(max_adv + 1, C)                  # id max_adv = null
        if lattn: s.cin = nn.Linear(latent, C); s.cpos = nn.Parameter(torch.randn(blk, C) * 0.02); s.lmem = nn.Linear(64, C)
        s.D = latent * blk

    def set_ckpt(s, on):
        for b in s.blocks: b.ckpt = on

    def load_pretrained(s, path):
        """official .pth -> backbone; NAR norms start as copies of the pretrained ones. Returns the new (untrained) param names."""
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        miss, unexp = s.load_state_dict(sd, strict=False); assert not unexp, unexp[:8]
        assert all(not k.startswith("blocks.") or any(t in k for t in ("_f.", "mix_b", ".la.")) for k in miss), [k for k in miss if k.startswith("blocks.")][:8]
        with torch.no_grad():
            for b in s.blocks:
                for src, dst in ((b.ln1, b.ln1_f), (b.ln2, b.ln2_f)): dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
            s.norm_f.weight.copy_(s.ln_out.weight); s.norm_f.bias.copy_(s.ln_out.bias)
        return [k for k in miss if not k.endswith(("_f.weight", "_f.bias")) or "mod" in k]

    def forward(s, z, xt, t, lay, lyr=None, a_c=None, gidx=None, txt=None):
        """z [B,NB,D] clean blocks (z[:, :-1] read), xt [B,K,blk,latent] noisy frames of the blocks lay["nar"], t [B,K], lyr [B,NB,W]
        cursor windows, a_c [B,NB] advance ids, gidx [K] block of each F group, txt [B,P] text ids (P = len(lay["t_pos"]))
        -> h [B,NB,C] (A rows), v [B,K,blk,latent], ht [B,P,C] (text rows, after ln_out), advance logits [B,NB,max_adv].
        Everything with parameters runs in here (FSDP2: root params are only unsharded inside the root forward)."""
        B, NB, _ = z.shape; K = xt.shape[1]; T = lay["isf"].numel(); C = s.C; dt = s.inp.weight.dtype; dev = z.device
        lay = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in lay.items()}
        a = torch.cat([s.bos.expand(B, 1, -1), s.inp(z[:, :-1].to(dt))], 1) + s.kind.weight[0]
        ct = s.tm(t_embed(t.reshape(-1).float(), 256).to(dt)).view(B, K, 1, s.tm[-1].out_features)
        fr = s.fin(xt.to(dt)) + s.fpos + s.kind.weight[1] + s.tin(ct)
        mx = None
        if lyr is not None:                                                                  # cursor window i on A_i and on block i's F groups
            le = s.wl(lyr) + s.wpos; ca = s.win(le.flatten(2).to(dt)); a = a + ca
            if K:
                fr = fr + (ca[:, gidx] + s.adv_emb(a_c[:, gidx]))[:, :, None]
                if s.lattn: mx = s.lmem(le[:, gidx].to(dt))
        x = torch.zeros(B, T, C, device=dev, dtype=dt); x[:, lay["a_pos"]] = a.to(dt)
        fp = lay["f_pos"].reshape(-1); x[:, fp] = fr.reshape(B, K * s.blk, C).to(dt)
        if txt is not None and len(lay["t_pos"]): x[:, lay["t_pos"]] = s.emb(txt).to(dt)
        dc = ct.shape[-1]; c = torch.zeros(B, T, dc, device=dev, dtype=dt); c[:, fp] = ct.expand(B, K, s.blk, dc).reshape(B, K * s.blk, dc).to(dt)
        mem = None
        if s.lattn and K:                                                                   # block i-1's frames (zeros for i = 0)
            zp = F.pad(z.view(B, NB, s.blk, s.latent), (0, 0, 0, 0, 1, 0))[:, lay["f_blk"]]
            mem = s.cin(zp.to(dt)) + s.cpos
            if mx is not None: mem = torch.cat([mx.to(mem.dtype), mem], 2)
        v_first = torch.zeros_like(x)
        for b in s.blocks: x, v_first = b(x, v_first, lay, c, mem)
        h = s.ln_out(x[:, lay["a_pos"]]); ht = s.ln_out(x[:, lay["t_pos"]]) if len(lay["t_pos"]) else None
        v = s.out(s.norm_f(x[:, fp], c[:, fp])).view(B, K, s.blk, s.latent)
        return h, v, ht, s.adv(h)

    def hidden(s, z, lyr, txt=None):
        """A stream only (exact: A never reads F) -> h [B,NB,C]"""
        B, NB, _ = z.shape; P = 0 if txt is None else txt.shape[1]
        return s(z, z.new_zeros(B, 0, s.blk, s.latent), z.new_zeros(B, 0), layout(NB, s.blk, [], s.iso, prefix=P), lyr=lyr, txt=txt)[0]

    def velocity(s, z, lyr, a_c, xt, t, nar, reps=1, txt=None):
        """as GatedCursorLM.velocity (+ optional text prefix) -> h [B,NB,C], v [B,G,D], ht, advance logits [B,NB,max_adv]"""
        B, NB, _ = z.shape; nar = sorted(int(i) for i in nar); G = xt.shape[1]; assert G == len(nar) * reps
        P = 0 if txt is None else txt.shape[1]; gidx = torch.tensor(nar, device=z.device).repeat_interleave(reps)
        h, v, ht, lg = s(z, xt.view(B, G, s.blk, s.latent), t, layout(NB, s.blk, nar, s.iso, reps, prefix=P), lyr=lyr, a_c=a_c, gidx=gidx, txt=txt)
        return h, v.reshape(B, G, -1), ht, lg

    # incremental inference (same API as gated_nar.GatedCursorLM): exact w.r.t. forward because A / text never read F and F
    # never writes S; per block one step_hidden, then each flow step is a blk-row pass against the cached state
    def cache_init(s, B, device, txt=None):
        """-> per-layer cache; txt [B,P] optional text prefix, run through the causal stream first"""
        dt = s.inp.weight.dtype; H = s.C // N_HEAD; z = lambda: torch.zeros(B, s.C, device=device, dtype=dt)
        cache = [dict(S=torch.zeros(B, H, N_HEAD, N_HEAD, device=device, dtype=torch.promote_types(dt, torch.float32)), p1=z(), p2=z())
                 for _ in s.blocks]
        if txt is not None:
            for p in range(txt.shape[1]): s._step_a(s.emb(txt[:, p]), cache)
        return cache

    def _step_a(s, a, cache):
        x = a[:, None].to(s.inp.weight.dtype); vf = None
        for b, cl in zip(s.blocks, cache): x, vf = b.step_a(x, vf, cl)
        return x[:, 0]

    def step_hidden(s, zprev, lyr_i, cache):
        """A token of block i: zprev [B,D] block i-1 (None at i = 0), lyr_i [B,W] window i -> h_i [B,C] (after ln_out)"""
        B = lyr_i.shape[0]; dt = s.inp.weight.dtype
        a = (s.bos.expand(B, -1) if zprev is None else s.inp(zprev.to(dt))) + s.kind.weight[0] + s.win((s.wl(lyr_i) + s.wpos).flatten(1).to(dt))
        return s.ln_out(s._step_a(a, cache))

    def step_velocity(s, xt, t, lyr_i, a_i, cache, zprev=None):
        """flow step of block i after step_hidden(.., lyr_i, cache): xt [B,D], t [B], a_i [B] advance ids -> v [B,D]"""
        B = xt.shape[0]; dt = s.inp.weight.dtype; ct = s.tm(t_embed(t.reshape(-1).float(), 256).to(dt))[:, None]
        le = s.wl(lyr_i) + s.wpos; ca = s.win(le.flatten(1).to(dt))
        x = s.fin(xt.view(B, s.blk, s.latent).to(dt)) + s.fpos + s.kind.weight[1] + s.tin(ct) + (ca + s.adv_emb(a_i))[:, None]
        mem = None
        if s.lattn:
            zp = xt.new_zeros(B, s.blk, s.latent) if zprev is None else zprev.view(B, s.blk, s.latent)
            mem = torch.cat([s.lmem(le.to(dt)), s.cin(zp.to(dt)) + s.cpos], 1)
        vf = None
        for b, cl in zip(s.blocks, cache): x, vf = b.step_f(x, vf, ct, mem, cl, s.iso)
        return s.out(s.norm_f(x, ct)).reshape(B, -1)
