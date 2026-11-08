"""CPU toy: gated read-only NAR rows (vae/gated_nar.py) vs the dropped state fork (docs/VAE_PLAN.md:315, = FLARE recurrence).
Question: does giving the noisy frames scratch state (they write into a copy of S after A_i) buy anything the gate cannot?
Data (blk 16): block i frames x_i,j = s_i,(j mod 8) (m_i + P_j) + 0.1 e, m_i = roll(mean_j |x_i-1,j|) (needs history = F->A).
Frames j and j + 8 share a random sign; 8 independent signs per block. Distance 8 > the in-block shift reach of 3 layers
(time-mix + FFN shift = 2 frames per layer per direction = 6), so the gated arms can only pair them through local attention.
Arms: bi (gated, forward + backward shift), lattn (gated + per-layer local attention over the block), fork1 (F rows write into a
copy of every layer's S after A_i, one causal pass), fork2 (block fed twice into the copy, loss on the 2nd copy = all-to-all).
Metrics on sampled blocks (teacher-forced history, 16 Euler steps): mode_err (per frame, vs the mode of its own sign),
pair_agree (frame j and j + 8 on one sign, chance 0.5), block_agree (all 8 pairs, chance 1/256).
run: CUDA_VISIBLE_DEVICES= venvs/yue2/bin/python vae/toy_fork.py --arm fork2"""
import argparse, json, math, sys, time, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.gated_nar import GatedNARLM, layout  # noqa: E402
from vae.renderer import t_embed  # noqa: E402

LAT, BLK, NB, HALF = 8, 16, 6, 8


def batch(B, P, gen):
    x = torch.empty(B, NB, BLK, LAT); u = torch.empty_like(x); m = torch.randn(B, LAT, generator=gen)
    for i in range(NB):
        if i: m = x[:, i - 1].abs().mean(1).roll(1, -1)
        s = (torch.randint(0, 2, (B, HALF, 1), generator=gen) * 2 - 1.0).repeat(1, BLK // HALF, 1)
        u[:, i] = m[:, None] + P; x[:, i] = s * u[:, i] + 0.1 * torch.randn(B, BLK, LAT, generator=gen)
    return x, u


def proj(tm, x, xp, isf):
    """GTimeMix projections of rows x [B,T,C] with token-shift sources xp -> r, w_raw, k, v, kk, a [B,T,H,N], g [B,T,C]"""
    B, T, C = x.shape; H, N = tm.h, tm.n; dx = xp - x; hv = lambda t: t.float().view(B, T, H, N)
    xr, xw, xk, xv, xa, xg = (x + dx * tm.mix[i] for i in range(6))
    r = tm.r(xr) + (tm.r_f(xr) if isf else 0); k, v = tm.k(xk), tm.v(xv)
    w_raw = tm.w2(torch.tanh(tm.w1(xw))); a = torch.sigmoid(tm.a2(tm.a1(xa))); g = tm.g2(torch.sigmoid(tm.g1(xg)))
    kk = F.normalize(hv(k * tm.k_k), dim=-1); k = k * (1 + (a - 1) * 0.5)
    return hv(r), hv(w_raw), hv(k), hv(v), kk, hv(a), g


def rec(r, w_raw, k, v, kk, a, S):
    """wkv7_ref from initial state S [B,H,N,N] -> y [B,T,H,N], state after every row [B,T,H,N,N]"""
    w = torch.exp(-math.exp(-0.5) * torch.sigmoid(w_raw)); ys, Ss = [], []
    for t in range(r.shape[1]):
        kt = kk[:, t]; sa = (S * kt.unsqueeze(-2)).sum(-1)
        S = S * w[:, t].unsqueeze(-2) - sa.unsqueeze(-1) * (kt * a[:, t]).unsqueeze(-2) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
        ys.append((S * r[:, t].unsqueeze(-2)).sum(-1)); Ss.append(S)
    return torch.stack(ys, 1), torch.stack(Ss, 1)


def tm_out(tm, y, g, isf):
    B, T, C = g.shape; y = tm.ln_x(y.reshape(B * T, C)).view(B, T, C) * g
    return tm.o(y) + (tm.o_f(y) if isf else 0)


class ForkNARLM(GatedNARLM):
    """same weights as GatedNARLM (iso-param + a copy embedding); F rows of block i run their own recurrence from a copy of each
    layer's S after A_i and write into it (k / v / w / a = the AR weights). passes = 2: the block is fed twice, output = copy 2.
    A rows never see the copies, so h is exact w.r.t. xt as in the gated model; the flow loss reaches A through the copied S."""
    def __init__(s, passes=2, **kw):
        super().__init__(**kw); s.passes = passes; d = s.bos.numel(); s.cp = nn.Parameter(torch.randn(passes, 1, d) * 0.02)

    def forward(s, z, xt, t, lay=None):
        B, NB_, _ = z.shape; K = xt.shape[1]; d = s.bos.numel(); L = s.passes * s.blk
        a = torch.cat([s.bos.expand(B, 1, -1), s.inp(z[:, :-1])], 1) + s.kind.weight[0]
        ct = s.tm(t_embed(t.reshape(-1).float(), 256)).view(B, K, 1, d)
        fr = s.fin(xt) + s.fpos + s.kind.weight[1] + ct
        f = torch.cat([fr + s.cp[p] for p in range(s.passes)], 2)                              # [B,K,L,d]
        sh = lambda u: F.pad(u, (0, 0, 1, 0))[:, :-1]                                            # previous A row (zeros at A_0)
        for b in s.blocks:
            tm = b.tm; H, N = tm.h, tm.n
            ua = b.ln1(a); *pa, ga = proj(tm, ua, sh(ua), False)
            ya, Sa = rec(*pa, a.new_zeros(B, H, N, N))                                          # Sa[:, i] = S after A_i
            uf = b.ln1_f(f, ct); upf = torch.cat([ua[:, :, None], uf[:, :, :-1]], 2)          # F_i,1 <- A_i, then causal
            *pf, gf = proj(tm, uf.reshape(B * K, L, d), upf.reshape(B * K, L, d), True)
            yf, _ = rec(*pf, Sa.reshape(B * K, H, N, N))                                         # fork: writes go to the copy
            a = a + tm_out(tm, ya, ga, False); f = f + tm_out(tm, yf, gf, True).view(B, K, L, d)
            u2 = b.ln2(a); a = a + b.ffn.step(u2, sh(u2))
            uf2 = b.ln2_f(f, ct); f = f + b.ffn.step(uf2, torch.cat([u2[:, :, None], uf2[:, :, :-1]], 2), uf2)
        return s.norm(a), s.out(s.norm_f(f[:, :, -s.blk:], ct))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="fork2", choices=["bi", "lattn", "fork1", "fork2"])
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="out/toy_fork/results.jsonl")
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(1000 + a.seed); P = torch.randn(BLK, LAT, generator=torch.Generator().manual_seed(7)) * 0.5
    kw = dict(latent=LAT, blk=BLK, d=64, n=3, head=32, lora=16)
    M = ForkNARLM(passes=int(a.arm[-1]), **kw) if a.arm.startswith("fork") else GatedNARLM(lattn=a.arm == "lattn", **kw)
    lay = layout(NB, BLK, range(NB))
    opt = torch.optim.AdamW(M.parameters(), lr=2e-3, weight_decay=0.0); t0 = time.time()
    for step in range(a.steps):
        x, _ = batch(a.bs, P, gen); e = torch.randn_like(x); t = torch.rand(a.bs, NB)
        xt = (1 - t[..., None, None]) * e + t[..., None, None] * x
        _, v = M(x.flatten(2), xt, t, lay); loss = (v - (x - e)).square().mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
        for g in opt.param_groups: g["lr"] = 2e-3 * min(1, (step + 1) / 100) * (0.1 + 0.9 * (1 - step / a.steps))
        if step % 250 == 0: print(f"{a.arm} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    M.eval(); res = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "final_loss": loss.item(),
                     "params": sum(p.numel() for p in M.parameters())}
    with torch.no_grad():
        x, u = batch(256, P, torch.Generator().manual_seed(99)); e0 = torch.randn(x.shape, generator=torch.Generator().manual_seed(5))
        ev = []                                                                                   # teacher-forced eval loss, fixed t grid
        for tv in (0.1, 0.3, 0.5, 0.7, 0.9):
            xt = (1 - tv) * e0 + tv * x; ev.append((M(x.flatten(2), xt, torch.full((256, NB), tv), lay)[1] - (x - e0)).square().mean().item())
        res["eval_loss"] = sum(ev) / len(ev)
        xs = e0.clone(); ts = torch.linspace(0, 1, 17)
        for k in range(16): xs = xs + (ts[k + 1] - ts[k]) * M(x.flatten(2), xs, ts[k].expand(256, NB), lay)[1]
        sg = torch.sign((xs * u).sum(-1, keepdim=True))
        res["mode_err"] = ((xs - sg * u).norm(dim=-1) / u.norm(dim=-1)).mean().item()
        sg = sg.squeeze(-1); pair = sg[..., :HALF] == sg[..., HALF:]
        res["pair_agree"] = pair.float().mean().item(); res["block_agree"] = pair.all(-1).float().mean().item()
    res["sec"] = round(time.time() - t0); print(json.dumps(res), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fo: fo.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
