"""CPU toy for vae/gated_nar.py: does the gated NAR expert learn what needs F->A (read the past through S) and F->F (frames agree)?
Data: block i frames x_i,j = s_i (m_i + P_j) + 0.1 e, m_i = roll(mean_j x_i-1,j) (needs the previous block = F->A), s_i = +-1 random
per block, shared by its frames (bimodal; a frame that cannot see its neighbours picks its own sign = F->F).
Sampled with teacher-forced history; mode err = |x - s_hat u| / |u| per frame, agree = all frames of a block on one sign
(chance with independent frames 2 / 2^blk = 0.125 at blk 4).
Arms: bi (default), causal (backward shift mix_b frozen at 0), iso (no in-block shift), nohist (history zeroed = no F->A),
lattn (bi + per-layer local attention of F rows on block i-1's clean frames + own frames), lattn_iso (local attention, no shifts).
run: CUDA_VISIBLE_DEVICES= venvs/yue2/bin/python vae/toy_gated_nar.py --arm bi"""
import argparse, json, sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.gated_nar import GatedNARLM, layout  # noqa: E402

LAT, BLK, NB = 8, 4, 12


def batch(B, P, gen):
    x = torch.empty(B, NB, BLK, LAT); u = torch.empty_like(x)
    m = torch.randn(B, LAT, generator=gen)
    for i in range(NB):
        if i: m = x[:, i - 1].mean(1).roll(1, -1)
        s = torch.randint(0, 2, (B, 1, 1), generator=gen) * 2 - 1.0
        u[:, i] = m[:, None] + P; x[:, i] = s * u[:, i] + 0.1 * torch.randn(B, BLK, LAT, generator=gen)
    return x, u


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="bi", choices=["bi", "causal", "iso", "nohist", "lattn", "lattn_iso"])
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(1000 + a.seed); P = torch.randn(BLK, LAT, generator=torch.Generator().manual_seed(7)) * 0.5
    M = GatedNARLM(latent=LAT, blk=BLK, d=64, n=3, head=32, lora=16, lattn=a.arm.startswith("lattn"))
    if a.arm == "causal":
        for n, p in M.named_parameters():
            if "mix_b" in n: p.requires_grad_(False)
    lay = layout(NB, BLK, range(NB), iso=a.arm in ("iso", "lattn_iso"))
    opt = torch.optim.AdamW([p for p in M.parameters() if p.requires_grad], lr=2e-3, weight_decay=0.0)
    hist = (lambda x: torch.zeros_like(x)) if a.arm == "nohist" else (lambda x: x)
    t0 = time.time()
    for step in range(a.steps):
        x, _ = batch(a.bs, P, gen); e = torch.randn_like(x); t = torch.rand(a.bs, NB)
        xt = (1 - t[..., None, None]) * e + t[..., None, None] * x
        _, v = M(hist(x).flatten(2), xt, t, lay); loss = (v - (x - e)).square().mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
        for g in opt.param_groups: g["lr"] = 2e-3 * min(1, (step + 1) / 100) * (0.1 + 0.9 * (1 - step / a.steps))
        if step % 250 == 0: print(f"{a.arm} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    M.eval(); res = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "final_loss": loss.item()}
    with torch.no_grad():
        x, u = batch(256, P, torch.Generator().manual_seed(99)); xs = torch.randn_like(x); ts = torch.linspace(0, 1, 17)
        for k in range(16):
            xs = xs + (ts[k + 1] - ts[k]) * M(hist(x).flatten(2), xs, ts[k].expand(256, NB), lay)[1]
        s = torch.sign((xs * u).sum(-1, keepdim=True))
        res["mode_err"] = ((xs - s * u).norm(dim=-1) / u.norm(dim=-1)).mean().item()
        s = s.squeeze(-1); res["agree"] = (s == s[..., :1]).all(-1).float().mean().item()
    res["sec"] = round(time.time() - t0); print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
