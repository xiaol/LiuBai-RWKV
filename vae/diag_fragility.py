#!/usr/bin/env python3
"""Is the VAE latent space too fragile for a generator? On held-out 8 s crops, MMS letter CER of
  decode(mu + eps), eps ~ N(0, s^2 * var_c(mu)) for relative noise s^2 in {0, 0.02, 0.05, 0.1, 0.2, 0.35}
  decode(mu blurred over time) (a generator's typical error is smooth, not white)
  decode(posterior sample mu + sigma*eps) (what the decoder saw in training)
and the posterior std relative to the latent std (how much noise the decoder was trained to tolerate).
Rendered latents sit at NMSE ~0.31-0.36 from mu: if CER at s^2 ~ 0.3 is near the renders' 0.85, the latent space is the bottleneck."""
import sys, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.mms_judge import MMSJudge  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402

dev = "cuda"; torch.manual_seed(0)
ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else ROOT / "out/vae/v1/G_s200000.pt", map_location="cpu"); a = ck["args"]
V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
J = MMSJudge().to(dev)
vd = Crops(str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"), holdout=0, max_rows=1000, seed=4242); it = iter(vd)
res = {}; post = []
for _ in range(4):
    wav, tg, lens, ok, cf = collate([next(it) for _ in range(8)]); x = wav.to(dev); S = x.shape[-1]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16): mu, lv = V.encode(x)
        mu, lv = mu.float(), lv.float(); var = mu.var(dim=(0, 1), keepdim=True); post.append(((0.5 * lv).exp().mean(dim=(0, 1)) / var.squeeze().sqrt()).mean().item())
        cands = {"orig": None}
        for s2 in (0.0, 0.02, 0.05, 0.1, 0.2, 0.35): cands[f"white {s2}"] = mu + torch.randn_like(mu) * (s2 * var).sqrt()
        k = torch.tensor([0.25, 0.5, 0.25], device=dev).view(1, 1, 3); bl = F.conv1d(F.pad(mu.transpose(1, 2).reshape(-1, 1, mu.shape[1]), (1, 1), mode="replicate"), k)
        cands["blur3"] = bl.view(mu.shape[0], mu.shape[2], -1).transpose(1, 2)
        cands["posterior"] = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        mt, ml = J.targets(tg, lens); offs = torch.cat([torch.zeros(1, dtype=torch.long), ml.cumsum(0)])
        for name, z in cands.items():
            if z is None: y = x
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16): y = V.decode(z).float()
                y = F.pad(y, (0, max(0, S - y.shape[-1])))[..., :S]
            lp = J(y); fps = lp.shape[1] / 8.0; e = n = 0
            for j in ok.nonzero().flatten().tolist():
                p = lp[j, :int(cf[j] / 50 * fps)].argmax(-1).tolist(); hyp = [t for i, t in enumerate(p) if t != 0 and (i == 0 or t != p[i - 1])]
                ref = mt[offs[j]:offs[j + 1]].tolist(); e += edit(hyp, ref); n += len(ref)
            nm = 0.0 if z is None else ((z - mu) ** 2).mean().item() / var.mean().item()
            r = res.setdefault(name, [0, 0, 0, 0]); r[0] += e; r[1] += n; r[2] += nm; r[3] += 1
print(f"posterior std / latent std: {sum(post) / len(post):.4f}")
for k, (e, n, nm, c) in res.items(): print(f"{k:14s} latent NMSE {nm / c:.3f}  MMS letter CER {e / max(n, 1):.3f}")
