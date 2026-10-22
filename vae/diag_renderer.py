#!/usr/bin/env python3
"""Does the flow renderer use its condition? On held-out 8 s crops (FSQ renderer):
  mel dB of decode(sample(c)) vs original, with c = true codes / codes of a different crop (shuffled) / null (CFG token),
  for 8, 24, 64 Euler steps; plus decode(mu) (VAE ceiling) and the per-frame std ratio sample/mu (over-smoothing).
If true ~= shuffled, the renderer ignores the tokens; if steps matter a lot, the sampler is the bottleneck."""
import sys, math, numpy as np, torch, torch.nn.functional as F, torchaudio
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.renderer import Renderer  # noqa: E402
from vae.data import Crops, collate  # noqa: E402

dev = "cuda"; ck = torch.load("out/vae/v1/G_s200000.pt", map_location="cpu"); a = ck["args"]
V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
rk = torch.load(sys.argv[1], map_location="cpu"); ra = rk["args"]
tk = torch.load(ra["tok"], map_location="cpu"); TK = SemTokenizer(); TK.load_state_dict(tk["T"]); TK = TK.to(dev).eval()
R = Renderer(d=ra["d"], n=ra["n"], vocab=TK.fsq.n_codes); R.load_state_dict(rk["R"]); R = R.to(dev).eval()
mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80, power=1.0).to(dev)
mdb = lambda y, x: ((torch.log(mel(y.mean(1)) + 1e-5) - torch.log(mel(x.mean(1)) + 1e-5)).abs().mean() * 20 / math.log(10)).item()
vd = Crops(str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"), holdout=0, max_rows=1000, seed=12345); it = iter(vd)
x = collate([next(it) for _ in range(16)])[0].to(dev); S = x.shape[-1]
dec = lambda z: F.pad(V.decode(z.float()).float(), (0, max(0, S - V.decode(z.float()).shape[-1])))[..., :S]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    mu = V.encode(x)[0].float(); c = TK.encode(mu)[1]
    print(f"VAE ceiling mel {mdb(dec(mu), x):.2f} dB")
    for steps in (8, 24, 64):
        for name, cc in (("true", c), ("shuffled", c.roll(1, 0))):
            z = R.sample(cc, T=mu.shape[1], steps=steps).float()
            print(f"steps {steps:2d} cond {name:8s} mel {mdb(dec(z), x):.2f} dB  std ratio {(z.std(1) / mu.std(1)).mean().item():.3f}  latent nmse {((z - mu) ** 2).mean().item() / mu.var().item():.3f}")
    ch = R.cond(c); null = R.null.expand_as(ch)
    v_t = R(torch.randn_like(mu), torch.full((16,), 0.5, device=dev), ch); v_n = R(torch.randn_like(mu), torch.full((16,), 0.5, device=dev), null)
    print(f"velocity |cond - null| / |cond| at t=0.5: {((v_t - v_n).norm() / v_t.norm()).item():.3f}")
