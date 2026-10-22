#!/usr/bin/env python3
"""Regression-to-the-mean check for the tokenizer decoder: per-channel std ratio rec/mu, frame-to-frame change (oversmoothing),
and audio from decode(mu) / decode(rec) / decode(rec rescaled to mu's per-channel mean+std). 6 gate songs x 3 chunks of 30 s."""
import sys, numpy as np, soundfile as sf, torch, torchaudio
from math import gcd
from pathlib import Path
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402

dev = "cuda"; ck = torch.load(sys.argv[1], map_location="cpu"); a = ck["args"]
V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
tk = torch.load(sys.argv[2], map_location="cpu"); T = SemTokenizer(); T.load_state_dict(tk["T"]); T = T.to(dev).eval()
mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80, power=1.0).to(dev)
mdb = lambda y, x: ((torch.log(mel(y.mean(0)) + 1e-5) - torch.log(mel(x.mean(0)) + 1e-5)).abs().mean() * 20 / np.log(10)).item()
dz = lambda z: (z[0, 1:] - z[0, :-1]).pow(2).mean().sqrt().item()
W = 30 * 24000; R = {}
for d in sorted(p for p in (ROOT / "out/yue2_roundtrip_gate20").iterdir() if (p / "A_original.flac").exists())[:6]:
    x, sr = sf.read(d / "A_original.flac", dtype="float32", always_2d=True); g = gcd(sr, 24000); x = resample_poly(x, 24000 // g, sr // g, axis=0).astype(np.float32)
    for s0 in list(range(0, len(x) - W, W * 2))[:3]:
        xc = torch.tensor(x[s0:s0 + W].T, device=dev)
        with torch.inference_mode():
            mu = V.encode(xc[None])[0].float(); q, _ = T.encode(mu); rec = T.decode(q).float()
            ratio = rec[0].std(0) / mu[0].std(0)
            resc = (rec - rec.mean(1, keepdim=True)) / rec.std(1, keepdim=True) * mu.std(1, keepdim=True) + mu.mean(1, keepdim=True)
            for k, z in (("mu", mu), ("rec", rec), ("rec_rescaled", resc)):
                y = torch.nn.functional.pad(V.decode(z)[0].float(), (0, W))[..., :W]
                R.setdefault(f"mel_{k}", []).append(mdb(y, xc)); R.setdefault(f"lr_{k}", []).append(float(torch.corrcoef(y)[0, 1])); R.setdefault(f"dz_{k}", []).append(dz(z))
            R.setdefault("std_ratio_mean", []).append(ratio.mean().item()); R.setdefault("std_ratio_min", []).append(ratio.min().item())
for k, v in R.items(): print(f"{k:16s} {np.mean(v):.3f}")
