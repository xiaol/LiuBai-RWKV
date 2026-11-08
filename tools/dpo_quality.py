#!/usr/bin/env python3
"""Sound quality of the DPO gate outputs vs the live-WKV base (same 96 causal crops, out/clm_eval/*):
MERT-v2 cos to the original, LTAS dB distance to the original, L/R corr, RMS dB, crest dB, clip %. Paired bootstrap of the mean
diff vs base. Run in venvs/yue2. Writes out/clm_eval/dpo_quality.json.
usage: dpo_quality.py"""
import json, sys, numpy as np, soundfile as sf, torch
from math import gcd
from pathlib import Path
from scipy.signal import resample_poly
from transformers import AutoModel, AutoFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]; E = ROOT / "out/clm_eval"; dev = "cuda"
BASE = "e1_gated_s0_wkv_20k_c96_cfg{}"; ARMS = {"base": BASE, "dpo_a": "dpo_a_onpol_1500_c96_cfg{}", "dpo_b": "dpo_b_mixed_1500_c96_cfg{}"}
RUNS = [("orig", BASE.format(1), "A_original"), ("vae", BASE.format(1), "B_vae")]
for c in (1, 3):
    for k, pat in ARMS.items(): RUNS += [(f"{k}_c{c}", pat.format(c), "C_own"), (f"{k}_c{c}_swap", pat.format(c), "C_swap")]
MP = ROOT / "models/yue2/MERT-v2-FullSong"
fe = AutoFeatureExtractor.from_pretrained(str(MP), trust_remote_code=True)
mert = AutoModel.from_pretrained(str(MP), trust_remote_code=True).to(dev).eval(); win = torch.hann_window(2048, device=dev)


def stats(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    m = x.mean(1); g = gcd(sr, 24000); m24 = resample_poly(m, 24000 // g, sr // g).astype(np.float32)
    inp = {k: v.to(dev) for k, v in fe(m24, sampling_rate=24000, return_tensors="pt").items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        e = torch.nn.functional.normalize(mert(**inp).last_hidden_state[0].float().mean(0), dim=0)
    S = torch.stft(torch.tensor(m, device=dev), 2048, 512, window=win, return_complex=True).abs().pow(2).mean(1)
    rms = float(np.sqrt((x ** 2).mean()) + 1e-9); pk = float(np.abs(x).max() + 1e-9)
    return dict(emb=e, ltas=10 * torch.log10(S + 1e-10), lr=float(np.corrcoef(x[:, 0], x[:, -1])[0, 1]) if x.shape[1] > 1 else 1.0,
                rms=20 * np.log10(rms), crest=20 * np.log10(pk / rms), clip=float((np.abs(x) > 0.99).mean()))


n = 96; res = {k: [] for k, _, _ in RUNS}
for i in range(n):
    s = {k: stats(E / d / f"{i:03d}" / f"{f}.flac") for k, d, f in RUNS}; o = s["orig"]
    for k in s:
        res[k].append(dict(cos=float(o["emb"] @ s[k]["emb"]), ltas=float((s[k]["ltas"] - o["ltas"]).abs().mean()), lr=s[k]["lr"],
                           rms=s[k]["rms"], crest=s[k]["crest"], clip=s[k]["clip"]))
    if i % 24 == 0: print(i, flush=True)
print(f"{'':14s} {'cosOrig':>8s} {'LTAS dB':>8s} {'L/R':>6s} {'RMS dB':>7s} {'crest':>6s} {'clip%':>6s}")
for k in res:
    a = {m: np.array([r[m] for r in res[k]]) for m in res[k][0]}
    print(f"{k:14s} {a['cos'].mean():8.3f} {a['ltas'].mean():8.2f} {a['lr'].mean():6.3f} {a['rms'].mean():7.1f} {a['crest'].mean():6.1f} {a['clip'].mean() * 100:6.3f}")
rng = np.random.default_rng(0)
for c in (1, 3):
    for arm in ("dpo_a", "dpo_b"):
        for m in ("cos", "ltas", "crest", "rms"):
            d = np.array([res[f"{arm}_c{c}"][i][m] - res[f"base_c{c}"][i][m] for i in range(n)])
            bs = [d[rng.integers(0, n, n)].mean() for _ in range(4000)]
            print(f"cfg{c} {arm} - base {m:5s}: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]")
json.dump(res, open(E / "dpo_quality.json", "w"))
