#!/usr/bin/env python3
"""Latent statistics v1 vs v2 on 16 held-out 8 s crops: per-channel std spread, mean offset, posterior std, PCA effective
dimensionality, frame-to-frame similarity. Explains how hard the latent is to quantise (vae/tokenizer.py)."""
import sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.data import Crops, collate  # noqa: E402

it = iter(Crops(str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"), holdout=0, max_rows=1000, seed=9)); x = collate([next(it) for _ in range(16)])[0].cuda()
for tag, p in (("v1", "out/vae/v1/G_s200000.pt"), ("v2", "out/vae/v2/G_s60000.pt")):
    ck = torch.load(ROOT / p, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(v) for v in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): mu, lv = V.encode(x)
    mu = mu.float(); f = mu.reshape(-1, 64); sd = f.std(0); m = f.mean(0); ps = (0.5 * lv.float()).exp().reshape(-1, 64).mean(0)
    sv = torch.linalg.svdvals(f - m); ev = sv ** 2 / (sv ** 2).sum()
    print(f"{tag} channel std min/median/max {sd.min():.3f} {sd.median():.3f} {sd.max():.3f} | |mean| median {m.abs().median():.3f} | "
          f"posterior std median {ps.median():.4f} | PCA top4 {ev[:4].sum():.2f} top16 {ev[:16].sum():.2f} dims for 90% {int((ev.cumsum(0) < 0.9).sum()) + 1} | "
          f"frame-to-frame cos {torch.nn.functional.cosine_similarity(mu[:, 1:] - m, mu[:, :-1] - m, dim=-1).mean():.3f}")
