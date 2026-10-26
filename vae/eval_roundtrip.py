#!/usr/bin/env python3
"""Full-song VAE round trip on the gate songs, in the layout tools/yue2_roundtrip_eval.py scores.

For each song dir under --root (made by tools/yue2_roundtrip.py: A_original.flac at 48 kHz): original -> 24 kHz -> VAE encode
(posterior mean, whole song, so the bidirectional RWKV sees the full context) -> decode -> upsampled to 48 kHz ->
C_roundtrip_<tag>.flac. Also writes C_roundtrip_bw24.flac once (original band-limited to 24 kHz and back): the bandwidth-only
reference, since every 24 kHz pilot loses everything above 12 kHz by construction.
usage: eval_roundtrip.py --ckpt out/vae/pilotB_ctc/G_s40000.pt --tag vaeB [--root out/yue2_roundtrip_gate20]
"""
import argparse, sys
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402


def rs(x, a, b):
    g = gcd(a, b); return resample_poly(x, b // g, a // g, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(ROOT / "out/yue2_roundtrip_gate20"))
    ap.add_argument("--chunk", type=float, default=0, help="> 0: encode/decode in windows of this many seconds (training crop length), "
                    "cross-faded over --overlap s; 0 = whole song in one pass")
    ap.add_argument("--noise", type=float, default=0.0, help="relative latent noise power added before decoding (generator-error proxy)")
    ap.add_argument("--overlap", type=float, default=1.0); ap.add_argument("--only", default="", help="comma list of song ids")
    args = ap.parse_args(); dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu"); a = ck["args"]
    from vae.model import load_vae; G = load_vae(args.ckpt, dev)
    def run(seg):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            s2 = torch.full((1,), args.noise, device=dev) if args.noise > 0 else None
            y, mu, _, _ = G(torch.tensor(seg.T[None], device=dev), sample=False, s2=s2); return y[0].float().cpu().numpy().T, mu.shape
    def chunked(x):
        W = int(args.chunk * 24000) // G.hop * G.hop; O = int(args.overlap * 24000) // G.hop * G.hop; H = W - O
        out = np.zeros_like(x); wsum = np.zeros((len(x), 1), np.float32); fade = np.ones(W, np.float32)
        fade[:O] = np.linspace(0, 1, O); fade[W - O:] = np.linspace(1, 0, O)
        for s0 in range(0, max(1, len(x) - O), H):
            seg = x[s0:s0 + W]
            if len(seg) < W: seg = np.pad(seg, ((0, W - len(seg)), (0, 0)))
            y, _ = run(seg); n = min(W, len(x) - s0); f = fade[:n].copy()
            if s0 == 0: f[:O] = 1
            if s0 + W >= len(x): f[max(0, n - O):] = 1
            out[s0:s0 + n] += y[:n] * f[:, None]; wsum[s0:s0 + n] += f[:, None]
        return out / np.maximum(wsum, 1e-6)
    only = {x for x in args.only.split(",") if x}
    for d in sorted(p for p in Path(args.root).iterdir() if (p / "A_original.flac").exists() and (not only or p.name in only)):
        x, sr = sf.read(d / "A_original.flac", dtype="float32", always_2d=True)
        x24 = rs(x, sr, 24000); S = len(x24) - len(x24) % G.hop
        if not (d / "C_roundtrip_bw24.flac").exists(): sf.write(d / "C_roundtrip_bw24.flac", rs(x24, 24000, 48000), 48000, subtype="PCM_16")
        if args.chunk > 0: y = chunked(x24[:S]); mu = torch.zeros(1, S // G.hop, 64)
        else: y, _ = run(x24[:S]); mu = torch.zeros(1, S // G.hop, 64)
        sf.write(d / f"C_roundtrip_{args.tag}.flac", np.clip(rs(y, 24000, 48000), -1, 1), 48000, subtype="PCM_16")
        print(d.name, f"{S / 24000:.0f}s latent {tuple(mu.shape)}", flush=True)


if __name__ == "__main__":
    main()
