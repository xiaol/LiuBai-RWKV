#!/usr/bin/env python3
"""Token round trip on the gate songs: original -> 24 kHz -> VAE encoder (mu) -> SemTokenizer FSQ codes -> tokenizer decoder
(latent regression) -> VAE decoder -> 48 kHz C_roundtrip_<tag>.flac, scored by tools/yue2_roundtrip_eval.py like every other
variant. The tokenizer decoder is a plain regressor (no generative renderer yet, Phase C), so this is a lower bound on what a
renderer can do from the codes; the lyric WER is the number to compare with the YuE2-space token round trip (R1-a 0.620).
Also writes tokens_<tag>.npy (int16 codes @12.5 Hz).
usage: eval_token_roundtrip.py --vae out/vae/v1/G_s200000.pt --tok out/tok/b1_ctc1/last.pt --tag tokb1 [--chunk 30]
"""
import argparse, sys
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402


def rs(x, a, b):
    g = gcd(a, b); return resample_poly(x, b // g, a // g, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vae", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(ROOT / "out/yue2_roundtrip_gate20")); ap.add_argument("--chunk", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=2.0); args = ap.parse_args(); dev = "cuda"
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a.get("mults", "1,2,4,8,16,16").split(",")), d=a["d"], n_rwkv=a["n_rwkv"])
    V.load_state_dict(ck["G"]); V = V.to(dev).eval()
    tk = torch.load(args.tok, map_location="cpu"); nq = tk["args"].get("n_q", 1)
    T = SemTokenizer(levels=tuple(int(x) for x in tk["args"]["levels"].split(",")), n_q=nq)
    T.load_state_dict(tk["T"]); T = T.to(dev).eval()

    @torch.inference_mode()
    def run(seg):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, _ = V.encode(torch.tensor(seg.T[None], device=dev)); q, ids = T.encode(mu.float()); z = T.decode(q)
            y = V.decode(z)[0].float()
        S = seg.shape[0]; y = torch.nn.functional.pad(y, (0, max(0, S - y.shape[-1])))[..., :S]
        return y.cpu().numpy().T, ids[0].cpu().numpy()

    for d in sorted(p for p in Path(args.root).iterdir() if (p / "A_original.flac").exists()):
        x, sr = sf.read(d / "A_original.flac", dtype="float32", always_2d=True)
        x24 = rs(x, sr, 24000); x24 = x24[:len(x24) - len(x24) % V.hop]
        W = int(args.chunk * 24000) // V.hop * V.hop; O = int(args.overlap * 24000) // V.hop * V.hop; H = W - O
        out = np.zeros_like(x24); wsum = np.zeros((len(x24), 1), np.float32); toks = np.zeros((len(x24) // V.hop,) + ((nq,) if nq > 1 else ()), np.int32)   # int16 overflowed for 122,880 codes
        fade = np.ones(W, np.float32); fade[:O] = np.linspace(0, 1, O); fade[W - O:] = np.linspace(1, 0, O)
        for s0 in range(0, max(1, len(x24) - O), H):
            seg = x24[s0:s0 + W]; n = len(seg)
            if n < W: seg = np.pad(seg, ((0, W - n), (0, 0)))
            y, ids = run(seg); f = fade[:n].copy()
            if s0 == 0: f[:O] = 1
            if s0 + W >= len(x24): f[max(0, n - O):] = 1
            out[s0:s0 + n] += y[:n] * f[:, None]; wsum[s0:s0 + n] += f[:, None]
            k0 = s0 // V.hop; k1 = min(len(toks), k0 + n // V.hop); toks[k0:k1] = ids[:k1 - k0]
        y = out / np.maximum(wsum, 1e-6)
        sf.write(d / f"C_roundtrip_{args.tag}.flac", np.clip(rs(y, 24000, 48000), -1, 1), 48000, subtype="PCM_16")
        np.save(d / f"tokens_{args.tag}.npy", toks)
        print(d.name, f"{len(x24) / 24000:.0f}s tokens {len(toks)} unique {len(np.unique(toks))}", flush=True)


if __name__ == "__main__":
    main()
