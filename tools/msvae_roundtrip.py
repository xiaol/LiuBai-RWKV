#!/usr/bin/env python3
"""MSVAE (ASLP-lab, arXiv 2610.17187) full-song round trip on the gate songs, in the layout tools/yue2_roundtrip_eval.py scores.

A_original.flac (48 kHz) -> 24 kHz stereo -> MSVAE encode (posterior mean, the release's 128-frame / 32-overlap windows)
-> [optional latent noise, same definition as vae/model.py s2: per-channel time variance x --noise] -> BigVGAN decode
(128 / 32 windows) -> 48 kHz -> C_roundtrip_<tag>.flac, plus latent_<tag>.npy [T, 64] at 25 Hz.
Run with /root/envs/msvae-venv/bin/python; code at /models/msvae/code (inference-only GitHub release), weights in /models/msvae.
usage: msvae_roundtrip.py --tag msvae [--noise 0.3] [--only id1,id2]
"""
import argparse, sys, time
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch, yaml
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]


def rs(x, a, b):
    g = gcd(a, b); return resample_poly(x, b // g, a // g, axis=0).astype(np.float32)


def load_msvae(code, ckpt, muq, dev):
    sys.path.insert(0, str(code))
    from src.model import DAC
    from src.infer.infer import load_checkpoint
    cfg = yaml.safe_load(open(Path(code) / "src/config/vae/config.yaml"))
    kw = {k[4:]: v for k, v in cfg.items() if k.startswith("DAC.")}
    kw["bigvgan_conf"] = str(Path(code) / kw["bigvgan_conf"]); kw["muq_model_name"] = str(muq)
    return load_checkpoint(DAC(**kw).float(), str(ckpt), dev).eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(ROOT / "out/yue2_roundtrip_gate20"))
    ap.add_argument("--code", default="/models/msvae/code"); ap.add_argument("--ckpt", default="/models/msvae/weights/model.pt")
    ap.add_argument("--muq", default="/models/msvae/muq"); ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--only", default=""); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    M = load_msvae(args.code, args.ckpt, args.muq, dev)
    from src.infer.infer import encode_audio, decode_audio, preprocess
    g = torch.Generator(device=dev).manual_seed(args.seed); only = {x for x in args.only.split(",") if x}
    for d in sorted(p for p in Path(args.root).iterdir() if (p / "A_original.flac").exists() and (not only or p.name in only)):
        x, sr = sf.read(d / "A_original.flac", dtype="float32", always_2d=True)
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        x24 = rs(x, sr, 24000); n = len(x24); t0 = time.time()
        with torch.inference_mode():
            a = preprocess(torch.from_numpy(x24.T.copy()), M.encoder_rates, min_length=128 * M.hop_length)[None].to(dev)
            z = encode_audio(a, M, True, 32, 128, "mean")                       # [1, 64, T]
            if args.noise > 0:
                z = z + torch.randn(z.shape, device=dev, generator=g) * (args.noise * z.var(dim=2, keepdim=True)).sqrt()
            y = decode_audio(z, M, True, 32, 128)[0, :, :round(n * 2)].float().cpu().numpy().T
        sf.write(d / f"C_roundtrip_{args.tag}.flac", np.clip(y, -1, 1), 48000, subtype="PCM_16")
        zn = z[0].T.float().cpu().numpy(); np.save(d / f"latent_{args.tag}.npy", zn.astype(np.float16))
        print(d.name, f"{n / 24000:.0f}s latent {zn.shape} std {zn.std(0).mean():.3f} |mu| {np.abs(zn.mean(0)).mean():.3f} "
              f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
