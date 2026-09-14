#!/usr/bin/env python3
"""Extract MERT-v2-FullSong multi-layer features for inverse-tokenizer training (PLAN §8.3 R1).

Sources (any mix):
  * minted-corpus tracks  <dir>/<pid>/{latent.npy, semantic.npy}  — audio is reconstructed with YuE2-Vae.decode(latent)
    (audio.flac in that corpus *is* decode(latent), so nothing is lost and the 160 GB flac download is unnecessary)
  * own minted songs      <dir>/<id>/{audio.flac, semantic.npy}
  * real songs            <dir>/<id>/audio.flac (no tokens; used for the NAR-teacher term later)
Writes <id>/mert_L{layers}.npy as fp16 [K, T25, 1024] (25 Hz, linear-resampled like prep_real.py) next to the inputs.
Run inside venvs/yue2. usage: yue2_extract_mert.py --dirs data/yue2_minted/tracks data/yue2_minted_own [--layers 12,16,20,23]
"""
import argparse, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[1]
YUE2 = ROOT / "models/yue2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--layers", default="12,16,20,23")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--vae", default="YuE2-Vae")
    args = ap.parse_args(); dev = "cuda"; layers = [int(x) for x in args.layers.split(",")]; tag = "".join(f"_{l}" for l in layers)
    from transformers import AutoModel, AutoFeatureExtractor
    from yue2.modeling_vae import YuE2VAE
    proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
    mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()
    vae = None
    todo = []
    for d in args.dirs:
        for t in sorted(Path(d).iterdir()):
            if not t.is_dir() or (t / f"mert_L{tag}.npy").exists(): continue
            if (t / "audio.flac").exists() or (t / "latent.npy").exists(): todo.append(t)
    if args.limit: todo = todo[:args.limit]
    print(f"{len(todo)} tracks to extract, layers {layers}", flush=True); t0 = time.time()
    for i, t in enumerate(todo):
        try:
            if (t / "audio.flac").exists():
                a, sr = sf.read(t / "audio.flac", dtype="float32", always_2d=True); mono = a.mean(1)
                g = gcd(sr, 24000); m24 = resample_poly(mono, 24000 // g, sr // g).astype(np.float32) if sr != 24000 else mono
            else:
                if vae is None:
                    vae = YuE2VAE.from_pretrained(str(YUE2 / args.vae), decoder_only=True, device=dev, local_files_only=True)
                z = np.load(t / "latent.npy").astype(np.float32)
                if z.ndim != 2 or z.shape[1] != 64 or len(z) < 25: raise ValueError(f"bad latent {z.shape}")
                with torch.inference_mode():
                    audio = vae.decode_tiled(torch.tensor(z.T[None]).contiguous(), core_frames=750, halo_frames=16, output_device="cpu")
                mono = audio[0].float().clamp(-1, 1).mean(0).numpy(); m24 = resample_poly(mono, 1, 2).astype(np.float32)
            CH = 24000 * 30; chunks = [m24[s:s + CH] for s in range(0, len(m24), CH)]; chunks = [c for c in chunks if len(c) >= 24000]
            full = [c for c in chunks if len(c) == CH]; tail = [c for c in chunks if len(c) < CH]; feats = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for group in ([full] if full else []) + [[c] for c in tail]:
                    inp = {k: v.to(dev) for k, v in proc(group, sampling_rate=24000, return_tensors="pt").items()}
                    hs = mert(**inp, output_hidden_states=True).hidden_states
                    feats.append(torch.stack([hs[l].reshape(-1, 1024) for l in layers]))          # [K, B*T, 1024]
            Hm = torch.cat(feats, 1).float(); T25 = int(round(len(m24) / 24000 * 25))
            M = torch.nn.functional.interpolate(Hm.transpose(1, 2), size=T25, mode="linear", align_corners=False).transpose(1, 2)
            np.save(t / f"mert_L{tag}.npy", M.half().cpu().numpy())
            if i % 20 == 0 or i == len(todo) - 1:
                print(f"  [{i + 1}/{len(todo)}] {t.name} T25 {T25} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  FAIL {t}: {e}", flush=True)
    print("EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
