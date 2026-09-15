#!/usr/bin/env python3
"""Prepare REAL Suno songs for teacher-based training (PLAN §8.3 R1 flow-loss term and R2 NAR LoRA).

From a suno-94k shard tar, per song writes data/yue2_real/<shard>/<id>/:
  mert_L_12_16_20_23.npy  fp16 [4, T25, 1024]   MERT-v2-FullSong layers 12/16/20/23 at 25 Hz
  latent_true.npy         fp16 [T25, 64]         YuE2-Vae.encode(48 kHz stereo original)
  prefix.npy              int64                   cot=off YuE2 text prefix from the song's caption + lyrics
  meta.json               id, title, duration, caption, lyrics
Songs 30-300 s, non-instrumental unless --keep-instrumental. Run inside venvs/yue2 (one GPU, ~10 GiB).
usage: yue2_prep_real.py --shard 0 [--n 0] [--min-sec 30] [--max-sec 300]
"""
import argparse, io, json, tarfile, time
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
YUE2 = ROOT / "models/yue2"
LAYERS = [12, 16, 20, 23]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--min-sec", type=float, default=30); ap.add_argument("--max-sec", type=float, default=300)
    ap.add_argument("--keep-instrumental", action="store_true"); ap.add_argument("--vae", default="YuE2-Vae")
    args = ap.parse_args(); dev = "cuda"
    out = ROOT / f"data/yue2_real/shard-{args.shard:05d}"; out.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(l) for l in open(ROOT / f"data/meta/suno94k/shard-{args.shard:05d}.jsonl")]
    songs = [m for m in meta if not m.get("error") and args.min_sec <= m["duration"] <= args.max_sec
             and (args.keep_instrumental or ("[instrumental]" not in m["lyrics"].lower() and len(m["lyrics"]) > 50))]
    if args.n: songs = songs[:args.n]
    songs = [m for m in songs if not (out / m["id"] / "prefix.npy").exists()]
    print(f"{len(songs)} songs to prepare from shard {args.shard}", flush=True)
    from transformers import AutoModel, AutoFeatureExtractor
    from yue2.modeling_vae import YuE2VAE
    from yue2.protocol import SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
    mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()
    vae = YuE2VAE.from_pretrained(str(YUE2 / args.vae), decoder_only=False, device=dev, local_files_only=True)
    tok = YuE2TextTokenizer(str(YUE2 / "YuE2-3B/qwen.tiktoken"))
    tar = tarfile.open(ROOT / f"data/raw/suno94k/suno-various-94k-{args.shard:05d}.tar")
    names = set(tar.getnames()); t0 = time.time(); done = 0
    for m in songs:
        sid = m["id"]; fn = next((f"{sid}{s}" for s in (".mp3", ".wav", ".flac") if f"{sid}{s}" in names), None)
        if fn is None: print(f"  missing audio {sid}", flush=True); continue
        try:
            a, sr = sf.read(io.BytesIO(tar.extractfile(fn).read()), dtype="float32", always_2d=True)
            if a.shape[1] == 1: a = np.repeat(a, 2, 1)
            g = gcd(sr, 48000); st48 = resample_poly(a, 48000 // g, sr // g, axis=0).astype(np.float32) if sr != 48000 else a
            g = gcd(sr, 24000); m24 = resample_poly(a.mean(1), 24000 // g, sr // g).astype(np.float32)
            CH = 24000 * 30; chunks = [m24[s:s + CH] for s in range(0, len(m24), CH)]; chunks = [c for c in chunks if len(c) >= 24000]
            full = [c for c in chunks if len(c) == CH]; tail = [c for c in chunks if len(c) < CH]; feats = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for group in ([full] if full else []) + [[c] for c in tail]:
                    inp = {k: v.to(dev) for k, v in proc(group, sampling_rate=24000, return_tensors="pt").items()}
                    hs = mert(**inp, output_hidden_states=True).hidden_states
                    feats.append(torch.stack([hs[l].reshape(-1, 1024) for l in LAYERS]))
            Hm = torch.cat(feats, 1).float(); T25 = int(round(len(m24) / 24000 * 25))
            M = torch.nn.functional.interpolate(Hm.transpose(1, 2), size=T25, mode="linear", align_corners=False).transpose(1, 2).half().cpu().numpy()
            zs = []
            with torch.inference_mode():
                for s in range(0, len(st48), 48000 * 60):
                    seg = st48[s:s + 48000 * 60]
                    if len(seg) < 1920: break
                    zs.append(vae.encode(torch.tensor(seg.T[None], device=dev))[0].T.float().cpu())
            Z = torch.cat(zs, 0).numpy(); n = min(M.shape[1], len(Z))
            style = " ".join(m["caption"].split())[:1500]; lyrics = m["lyrics"].strip() or "[instrumental]"
            prefix = token_prefixes(SongRequest(style=style, lyrics=lyrics, cot="off", seed=1, id=sid), tok)
            d = out / sid; d.mkdir(exist_ok=True)
            np.save(d / "mert_L_12_16_20_23.npy", M[:, :n]); np.save(d / "latent_true.npy", Z[:n].astype(np.float16))
            np.save(d / "prefix.npy", np.array(prefix, dtype=np.int64))
            json.dump(dict(id=sid, title=m["title"], duration=m["duration"], caption=m["caption"], lyrics=m["lyrics"], frames=int(n)), open(d / "meta.json", "w"), indent=1)
            done += 1
            if done % 25 == 1: print(f"  [{done}/{len(songs)}] {sid} {m['duration']:.0f}s frames {n} prefix {len(prefix)} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  FAIL {sid}: {e}", flush=True)
    print(f"PREP DONE {done} songs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
