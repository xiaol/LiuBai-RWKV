#!/usr/bin/env python3
"""Qwen3-ASR-1.7B transcripts of each timed lyric line of tools/gen_song.py songs (the WildSongBench judge).
Runs in /root/envs/qwen-asr-venv (qwen-asr pins transformers 4.57.6; no uroman there). Letter CER is computed afterwards by
tools/score_song.py in venvs/yue2.

Each line window is [t0 - pre, min(t1 + post, t0 + 30)] (with timestamps the cursor may enter a line at t0 - 0.16 s and not
before, so the line should be sung inside it), transcribed on its own, language auto-detected.
usage: /root/envs/qwen-asr-venv/bin/python tools/transcribe_song_qwen.py DIR [DIR ...]   (writes DIR/asr_qwen.json)
"""
import argparse, glob, json
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from qwen_asr import Qwen3ASRModel


def text(r):
    if hasattr(r, "text"): return (r.text or "").strip()
    return ((r.get("text") or "") if isinstance(r, dict) else str(r)).strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--model", default="/models/rwkv-music/models/Qwen3-ASR-1.7B")
    ap.add_argument("--pre", type=float, default=0.3); ap.add_argument("--post", type=float, default=0.5); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    asr = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device, max_new_tokens=256,
                                        max_inference_batch_size=args.bs)
    for d in map(Path, args.dirs):
        songs = sorted((s for p in sorted(glob.glob(str(d / "gen*.json"))) for s in json.load(open(p, encoding="utf-8"))["songs"]), key=lambda s: s["k"])
        out = []
        for s in songs:
            y, sr = sf.read(d / s["file"], dtype="float32", always_2d=True); y = y.mean(1)
            g = gcd(sr, 16000); y = resample_poly(y, 16000 // g, sr // g).astype(np.float32)
            segs = [y[int(max(0, l["t0"] - args.pre) * 16000):int(min(l["t1"] + args.post, l["t0"] + 30) * 16000)] for l in s["lines"]]
            segs = [x if len(x) >= 8000 else np.pad(x, (0, 8000 - len(x))) for x in segs]
            hyp = [text(r) for r in asr.transcribe(audio=[(x, 16000) for x in segs])] if segs else []
            out.append(dict(k=s["k"], hyp=hyp)); print(f"{d.name} {s['k']:3d} {len(hyp)} lines", flush=True)
        json.dump(dict(model=args.model, pre=args.pre, post=args.post, songs=out), open(d / "asr_qwen.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
