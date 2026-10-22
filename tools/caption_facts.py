#!/usr/bin/env python3
"""Caption stage B: measured facts per song (no language model involved).

  sections  SongFormer (ASLP-lab, /models/caption/SongFormer) on 24 kHz mono -> [{start, end, label}]
            label in intro/verse/chorus/bridge/inst/outro/silence/...; LRC lines are attached to the section holding their t0
  tempo     librosa beat tracker on the full mix -> bpm (rounded), beat-period stability
  key       Krumhansl-Schmuckler on mean chroma (CQT) -> "A minor" + correlation margin as confidence
  dynamics  per-section RMS in dB relative to the song median (lets the fuser say "chorus lifts", "bridge strips down")
One row per song -> <out>. Run in venvs/yue2 with the SongFormer extras:
  HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=venvs/sf_extra:/models/caption/SongFormer SONGFORMER_LOCAL_DIR=/models/caption/SongFormer \
    CUDA_VISIBLE_DEVICES=0 venvs/yue2/bin/python tools/caption_facts.py --manifest M.jsonl --out F.jsonl
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf, librosa

SF_DIR = "/models/caption/SongFormer"
SR = 24000
NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load24(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    return librosa.resample(a.mean(1), orig_sr=sr, target_sr=SR) if sr != SR else a.mean(1)


def key_of(y):
    chroma = librosa.feature.chroma_cqt(y=y[::2], sr=SR // 2).mean(1)
    scores = [(np.corrcoef(np.roll(prof, k), chroma)[0, 1], f"{NOTES[k]} {mode}") for prof, mode in ((MAJ, "major"), (MIN, "minor")) for k in range(12)]
    scores.sort(reverse=True)
    return scores[0][1], round(float(scores[0][0] - scores[1][0]), 3)


def tempo_of(y):
    onset = librosa.onset.onset_strength(y=y, sr=SR)
    bpm, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR)
    ibi = np.diff(librosa.frames_to_time(beats, sr=SR))
    stab = float(1 - np.std(ibi) / np.mean(ibi)) if len(ibi) > 4 else 0.0
    return round(float(np.atleast_1d(bpm)[0])), round(stab, 3)


def cpu_facts(y):
    key, key_conf = key_of(y); bpm, stab = tempo_of(y)
    return dict(key=key, key_conf=key_conf, bpm=bpm, beat_stability=stab)


def section_dynamics(y, secs):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1200)[0]  # 20 Hz
    db = 20 * np.log10(rms + 1e-6); med = float(np.median(db))
    for s in secs:
        seg = db[int(s["start"] * 20):max(int(s["start"] * 20) + 1, int(s["end"] * 20))]
        s["rel_db"] = round(float(np.mean(seg)) - med, 1) if len(seg) else 0.0
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda", help="cpu works too (sample-sized runs while the GPUs are busy)")
    args = ap.parse_args()
    import torch
    from transformers import AutoModel
    os.environ.setdefault("SONGFORMER_LOCAL_DIR", SF_DIR)
    sfm = AutoModel.from_pretrained(SF_DIR, trust_remote_code=True, low_cpu_mem_usage=False).to(args.device).eval()
    songs = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")} if os.path.exists(args.out) else set()
    songs = [s for s in songs if s["id"] not in done]
    pool = ThreadPoolExecutor(args.cpu_workers)
    loads = [pool.submit(load24, s["audio_path"]) for s in songs]
    with open(args.out, "a", encoding="utf-8") as f:
        for s, fut in zip(songs, loads):
            y = fut.result(); cpu = pool.submit(cpu_facts, y)
            with torch.inference_mode():
                secs = [dict(start=round(float(x["start"]), 2), end=round(float(x["end"]), 2), label=x["label"]) for x in sfm(y)]
            secs = section_dynamics(y, secs)
            for x in secs: x["lines"] = [l["text"] for l in s.get("lines", []) if x["start"] <= l["t0"] < x["end"]]
            row = dict(id=s["id"], sections=secs, **cpu.result())
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(s["id"], row["bpm"], row["key"], [x["label"] for x in secs], flush=True)
    print("done", args.out)


if __name__ == "__main__":
    main()
