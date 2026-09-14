#!/usr/bin/env python3
"""Streamed dataset builder for webshart/suno-various-94k (config `original`).

For each shard N:
  1. download original/suno-various-94k-{N:05d}.tar (+ .json index) via hf-mirror with aria2c
  2. iterate the tar: <clip_id>.mp3 + <clip_id>.json (title, caption, lyrics, duration, ...)
  3. decode mp3 -> 16 kHz mono (CPU worker pool), encode with X-Codec on GPU (8 codebooks, 50 Hz)
  4. write data/tokens/suno94k/shard-{N:05d}.npz  (clip_id -> int16 [8, T])
           data/meta/suno94k/shard-{N:05d}.jsonl (one record per clip incl. n_frames)
  5. delete the tar unless --keep-tar

Resumable: shards whose .npz exists are skipped. Per-clip failures are logged and skipped.

Usage:
  python prepare_suno94k.py --shards 0-4 --gpu 0
  python prepare_suno94k.py --shards 0,7,12 --gpu 1 --workers 12
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

REPO = "webshart/suno-various-94k"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/original"
RAW = ROOT / "data" / "raw" / "suno94k"
TOK = ROOT / "data" / "tokens" / "suno94k"
META = ROOT / "data" / "meta" / "suno94k"
LOGS = ROOT / "logs"
N_CODEBOOKS = 8
MAX_SECONDS = 480          # songs longer than this are chunk-encoded to bound GPU memory
CHUNK_SECONDS = 240


def parse_shards(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def download(url, dest, retries=10):
    """aria2c with resume. 2026-09-13: retries raised from 3 (5 shards were skipped by a ~1 min CDN
    blip); total backoff now ≈ 25 min so a worker rides out an edge outage instead of skipping."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def complete():
        return dest.exists() and not Path(str(dest) + ".aria2").exists()

    if complete():
        return dest
    for attempt in range(retries):
        r = subprocess.run(["aria2c", "-q", "-x", "8", "-s", "8", "--file-allocation=none",
                            "-c", "-d", str(dest.parent), "-o", dest.name, url])
        if r.returncode == 0 and dest.exists():
            return dest
        time.sleep(min(30 * (attempt + 1), 300))
        if complete():  # prefetch_shards.py may have delivered it meanwhile
            return dest
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def decode_mp3_bytes(raw):
    import librosa
    wav, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
    return wav.astype(np.float32)


def iter_pairs(tar_path):
    """Yield (clip_id, meta_dict, mp3_bytes) from the tar; mp3 and json are adjacent per clip."""
    pending = {}
    with tarfile.open(tar_path, "r") as t:
        for m in t:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            cid, ext = os.path.splitext(name)
            if ext not in (".mp3", ".json"):
                continue
            data = t.extractfile(m).read()
            slot = pending.setdefault(cid, {})
            slot[ext] = data
            if ".mp3" in slot and ".json" in slot:
                pending.pop(cid)
                yield cid, json.loads(slot[".json"].decode("utf-8")), slot[".mp3"]
    for cid, slot in pending.items():
        print(f"  [warn] unpaired entry {cid}: has {list(slot)}", flush=True)


def process_shard(n, tok, workers, keep_tar):
    tag = f"shard-{n:05d}"
    out_npz, out_meta = TOK / f"{tag}.npz", META / f"{tag}.jsonl"
    if out_npz.exists():
        print(f"[{tag}] exists, skipping", flush=True); return
    fname = f"suno-various-94k-{n:05d}"
    tar_path = download(f"{BASE}/{fname}.tar", RAW / f"{fname}.tar")
    try:
        download(f"{BASE}/{fname}.json", RAW / f"{fname}.json")
    except Exception as e:
        print(f"  [warn] index json not fetched: {e}", flush=True)

    t0 = time.time(); codes_by_id = {}; recs = []; n_fail = 0; audio_sec = 0.0
    pool = ThreadPoolExecutor(max_workers=workers)

    def submit(item):
        cid, meta, mp3 = item
        return cid, meta, pool.submit(decode_mp3_bytes, mp3)

    # keep a bounded window of decode jobs in flight ahead of the GPU
    window, it = [], iter_pairs(tar_path)
    for item in it:
        window.append(submit(item))
        if len(window) < workers * 2:
            continue
        cid, meta, fut = window.pop(0)
        _consume(cid, meta, fut, tok, codes_by_id, recs, n)
        if len(recs) % 100 == 0:
            done_sec = sum(r.get("n_frames", 0) for r in recs) / 50.0
            el = time.time() - t0
            print(f"  [{tag}] {len(recs)} clips, {done_sec/3600:.2f} h audio, {el/60:.1f} min, "
                  f"{done_sec/max(el,1):.0f}x realtime", flush=True)
    for cid, meta, fut in window:
        _consume(cid, meta, fut, tok, codes_by_id, recs, n)
    pool.shutdown()

    for r in recs:
        audio_sec += r.get("n_frames", 0) / 50.0
    n_fail = sum(1 for r in recs if "error" in r)
    TOK.mkdir(parents=True, exist_ok=True); META.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **codes_by_id)
    with open(out_meta, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"[{tag}] {len(codes_by_id)} clips ok, {n_fail} failed, {audio_sec/3600:.1f} h audio, "
          f"{dt/60:.1f} min ({audio_sec/max(dt,1):.0f}x realtime), npz {out_npz.stat().st_size/2**20:.0f} MiB", flush=True)
    if n_fail > 0.1 * max(len(recs), 1):
        # likely a transient cause (GPU taken by another job, network) -> leave no npz so the shard is retried
        out_npz.unlink(missing_ok=True)
        print(f"[{tag}] too many failures ({n_fail}/{len(recs)}); npz removed, shard will be retried on next run", flush=True)
        return
    if not keep_tar:
        tar_path.unlink(missing_ok=True)


def _consume(cid, meta, fut, tok, codes_by_id, recs, shard):
    rec = {"id": cid, "shard": shard}
    for k in ("title", "duration", "caption", "lyrics", "creator", "model_name",
              "major_model_version", "created_at", "play_count", "upvote_count", "search_term"):
        if k in meta:
            rec[k] = meta[k]
    try:
        wav = fut.result()
        dur = len(wav) / 16000
        chunk = CHUNK_SECONDS if dur > MAX_SECONDS else None
        codes = tok.encode_wave(wav, n_codebooks=N_CODEBOOKS, chunk_seconds=chunk)
        codes_by_id[cid] = codes
        rec["n_frames"] = int(codes.shape[1]); rec["audio_seconds"] = round(dur, 2)
        if chunk:
            rec["chunked"] = CHUNK_SECONDS
        if len(recs) % 10 == 0:  # return cached blocks so the GPU stays shareable with other jobs
            import torch; torch.cuda.empty_cache()
    except Exception as e:  # bad mp3, OOM on a pathological file, ...
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  [fail] {cid}: {rec['error']}", flush=True)
        import torch; torch.cuda.empty_cache()
    recs.append(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="e.g. 0-9 or 0,3,7")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="mp3 decode threads")
    ap.add_argument("--keep-tar", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from xcodec_tokenizer import XCodecTokenizer
    tok = XCodecTokenizer(device="cuda:0")
    for n in parse_shards(args.shards):
        try:
            process_shard(n, tok, args.workers, args.keep_tar)
        except Exception as e:
            print(f"[shard-{n:05d}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
