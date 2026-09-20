#!/usr/bin/env python3
"""Re-stream suno-94k shards whose audio was not kept and store 24 kHz mono opus per song, so tools/lyric_align_fa.py can align
them (R3.2 data expansion, docs/STATUS_LOG.md 2026-09-20). No GPU: download tar (aria2c via hf-mirror + cas-bridge pin) →
decode mp3 → resample 24 kHz mono → opus (libsndfile OGG/OPUS, compression_level 0.7 ≈ 80 kbps, the level G1 used) →
data/yue2_corpus/suno94k/shard-NNNNN.audio/<id>.opus → delete tar. Only songs present in the shard's G1 jsonl (tokens exist).
Resume-safe: songs with an existing opus are skipped; a shard whose .audio dir already has >= 99 % of its songs is skipped.
usage (venvs/yue2): fetch_opus_shards.py --shards 19-42,61-84 [--procs 32] [--keep-tar]
"""
import argparse, io, json, os, subprocess, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_suno94k_yue2 import BASE, download, parse_shards  # noqa: E402

OUT = ROOT / "data/yue2_corpus/suno94k"; RAW = ROOT / "data/raw/suno94k"


def to_opus(args):
    cid, audio_bytes, dest, level = args
    try:
        a, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True); m = a.mean(1)
        if sr != 24000: g = gcd(sr, 24000); m = resample_poly(m, 24000 // g, sr // g).astype(np.float32)
        sf.write(dest, m, 24000, format="OGG", subtype="OPUS", compression_level=level); return cid, None
    except Exception as e: return cid, str(e)[:100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True); ap.add_argument("--procs", type=int, default=32); ap.add_argument("--level", type=float, default=0.7)
    ap.add_argument("--keep-tar", action="store_true")
    a = ap.parse_args(); t_all = time.time()
    with ProcessPoolExecutor(a.procs) as pool:
        for n in parse_shards(a.shards):
            tag = f"shard-{n:05d}"; jl = OUT / f"{tag}.jsonl"
            if not jl.exists(): print(f"[{tag}] no G1 jsonl; skip", flush=True); continue
            want = {json.loads(l)["id"] for l in open(jl) if '"error"' not in l}; adir = OUT / f"{tag}.audio"; adir.mkdir(exist_ok=True)
            have = {p.stem for p in adir.glob("*.opus")}
            if len(have) >= 0.99 * len(want): print(f"[{tag}] {len(have)}/{len(want)} opus present; skip", flush=True); continue
            t0 = time.time(); fname = f"suno-various-94k-{n:05d}"; tar_path = download(f"{BASE}/{fname}.tar", RAW / f"{fname}.tar"); t_dl = time.time() - t0
            futs = []
            with tarfile.open(tar_path, "r") as t:
                for m in t:
                    if not m.isfile(): continue
                    cid, ext = os.path.splitext(os.path.basename(m.name))
                    if ext not in (".mp3", ".wav", ".flac") or cid not in want or cid in have: continue
                    futs.append(pool.submit(to_opus, (cid, t.extractfile(m).read(), str(adir / f"{cid}.opus"), a.level)))
            fails = [r for r in (f.result() for f in futs) if r[1]]
            if not a.keep_tar: tar_path.unlink(missing_ok=True)
            size = sum(p.stat().st_size for p in adir.glob("*.opus")) / 2**30
            print(f"[{tag}] {len(futs) - len(fails)} opus written, {len(fails)} failed, download {t_dl / 60:.1f} min, total {(time.time() - t0) / 60:.1f} min, {size:.1f} GiB"
                  + (f" first fail: {fails[0]}" if fails else ""), flush=True)
    print(f"FETCH DONE in {(time.time() - t_all) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
