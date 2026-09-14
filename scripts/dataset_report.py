#!/usr/bin/env python3
"""Summarise tokenized shards: counts, duration/lyric stats, script/language mix, dedup, failures.

Usage: python dataset_report.py [--source suno94k] [--check-npz]
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def script_of(text):
    """Cheap script detector: fraction of CJK / Cyrillic / Latin letters."""
    cjk = len(re.findall(r"[一-鿿぀-ヿ가-힯]", text))
    cyr = len(re.findall(r"[Ѐ-ӿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    tot = cjk + cyr + lat
    if tot == 0:
        return "none"
    if cjk / tot > 0.3:
        return "cjk"
    if cyr / tot > 0.3:
        return "cyrillic"
    return "latin"


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="suno94k")
    ap.add_argument("--check-npz", action="store_true", help="open every npz and verify shapes/ranges")
    args = ap.parse_args()
    meta_dir = ROOT / "data" / "meta" / args.source
    tok_dir = ROOT / "data" / "tokens" / args.source
    recs = []
    for f in sorted(meta_dir.glob("shard-*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    ok = [r for r in recs if "error" not in r]
    bad = [r for r in recs if "error" in r]
    print(f"{args.source}: {len(recs)} records in {len(list(meta_dir.glob('shard-*.jsonl')))} shards, "
          f"{len(ok)} ok, {len(bad)} failed")
    if bad:
        c = collections.Counter(r["error"].split(":")[0] for r in bad)
        print("  failure types:", dict(c.most_common(5)))
    if not ok:
        return
    dur = np.array([r["audio_seconds"] for r in ok])
    frames = np.array([r["n_frames"] for r in ok])
    lyr = np.array([len(r.get("lyrics", "")) for r in ok])
    cap = np.array([len(r.get("caption", "")) for r in ok])
    print(f"  audio: {dur.sum()/3600:.1f} h total; duration s p5/50/95 = {pct(dur,5):.0f}/{pct(dur,50):.0f}/{pct(dur,95):.0f}; "
          f"max {dur.max():.0f}; <30 s: {(dur<30).sum()}, >300 s: {(dur>300).sum()}")
    print(f"  cb0 tokens: {frames.sum()/1e6:.1f} M total; per song p50 {pct(frames,50):.0f}, p95 {pct(frames,95):.0f}")
    print(f"  lyrics chars p5/50/95 = {pct(lyr,5):.0f}/{pct(lyr,50):.0f}/{pct(lyr,95):.0f}; empty lyrics: {(lyr==0).sum()}; "
          f"caption chars p50 {pct(cap,50):.0f}, empty captions: {(cap==0).sum()}")
    scripts = collections.Counter(script_of(r.get("lyrics", "")) for r in ok)
    print("  lyric script mix:", dict(scripts))
    markers = collections.Counter()
    for r in ok:
        for m in re.findall(r"\[([a-z][a-z \-]*?)\]", r.get("lyrics", "").lower()):
            markers[m] += 1
    print("  top section markers:", markers.most_common(12))
    models = collections.Counter(r.get("model_name") for r in ok)
    print("  suno model versions:", dict(models.most_common(6)))
    words = collections.Counter()
    for r in ok:
        for w in re.split(r"[,;/]+", r.get("caption", "").lower()):
            w = w.strip()
            if w:
                words[w] += 1
    print(f"  caption tag vocabulary: {len(words)} distinct; top: {words.most_common(15)}")
    h = collections.Counter(hashlib.md5(r.get("lyrics", "").strip().lower().encode()).hexdigest() for r in ok if r.get("lyrics", "").strip())
    dups = sum(c - 1 for c in h.values() if c > 1)
    print(f"  exact-duplicate lyrics: {dups} songs share lyrics with another song")
    creators = collections.Counter(r.get("creator") for r in ok)
    print(f"  creators: {len(creators)} distinct; top creator share {creators.most_common(1)[0][1]/len(ok):.1%}")
    if args.check_npz:
        n = 0
        for f in sorted(tok_dir.glob("shard-*.npz")):
            z = np.load(f)
            for k in z.files:
                a = z[k]
                assert a.dtype == np.int16 and a.ndim == 2 and a.shape[0] == 8, (f, k, a.shape, a.dtype)
                assert 0 <= a.min() and a.max() < 1024, (f, k, a.min(), a.max())
                n += 1
        print(f"  npz check: {n} arrays OK (int16 [8,T], values in 0..1023)")


if __name__ == "__main__":
    main()
