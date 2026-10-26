#!/usr/bin/env python3
"""Compact copy of a tools/ingest_lrc_corpus.py manifest for training loaders: only audio_path, duration, lines (the full
NetEase manifest is 5 GB and vae/data.Crops parses every row). Rows [0, holdout) -> <out>_val.jsonl (the test shard, never
trained on), the rest -> <out>_train.jsonl.
usage: slim_manifest.py data/real_corpus/manifest/wangyiyun.jsonl data/real_corpus/manifest/slim [--holdout 1000]
"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--holdout", type=int, default=1000)
a = ap.parse_args(); n = [0, 0]
with open(a.src, encoding="utf-8") as f, open(a.out + "_val.jsonl", "w") as fv, open(a.out + "_train.jsonl", "w") as ft:
    for i, l in enumerate(f):
        r = json.loads(l)
        row = dict(audio_path=r["audio_path"], duration=r.get("duration", 0),
                   lines=[dict(t0=x["t0"], t1=x["t1"], text=x["text"]) for x in r.get("lines") or []])
        (fv if i < a.holdout else ft).write(json.dumps(row, ensure_ascii=False) + "\n"); n[i >= a.holdout] += 1
print("val", n[0], "train", n[1])
