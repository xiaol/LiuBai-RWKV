#!/usr/bin/env python3
"""Index the YRC files (tools/yrc.py) found by `find ... -name "*.yrc"` lists -> data/real_corpus/yrc/yrc_index.jsonl:
one line per song: id (ne<song id>), path, n_lines, n_uniform (machine-filled lines), n_words, span (s), in_lyric (song in the
clean lyric_* index), split; plus stats.json. CPU only, multiprocess.
usage: python3 tools/yrc_index.py data/real_corpus/yrc/ncm_vip_files.txt data/real_corpus/yrc/wangyiyun_files.txt
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "tools"))
from yrc import parse  # noqa: E402
OUT = ROOT / "data/real_corpus/yrc"


def one(rel):
    p = "/models/musicdl/" + rel; sid = "ne" + rel.split("/")[-2]
    try:
        L = parse(p)
    except Exception as e:
        return dict(id=sid, path=p, err=repr(e)[:100])
    return dict(id=sid, path=p, n_lines=len(L), n_uniform=sum(l["uniform"] for l in L), n_words=sum(len(l["words"]) for l in L),
                span=round(L[-1]["t1"] - L[0]["t0"], 1) if L else 0.0)


def main():
    rels = [l.strip() for f in sys.argv[1:] for l in open(f) if l.strip()]
    split = {}
    for sp in ("dev", "test", "train"):
        for l in open(ROOT / f"data/real_corpus/clean/lyric_{sp}.jsonl", encoding="utf-8"):
            split[l[8:l.index('"', 8)]] = sp
    with Pool(48) as P: rows = P.map(one, rels, chunksize=200)
    seen, st = set(), dict(files=len(rows), err=0, songs=0, good=0, good_in_lyric=0, in_lyric=0, good_words=0)
    with open(OUT / "yrc_index.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if "err" in r: st["err"] += 1; continue
            if r["id"] in seen: continue
            seen.add(r["id"]); r["split"] = split.get(r["id"]); r["in_lyric"] = r["split"] is not None
            r["good"] = r["n_lines"] >= 4 and r["n_uniform"] <= 0.2 * r["n_lines"]
            st["songs"] += 1; st["in_lyric"] += r["in_lyric"]; st["good"] += r["good"]; st["good_in_lyric"] += r["good"] and r["in_lyric"]
            st["good_words"] += r["n_words"] if r["good"] else 0
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(st, open(OUT / "stats.json", "w"), indent=1); print(st)


if __name__ == "__main__":
    main()
