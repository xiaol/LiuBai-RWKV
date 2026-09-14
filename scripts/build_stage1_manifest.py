#!/usr/bin/env python3
"""Select and split songs for stage-1 training (PLAN.md §2.3 / §2.4 steps 3 and 6).

Reads data/meta/<source>/shard-*.jsonl, applies the filter list, tokenizes the text prompt with the
RWKV world tokenizer (so document lengths are exact), and writes

  data/manifests/<source>_stage1_train.jsonl
  data/manifests/<source>_stage1_val.jsonl
  data/manifests/<source>_stage1_summary.json

Each manifest line: id, shard, n_frames, n_text, n_doc, instrumental, creator, lyric_hash, text.
`text` is the exact prompt string that build_binidx.py encodes (so the two stay in sync).

Filters (defaults; all counts are reported):
  * 30 s <= duration <= --max-seconds (300): >300 s does not fit a 16k context with the prompt
  * total document length (text + control tokens + cb0 frames + end tokens) <= --ctx (16384)
  * songs whose lyrics are only section markers / < 20 chars are kept as *instrumental* documents
    with lyrics replaced by "[Instrumental]" (caption-only conditioning)
  * exact-duplicate lyrics are kept in train (different audio renders of one text) but a lyric
    hash that appears in train is removed from val to avoid text leakage
Split: val = creators whose md5(creator) falls in the lowest --val-frac (1 %) of the hash space,
so no creator is in both splits.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "RWKV-LM" / "RWKV-v5" / "tokenizer"))
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402

VOCAB_TXT = ROOT / "tools" / "RWKV-LM" / "RWKV-v7" / "rwkv_vocab_v20230424.txt"

# id layout from PLAN.md §2.3 (also imported by build_binidx.py)
WORLD_VOCAB = 65536
EOD, SOA, EOA, XCODEC = 65536, 65537, 65538, 65539
CB0_BASE = 65544
VOCAB_SIZE = 67072   # 65544 + 1024 = 66568, padded to a multiple of 512 (PLAN.md originally said 66560, which was too small)
N_CONTROL = 3 + 1 + 1   # EOD SOA XCODEC ... EOA, plus RWKV doc separator token 0

MARKER_RE = re.compile(r"\[[^\]]*\]")


def prompt_text(caption, lyrics):
    return f"[Genre] {caption.strip()}\n[Lyrics]\n{lyrics.strip()}\n"


def is_instrumental(lyrics):
    return len(MARKER_RE.sub("", lyrics).strip()) < 20


def lyric_hash(lyrics):
    return hashlib.md5(" ".join(lyrics.lower().split()).encode()).hexdigest()


def creator_bucket(creator, song_id):
    key = creator or song_id
    return int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="suno94k")
    ap.add_argument("--min-seconds", type=float, default=30)
    ap.add_argument("--max-seconds", type=float, default=300)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--val-frac", type=float, default=0.01)
    args = ap.parse_args()

    tok = TRIE_TOKENIZER(str(VOCAB_TXT))
    meta_dir = ROOT / "data" / "meta" / args.source
    out_dir = ROOT / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = []
    for f in sorted(meta_dir.glob("shard-*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    drop = collections.Counter()
    kept = []
    for r in recs:
        if "error" in r:
            drop["encode_error"] += 1; continue
        d = r["audio_seconds"]
        if d < args.min_seconds:
            drop["too_short"] += 1; continue
        if d > args.max_seconds:
            drop["too_long"] += 1; continue
        lyrics = r.get("lyrics", "") or ""
        inst = is_instrumental(lyrics)
        text = prompt_text(r.get("caption", "") or "", "[Instrumental]" if inst else lyrics)
        n_text = len(tok.encode(text))
        n_doc = n_text + N_CONTROL + r["n_frames"]
        if n_doc > args.ctx:
            drop["over_ctx"] += 1; continue
        kept.append({
            "id": r["id"], "shard": r["shard"], "n_frames": r["n_frames"], "n_text": n_text, "n_doc": n_doc,
            "instrumental": inst, "creator": r.get("creator"),
            "lyric_hash": None if inst else lyric_hash(lyrics),
            "upvote_count": r.get("upvote_count") or 0, "audio_seconds": d, "text": text,
        })

    val = [k for k in kept if creator_bucket(k["creator"], k["id"]) < args.val_frac]
    val_ids = {k["id"] for k in val}
    train = [k for k in kept if k["id"] not in val_ids]
    train_hashes = {k["lyric_hash"] for k in train if k["lyric_hash"]}
    leak = [k for k in val if k["lyric_hash"] in train_hashes]
    val = [k for k in val if k["lyric_hash"] not in train_hashes]

    def write(name, rows):
        p = out_dir / f"{args.source}_stage1_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for k in rows:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
        return p

    p_tr, p_va = write("train", train), write("val", val)
    dup_groups = collections.Counter(k["lyric_hash"] for k in train if k["lyric_hash"])
    summary = {
        "source": args.source, "filters": vars(args), "n_meta": len(recs), "dropped": dict(drop),
        "kept": len(kept), "train": len(train), "val": len(val), "val_removed_lyric_leak": len(leak),
        "train_instrumental": sum(k["instrumental"] for k in train),
        "train_cb0_tokens": int(sum(k["n_frames"] for k in train)),
        "train_text_tokens": int(sum(k["n_text"] for k in train)),
        "train_doc_tokens": int(sum(k["n_doc"] for k in train)),
        "val_doc_tokens": int(sum(k["n_doc"] for k in val)),
        "train_hours": round(sum(k["audio_seconds"] for k in train) / 3600, 1),
        "n_doc_p50_p95_max": [int(np.percentile([k["n_doc"] for k in kept], q)) for q in (50, 95)] + [max(k["n_doc"] for k in kept)],
        "n_text_p50_p95_max": [int(np.percentile([k["n_text"] for k in kept], q)) for q in (50, 95)] + [max(k["n_text"] for k in kept)],
        "train_lyric_dup_songs": int(sum(c - 1 for c in dup_groups.values() if c > 1)),
        "train_lyric_dup_max_group": int(max(dup_groups.values()) if dup_groups else 0),
        "train_creators": len({k["creator"] for k in train}), "val_creators": len({k["creator"] for k in val}),
        "vocab": {"EOD": EOD, "SOA": SOA, "EOA": EOA, "XCODEC": XCODEC, "CB0_BASE": CB0_BASE, "VOCAB_SIZE": VOCAB_SIZE},
        "files": {"train": str(p_tr), "val": str(p_va)},
    }
    with open(out_dir / f"{args.source}_stage1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
