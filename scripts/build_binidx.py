#!/usr/bin/env python3
"""Assemble RWKV-LM binidx files from stage-1 manifests (PLAN.md §2.4 step 7).

Document layout (ids from PLAN.md §2.3):
    world_tokens(text)  EOD  SOA  XCODEC  (CB0_BASE + cb0 frame)*  EOA  0
The trailing 0 is the RWKV-LM document separator (its trainer samples random ctx_len windows from
the flat token stream, so documents are not padded/aligned; 0 marks boundaries as in pretraining).

int32 dtype (vocab 66,560 > uint16). Output:
    data/binidx/<source>_stage1_{train,val}.bin/.idx  + <source>_stage1_binidx.json
The json records token counts and magic_prime for --ctx (RWKV-LM needs it on the command line).

Usage: python scripts/build_binidx.py [--source suno94k] [--ctx 16384] [--shuffle-seed 0]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools" / "RWKV-LM" / "RWKV-v7" / "train_temp"))
from build_stage1_manifest import CB0_BASE, EOA, EOD, SOA, VOCAB_SIZE, VOCAB_TXT, XCODEC  # noqa: E402
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402
from src.binidx import MMapIndexedDataset  # noqa: E402

DTYPE = np.int32


class Builder:
    """Same as RWKV-v5/make_data.py's MMapIndexedDatasetBuilder, dtype-parameterised."""

    def __init__(self, prefix):
        self.prefix = prefix
        self._data_file = open(prefix + ".bin", "wb")
        self._sizes, self._doc_idx, self.n_tokens = [], [0], 0

    def add_document(self, arr):
        assert arr.dtype == DTYPE
        self._data_file.write(arr.tobytes(order="C"))
        self._sizes.append(arr.size); self._doc_idx.append(len(self._sizes)); self.n_tokens += arr.size

    def finalize(self):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(self.prefix + ".idx", DTYPE) as index:
            index.write(self._sizes, self._doc_idx)


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def magic_prime(n_tokens, ctx):
    """Largest prime p ≡ 2 (mod 3) with 0.9 < p / (n_tokens // ctx) <= 1 (RWKV-LM dataset.py assert)."""
    slot = n_tokens // ctx
    for p in range(slot, int(slot * 0.9), -1):
        if p % 3 == 2 and is_prime(p):
            return p
    raise ValueError("no magic prime found")


def build(split, source, tok, shuffle_seed, ctx):
    rows = [json.loads(l) for l in open(ROOT / "data" / "manifests" / f"{source}_stage1_{split}.jsonl", encoding="utf-8")]
    random.Random(shuffle_seed).shuffle(rows)     # shuffle documents once; the trainer samples windows randomly anyway
    by_shard = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    order = {r["id"]: i for i, r in enumerate(rows)}
    docs = [None] * len(rows)
    t0 = time.time()
    for shard, rs in sorted(by_shard.items()):
        z = np.load(ROOT / "data" / "tokens" / source / f"shard-{shard:05d}.npz")
        for r in rs:
            cb0 = z[r["id"]][0].astype(np.int64)
            assert cb0.shape[0] == r["n_frames"] and cb0.min() >= 0 and cb0.max() < 1024, r["id"]
            text = tok.encode(r["text"])
            assert len(text) == r["n_text"], r["id"]
            doc = np.concatenate([np.array(text + [EOD, SOA, XCODEC], dtype=np.int64), cb0 + CB0_BASE,
                                  np.array([EOA, 0], dtype=np.int64)]).astype(DTYPE)
            assert doc.max() < VOCAB_SIZE and doc.size == r["n_doc"], r["id"]
            docs[order[r["id"]]] = doc
    prefix = str(ROOT / "data" / "binidx" / f"{source}_stage1_{split}")
    b = Builder(prefix)
    for d in docs:
        b.add_document(d)
    b.finalize()
    info = {"documents": len(docs), "tokens": b.n_tokens, "magic_prime": magic_prime(b.n_tokens, ctx) if b.n_tokens > ctx * 20 else None,
            "bin": prefix + ".bin", "seconds": round(time.time() - t0, 1)}
    print(f"[{split}] {info}", flush=True)
    return info


def verify(prefix, n_check=3):
    ds = MMapIndexedDataset(prefix)
    n = len(ds)
    out = []
    for i in (0, n // 2, n - 1)[:n_check]:
        d = np.asarray(ds[i])
        text_end = int(np.where(d == EOD)[0][0])
        out.append({"doc": i, "len": int(d.size), "n_text": text_end, "ctrl": d[text_end:text_end + 3].tolist(),
                    "cb0_first5": (d[text_end + 3:text_end + 8] - CB0_BASE).tolist(), "tail": d[-2:].tolist()})
    return {"n_docs": n, "dtype": str(ds._index.dtype), "samples": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="suno94k")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()
    (ROOT / "data" / "binidx").mkdir(parents=True, exist_ok=True)
    tok = TRIE_TOKENIZER(str(VOCAB_TXT))
    info = {"source": args.source, "ctx": args.ctx, "dtype": "int32", "vocab_size": VOCAB_SIZE,
            "layout": "text EOD SOA XCODEC cb0+CB0_BASE... EOA 0",
            "ids": {"EOD": EOD, "SOA": SOA, "EOA": EOA, "XCODEC": XCODEC, "CB0_BASE": CB0_BASE}}
    for split in ("train", "val"):
        info[split] = build(split, args.source, tok, args.shuffle_seed, args.ctx)
        info[split]["verify"] = verify(str(ROOT / "data" / "binidx" / f"{args.source}_stage1_{split}"))
    with open(ROOT / "data" / "binidx" / f"{args.source}_stage1_binidx.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
