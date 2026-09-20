#!/usr/bin/env python3
"""Manifest + RWKV-LM binidx for R3 stage-1 in YuE2 token space (PLAN §8.3 R3 / §8.6), from the G1 corpus.

Input : data/yue2_corpus/suno94k/shard-NNNNN.{jsonl,tokens.npz}   (scripts/prepare_suno94k_yue2.py, head R1-a)
Output: data/manifests/yue2_stage1_{train,val}.jsonl, data/manifests/yue2_stage1_summary.json
        data/binidx/yue2_stage1_{train,val}.{bin,idx}, data/binidx/yue2_stage1_binidx.json
Document: world_tokens("[Genre] caption\n[Lyrics]\nlyrics\n")  EOD  SOA  YUE2CODEC  (YUE2_BASE + code)*  EOA  0
Filters: 30 s <= duration <= --max-seconds, n_doc <= --ctx (8192: 300 s = 7,500 frames + prompt), lyric-only-marker
songs kept as "[Instrumental]" documents. Split: --test-shards (default 2) are excluded entirely (unseen test
material, also unseen by the R1 head), then val = creators in the lowest --val-frac of the md5 hash space, with
lyric-hash leakage removed. Re-runnable while G1 is still producing shards: only finished shards (jsonl present) are used.
usage: build_yue2_binidx.py [--ctx 8192] [--test-shards 2] [--shards all|0-10] [--shuffle-seed 0] [--tag yue2]
"""
import argparse, collections, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools" / "RWKV-LM" / "RWKV-v7" / "train_temp"))
sys.path.insert(0, str(ROOT / "tools" / "RWKV-LM" / "RWKV-v5" / "tokenizer"))
from yue2_layout import EOD, SOA, EOA, YUE2CODEC, YUE2_BASE, N_CODES, VOCAB_SIZE, N_CONTROL  # noqa: E402
from build_stage1_manifest import prompt_text, is_instrumental, lyric_hash, creator_bucket, VOCAB_TXT  # noqa: E402
from build_binidx import Builder, magic_prime, DTYPE  # noqa: E402
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402
from src.binidx import MMapIndexedDataset  # noqa: E402

CORPUS = ROOT / "data/yue2_corpus/suno94k"


def parse_shards(spec):
    if spec == "all": return None
    out = set()
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        elif part: out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192); ap.add_argument("--min-seconds", type=float, default=30); ap.add_argument("--max-seconds", type=float, default=300)
    ap.add_argument("--test-shards", default="2"); ap.add_argument("--shards", default="all"); ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--shuffle-seed", type=int, default=0); ap.add_argument("--tag", default="yue2")
    args = ap.parse_args(); t0 = time.time()
    test = parse_shards(args.test_shards) or set(); only = parse_shards(args.shards)
    tok = TRIE_TOKENIZER(str(VOCAB_TXT))
    recs, heads = [], collections.Counter()
    for jl in sorted(CORPUS.glob("shard-*.jsonl")):
        n = int(jl.stem.split("-")[1])
        if n in test or (only is not None and n not in only) or not jl.with_name(f"{jl.stem}.tokens.npz").exists(): continue
        for l in open(jl, encoding="utf-8"):
            r = json.loads(l); r["shard"] = n; recs.append(r); heads[r.get("head", "?")] += 1
    drop = collections.Counter(); kept = []
    for r in recs:
        if "error" in r or "n_frames" not in r: drop["encode_error"] += 1; continue
        d = r["n_frames"] / 25
        if d < args.min_seconds: drop["too_short"] += 1; continue
        if d > args.max_seconds: drop["too_long"] += 1; continue
        lyrics = r.get("lyrics", "") or ""; inst = is_instrumental(lyrics)
        text = prompt_text(r.get("caption", "") or "", "[Instrumental]" if inst else lyrics)
        n_text = len(tok.encode(text)); n_doc = n_text + N_CONTROL + r["n_frames"]
        if n_doc > args.ctx: drop["over_ctx"] += 1; continue
        kept.append(dict(id=r["id"], shard=r["shard"], n_frames=r["n_frames"], n_text=n_text, n_doc=n_doc, instrumental=inst,
                         creator=r.get("creator"), lyric_hash=None if inst else lyric_hash(lyrics), audio_seconds=d, text=text))
    val = [k for k in kept if creator_bucket(k["creator"], k["id"]) < args.val_frac]; val_ids = {k["id"] for k in val}
    train = [k for k in kept if k["id"] not in val_ids]; train_hashes = {k["lyric_hash"] for k in train if k["lyric_hash"]}
    leak = [k for k in val if k["lyric_hash"] in train_hashes]; val = [k for k in val if k["lyric_hash"] not in train_hashes]
    man = ROOT / "data/manifests"; man.mkdir(parents=True, exist_ok=True); (ROOT / "data/binidx").mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(man / f"{args.tag}_stage1_{name}.jsonl", "w", encoding="utf-8") as f:
            for k in rows: f.write(json.dumps(k, ensure_ascii=False) + "\n")

    def build(split, rows):
        rows = list(rows); random.Random(args.shuffle_seed).shuffle(rows); order = {r["id"]: i for i, r in enumerate(rows)}; docs = [None] * len(rows)
        by_shard = collections.defaultdict(list)
        for r in rows: by_shard[r["shard"]].append(r)
        for shard, rs in sorted(by_shard.items()):
            z = np.load(CORPUS / f"shard-{shard:05d}.tokens.npz")
            for r in rs:
                codes = z[r["id"]].astype(np.int64); assert codes.shape[0] == r["n_frames"] and codes.min() >= 0 and codes.max() < N_CODES, r["id"]
                text = tok.encode(r["text"]); assert len(text) == r["n_text"], r["id"]
                doc = np.concatenate([np.array(text + [EOD, SOA, YUE2CODEC], dtype=np.int64), codes + YUE2_BASE, np.array([EOA, 0], dtype=np.int64)]).astype(DTYPE)
                assert doc.max() < VOCAB_SIZE and doc.size == r["n_doc"], r["id"]; docs[order[r["id"]]] = doc
        prefix = str(ROOT / "data/binidx" / f"{args.tag}_stage1_{split}"); b = Builder(prefix)
        for d in docs: b.add_document(d)
        b.finalize(); ds = MMapIndexedDataset(prefix); d0 = np.asarray(ds[0]); te = int(np.where(d0 == EOD)[0][0])
        return dict(documents=len(docs), tokens=b.n_tokens, magic_prime=magic_prime(b.n_tokens, args.ctx) if b.n_tokens > args.ctx * 20 else None, bin=prefix + ".bin",
                    verify=dict(n_docs=len(ds), len0=int(d0.size), n_text0=te, ctrl0=d0[te:te + 3].tolist(), codes_first5=(d0[te + 3:te + 8] - YUE2_BASE).tolist(), tail0=d0[-2:].tolist()))

    info = dict(tag=args.tag, ctx=args.ctx, vocab_size=VOCAB_SIZE, layout="text EOD SOA YUE2CODEC code+YUE2_BASE... EOA 0",
                ids=dict(EOD=EOD, SOA=SOA, EOA=EOA, YUE2CODEC=YUE2CODEC, YUE2_BASE=YUE2_BASE), heads=dict(heads),
                shards_used=sorted({r["shard"] for r in recs}), test_shards=sorted(test), n_meta=len(recs), dropped=dict(drop), kept=len(kept),
                val_removed_lyric_leak=len(leak), train_hours=round(sum(k["audio_seconds"] for k in train) / 3600, 1),
                train_instrumental=sum(k["instrumental"] for k in train), train_creators=len({k["creator"] for k in train}),
                n_doc_p50_p95_max=[int(np.percentile([k["n_doc"] for k in kept], q)) for q in (50, 95)] + [max(k["n_doc"] for k in kept)] if kept else None)
    for split, rows in (("train", train), ("val", val)): info[split] = build(split, rows)
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(man / f"{args.tag}_stage1_summary.json", "w"), indent=2); json.dump(info, open(ROOT / "data/binidx" / f"{args.tag}_stage1_binidx.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
