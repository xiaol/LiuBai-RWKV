#!/usr/bin/env python3
"""Grow an RWKV checkpoint's emb.weight / head.weight to the stage-1 vocab (PLAN.md §2.3 / §3).

New rows (ids >= 65,536) are initialised as
    emb  : mean of existing rows + N(0, 0.02 * std of existing rows)
    head : mean of existing rows + N(0, 0.02 * std of existing rows)
so the model's initial prediction for/with audio tokens is neutral rather than random.
Everything else is copied unchanged (bf16 stays bf16). RWKV-LM's train.py loads with a strict
load_state_dict, so this must run once before training.

Usage:
  python scripts/resize_vocab.py --src models/rwkv-g1k-3b-temp/rwkv-g1k-3b-temp-5441.pth \
                                 --dst out/stage1/rwkv-init.pth [--vocab 67072] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_stage1_manifest import CB0_BASE, EOA, EOD, SOA, VOCAB_SIZE, WORLD_VOCAB, XCODEC  # noqa: E402


def grow(w, vocab, gen, noise_scale=0.02):
    old_n, d = w.shape
    assert vocab > old_n, (vocab, old_n)
    w32 = w.float()
    mean, std = w32.mean(0, keepdim=True), w32.std()
    new = mean.repeat(vocab - old_n, 1) + torch.randn(vocab - old_n, d, generator=gen) * (noise_scale * std)
    return torch.cat([w32, new], 0).to(w.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sd = torch.load(args.src, map_location="cpu", mmap=True, weights_only=True)
    old = sd["emb.weight"].shape[0]
    assert old == WORLD_VOCAB and sd["head.weight"].shape[0] == old, (old, sd["head.weight"].shape)
    gen = torch.Generator().manual_seed(args.seed)
    out = {}
    for k, v in sd.items():
        if k in ("emb.weight", "head.weight"):
            out[k] = grow(v, args.vocab, gen)
        else:
            out[k] = v.clone()
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    info = {"src": args.src, "dst": args.dst, "old_vocab": old, "new_vocab": args.vocab, "seed": args.seed,
            "emb_shape": list(out["emb.weight"].shape), "head_shape": list(out["head.weight"].shape),
            "dtype": str(out["emb.weight"].dtype),
            "ids": {"EOD": EOD, "SOA": SOA, "EOA": EOA, "XCODEC": XCODEC, "CB0_BASE": CB0_BASE},
            "n_params": sum(v.numel() for v in out.values())}
    with open(str(Path(args.dst).with_suffix("")) + ".resize.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
