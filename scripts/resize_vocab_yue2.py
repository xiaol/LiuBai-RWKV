#!/usr/bin/env python3
"""Build the R3 init checkpoint (PLAN §8.3 R3, §8.6): grow emb/head to the YuE2 layout vocab (98,816).

From a cb0 stage-1 checkpoint (vocab 67,072, e.g. out/stage1/step-20000.pth): rows < YUE2_BASE (world tokens +
EOD/SOA/EOA/XCODEC/YUE2CODEC slots) are copied; rows >= YUE2_BASE (the old cb0 code rows, meaningless in the new
layout) are dropped and all code rows are re-initialised to mean + N(0, 0.02 std) of the world rows.
From the raw g1k checkpoint (vocab 65,536) everything above 65,536 is new. Other tensors are copied unchanged.
usage: resize_vocab_yue2.py --src out/stage1/step-20000.pth --dst out/stage1_yue2/rwkv-init.pth [--seed 0]
"""
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from yue2_layout import WORLD_VOCAB, YUE2_BASE, VOCAB_SIZE, EOD, SOA, EOA, XCODEC, YUE2CODEC  # noqa: E402


def grow(w, keep, vocab, gen, noise=0.02):
    w32 = w.float()[:keep]; base = w.float()[:WORLD_VOCAB]; mean, std = base.mean(0, keepdim=True), base.std()
    new = mean.repeat(vocab - keep, 1) + torch.randn(vocab - keep, w.shape[1], generator=gen) * (noise * std)
    return torch.cat([w32, new], 0).to(w.dtype)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); sd = torch.load(a.src, map_location="cpu", mmap=True, weights_only=True)
    old = sd["emb.weight"].shape[0]; assert sd["head.weight"].shape[0] == old and old >= WORLD_VOCAB, old
    keep = min(old, YUE2_BASE); gen = torch.Generator().manual_seed(a.seed); out = {}
    for k, v in sd.items(): out[k] = grow(v, keep, VOCAB_SIZE, gen) if k in ("emb.weight", "head.weight") else v.clone()
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True); torch.save(out, a.dst)
    info = dict(src=a.src, dst=a.dst, old_vocab=old, kept_rows=keep, new_vocab=VOCAB_SIZE, seed=a.seed, dtype=str(out["emb.weight"].dtype),
                ids=dict(EOD=EOD, SOA=SOA, EOA=EOA, XCODEC=XCODEC, YUE2CODEC=YUE2CODEC, YUE2_BASE=YUE2_BASE), n_params=sum(v.numel() for v in out.values()))
    json.dump(info, open(str(Path(a.dst).with_suffix("")) + ".resize.json", "w"), indent=2); print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
