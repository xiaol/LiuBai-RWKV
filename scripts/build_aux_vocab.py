#!/usr/bin/env python3
"""Compact lyric-token vocabulary for the lyric-cursor auxiliary head (docs/ARCHITECTURE_NOTES.md sections 7 and 9).
Scans the line-format train binidx, takes the text of every line (SOS ... SOA) cleaned by lyric_targets() in src/model.py
("[marker]" prefix and newlines removed) and writes the most frequent world-token ids, most frequent first.
Head classes: 0 = CTC blank, 1 = UNK (any lyric token outside the list), 2.. = these ids.
usage: build_aux_vocab.py [--data data/binidx/yue2_line_stage1_train] [--size 16382] [--out data/binidx/yue2_line_aux_vocab.npy]
"""
import argparse, collections, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOS, SOA = 65541, 65537


def lyric_targets(text):
    """Line text tokens -> sung-word tokens: drop a leading "[marker]" and every newline (token 11). Same as src/model.py."""
    text = list(text)
    if text and text[0] == 92 and 94 in text: text = text[text.index(94) + 1:]
    return [t for t in text if t != 11]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(ROOT / "data/binidx/yue2_line_stage1_train"))
    ap.add_argument("--size", type=int, default=16382); ap.add_argument("--out", default=str(ROOT / "data/binidx/yue2_line_aux_vocab.npy"))
    a = ap.parse_args()
    t = np.asarray(np.memmap(a.data + ".bin", dtype=np.int32, mode="r"))
    sos = np.flatnonzero(t == SOS); soa = np.flatnonzero(t == SOA); j = np.searchsorted(soa, sos)
    cnt = collections.Counter(); n = 0
    for s, k in zip(sos, j):
        if k < len(soa): cnt.update(lyric_targets(t[s + 1:soa[k]].tolist())); n += 1
    ids, c = zip(*cnt.most_common()); c = np.array(c)
    keep = np.array(ids[:a.size], dtype=np.int64); np.save(a.out, keep)
    print(f"{n} lines, {c.sum()} lyric tokens, {len(ids)} distinct; kept {len(keep)} covering {c[:a.size].sum() / c.sum():.4f} -> {a.out}")


if __name__ == "__main__":
    main()
