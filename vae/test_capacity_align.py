"""Checks for the 2026-10-22 tests: multi-code tokenizer shapes + old-checkpoint compatibility, interleave with 3-tuples / negative
starts, and the word aligner on real held-out crops (how many words get aligned, where they land vs the LRC line start).
run: CUDA_VISIBLE_DEVICES=1 venvs/yue2/bin/python vae/test_capacity_align.py"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.lyric_renderer import interleave  # noqa: E402
from vae.align_words import WordAligner  # noqa: E402
from vae.data import Crops, collate, LETTERS  # noqa: E402

for nq in (1, 2, 4):
    T = SemTokenizer(levels=(8, 8, 8, 8, 6, 5), n_q=nq).cuda(); mu = torch.randn(2, 100, 64).cuda()
    rec, lp, ids = T(mu)
    assert rec.shape == mu.shape and lp.shape[:2] == (2, 400) and int(ids.max()) < T.fsq.n_codes
    assert ids.shape == ((2, 100) if nq == 1 else (2, 100, nq)), ids.shape
    print(f"n_q {nq}: ids {tuple(ids.shape)} params {sum(p.numel() for p in T.parameters()) / 1e6:.1f}M bits/s {12.5 * nq * 16.9:.0f}")
tk = torch.load("out/tok/v2_ctc1_92k/last.pt", map_location="cpu"); SemTokenizer(levels=(8, 8, 8, 8, 6, 5)).load_state_dict(tk["T"]); print("old checkpoint loads")

pt, fi, li = interleave(6, [(-3, [5], 2), (3, [7], 9), (0, [6])], 100)       # cut start clamps to 0, 2- and 3-tuples mix
assert fi.tolist().count(-1) == 6 and pt.tolist()[:2] == [1, 1], (pt.tolist(), fi.tolist(), li.tolist())
print("interleave ok:", li.tolist())

A = WordAligner("cuda"); it = iter(Crops("data/real_corpus/manifest/wangyiyun.jsonl", holdout=0, max_rows=1000, seed=4242, with_lines=True))
wav, tg, lens, ok, cf, lines = collate([next(it) for _ in range(16)]); nf = 100
out = A(wav.cuda(), lines, nf)
dec = lambda ids: "".join(LETTERS[i - 1] for i in ids)
for b in range(3):
    print(f"crop {b}: {len(lines[b])} lines -> {len(out[b])} entries")
    for l in lines[b][:2]: print("   LRC ", l[0], l[2], dec(l[1])[:60])
    for e in sorted(out[b], key=lambda e: e[0])[:8]: print("   word", e[0], dec(e[1]))
print("stats", A.stats)
