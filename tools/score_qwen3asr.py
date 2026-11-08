#!/usr/bin/env python3
"""Letter CER of tools/rescore_qwen3asr.py transcripts (gate romanisation, vs the lyrics each output was given), paired vs a
base run, bootstrap 95 % CI of the median diff. Run in venvs/yue2. usage: score_qwen3asr.py BASE RUN [RUN ...]"""
import json, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.data import romanise  # noqa: E402
from vae.train import edit  # noqa: E402

def cer(hyp, ref):
    h, r = romanise(hyp).replace(" ", ""), ref.replace(" ", ""); return edit(list(h), list(r)) / max(1, len(r))

def load(run):
    rows = json.load(open(ROOT / "out/clm_eval" / run / "qwen3asr.json"))["rows"]
    return {c: np.array([cer(r[f"tr_{c}"], r[f"ref_{c}"]) for r in rows]) for c in ("A_original", "C_own", "C_swap")}

rng = np.random.default_rng(0); base = load(sys.argv[1])
print(f"base {sys.argv[1]}: orig {np.median(base['A_original']):.3f} own {np.median(base['C_own']):.3f} swap-given {np.median(base['C_swap']):.3f}")
for run in sys.argv[2:]:
    a = load(run)
    for c in ("C_own", "C_swap"):
        d = a[c] - base[c]; bs = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(5000)]
        print(f"{run:32s} {c:6s} {np.median(a[c]):.3f} (base {np.median(base[c]):.3f}) diff {np.median(d):+.3f} "
              f"[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] better {(d < 0).sum()}/{len(d)} babble>1.2 {(a[c] > 1.2).sum()} ({(base[c] > 1.2).sum()})")
