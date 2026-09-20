#!/usr/bin/env python3
"""Teacher-forced lyric-binding diagnostic on section-interleaved documents (R3.1 gate, PLAN §9 2026-09-18/19).

Documents (data/binidx/yue2_sec_stage1_val, scripts/build_yue2_sec_binidx.py):
    header EOD ( SOS section-text SOA YUE2CODEC codes EOA )* 0
Code-token CE under four prompts:
  matched   : the song's own header and section texts
  shuffled  : header and every section text taken from another val song (wrong lyrics, same format)
  permuted  : the song's own section texts in a rotated order (right song, wrong local text)
  none      : header and section texts removed (EOD SOS SOA YUE2CODEC codes EOA ...)
G3 (whole-song format) gave matched 4.21 vs shuffled 4.31: a 0.10-nat gap = the model ignores the lyrics. For a model
that sings the given words, matched must beat shuffled and permuted by a wide margin; the per-section "first 50 frames"
CE shows whether the binding acts at the section start. Also reports SOS/EOA (section-boundary) CE.
Run in rwkv_py312 (source /root/envs/env_rwkv), one GPU.
usage: rwkv_eval_loss_sec.py --ckpt out/stage1_sec/step-N.pth [--data data/binidx/yue2_sec_stage1_val] [--n 40]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tools/RWKV-LM/RWKV-v7/train_temp"))
from yue2_layout import EOD, SOA, EOA, YUE2CODEC, SOS, YUE2_BASE, N_CODES  # noqa: E402
from rwkv_generate import RWKV7, load_kernel  # noqa: E402
from rwkv_eval_loss import full_logits_loss  # noqa: E402
from src.binidx import MMapIndexedDataset  # noqa: E402


def parse(doc):
    """-> header tokens, [(text tokens, code tokens)] or None if not a section document."""
    doc = [int(t) for t in doc]
    if SOS not in doc: return None
    e = doc.index(EOD); header = doc[:e]; i = e + 1; secs = []
    while i < len(doc) and doc[i] == SOS:
        j = doc.index(SOA, i); k = doc.index(EOA, j); secs.append((doc[i + 1:j], doc[j + 2:k])); i = k + 1
    return header, secs


def assemble(header, secs, texts):
    """Build the token sequence and, per target position, a tag: 'c' code, 'c0' first-50 code of a section, 'b' SOS/EOA, 't' text."""
    seq = list(header) + [EOD]; tags = ["t"] * len(header) + ["t"]
    for (_, codes), text in zip(secs, texts):
        seq += [SOS] + list(text) + [SOA, YUE2CODEC]; tags += ["b"] + ["t"] * len(text) + ["t", "t"]
        seq += codes; tags += ["c0"] * min(50, len(codes)) + ["c"] * max(0, len(codes) - 50)
        seq += [EOA]; tags += ["b"]
    seq.append(0); tags.append("b")
    return seq, tags[1:]                 # tags align with targets seq[1:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", default=str(ROOT / "data/binidx/yue2_sec_stage1_val"))
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); load_kernel(); model = RWKV7(a.ckpt); ds = MMapIndexedDataset(a.data)
    rng = np.random.default_rng(a.seed); order = rng.permutation(len(ds)); docs = []
    for i in order:
        p = parse(np.asarray(ds[int(i)]))
        if p is not None and len(p[1]) >= 2: docs.append(p)
        if len(docs) >= a.n: break
    print(f"{len(docs)} section documents from {a.data}, ckpt {a.ckpt}", flush=True)
    keys = ("matched", "shuffled", "permuted", "none"); res = {k: {"c": [], "c0": [], "b": []} for k in keys}; textloss = []; t0 = time.time()
    for j, (header, secs) in enumerate(docs):
        oh, osecs = docs[(j + 1) % len(docs)]
        variants = {"matched": (header, [t for t, _ in secs]),
                    "shuffled": (oh, [osecs[k % len(osecs)][0] for k in range(len(secs))]),
                    "permuted": (header, [secs[(k + 1) % len(secs)][0] for k in range(len(secs))]),
                    "none": ([], [[] for _ in secs])}
        for k, (h, texts) in variants.items():
            seq, tags = assemble(h, secs, texts); seq, tags = seq[:a.max_len], tags[:a.max_len - 1]
            l = full_logits_loss(model, seq); tags = np.array(tags)
            for tag in ("c", "c0", "b"):
                sel = (tags == tag) | ((tag == "c") & (tags == "c0")); res[k][tag].append(float(l[sel].mean()))
            if k == "matched": textloss.append(float(l[tags == "t"].mean()))
        if j % 5 == 4: print(f"  {j + 1}/{len(docs)} code CE: " + " ".join(f"{k} {np.mean(res[k]['c']):.3f}" for k in keys) + f"  {time.time() - t0:.0f}s", flush=True)
    m = {k: {t: float(np.mean(v)) for t, v in d.items()} for k, d in res.items()}
    print(f"\nRESULT n={len(docs)} section docs, ckpt {Path(a.ckpt).name}")
    print(f"  code CE (all)      : " + " | ".join(f"{k} {m[k]['c']:.4f}" for k in keys))
    print(f"  code CE (first 50) : " + " | ".join(f"{k} {m[k]['c0']:.4f}" for k in keys))
    print(f"  boundary CE (SOS/EOA/0): " + " | ".join(f"{k} {m[k]['b']:.3f}" for k in keys) + f" | text CE {np.mean(textloss):.3f}")
    d = np.array(res["shuffled"]["c"]) - np.array(res["matched"]["c"]); dp = np.array(res["permuted"]["c"]) - np.array(res["matched"]["c"])
    print(f"  lyric-binding gap (shuffled - matched): mean {d.mean():.4f} sd {d.std():.4f} frac>0 {np.mean(d > 0):.2f} | (permuted - matched): mean {dp.mean():.4f} frac>0 {np.mean(dp > 0):.2f}")
    print(f"  G3 whole-song reference: matched 4.209 shuffled 4.309 (gap 0.10); uniform over {N_CODES} codes = {np.log(N_CODES):.3f} nats")


if __name__ == "__main__":
    main()
