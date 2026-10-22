#!/usr/bin/env python3
"""Caption stage A: Qwen3-Omni-30B-A3B-Captioner on <= 30 s windows of every song in a manifest (tools/ingest_lrc_corpus.py).

The Captioner takes audio only (no text prompt) and its model card recommends <= 30 s per clip, so each song is cut into
windows of WIN_MIN..WIN_MAX seconds whose boundaries snap to LRC line starts (a sung line is never split); songs without
lines get plain 30 s windows. Output, one row per song: {id, windows: [{t0, t1, caption}]} -> <out>.
Run with the vllm13 venv (driver 580.178 = native CUDA 13 since 2026-10; do NOT set the old forward-compat LD_LIBRARY_PATH, it now fails with error 803):
  CUDA_VISIBLE_DEVICES=0 \
    /root/envs/vllm13-venv/bin/python tools/caption_omni.py --manifest M.jsonl --out O.jsonl [--quant fp8] [--tp 1]
"""
import argparse, json, os
from math import gcd
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

MODEL = "/models/caption/Qwen3-Omni-30B-A3B-Captioner"
SR, WIN_MIN, WIN_MAX = 16000, 15.0, 30.0


def windows(dur, lines):
    """Greedy cut points: the latest line start that keeps the window <= WIN_MAX (and >= WIN_MIN); else a hard cut."""
    starts = sorted({l["t0"] for l in lines if 0 < l["t0"] < dur})
    out, t = [], 0.0
    while dur - t > 1.0:
        if dur - t <= WIN_MAX: out.append((t, dur)); break
        cands = [s for s in starts if t + WIN_MIN <= s <= t + WIN_MAX]
        e = cands[-1] if cands else t + WIN_MAX
        out.append((t, e)); t = e
    return out


def load16(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != SR: g = gcd(sr, SR); a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="fp8", help="fp8 = weight-only FP8 (Marlin on A100), 'none' = bf16")
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64, help="windows per generate() call")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    llm = LLM(model=MODEL, trust_remote_code=True, tensor_parallel_size=args.tp, quantization=None if args.quant == "none" else args.quant,
              gpu_memory_utilization=0.92, max_model_len=8192, max_num_seqs=16, limit_mm_per_prompt={"audio": 1}, seed=1234)
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens)
    prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio": "x"}]}], add_generation_prompt=True, tokenize=False)
    songs = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    done = set()
    if os.path.exists(args.out): done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")}
    with open(args.out, "a", encoding="utf-8") as f:
        pend = []  # (song, [(t0, t1)], [requests])
        def flush():
            reqs = [r for _, _, rs in pend for r in rs]
            if not reqs: return
            outs = llm.generate(reqs, sp); k = 0
            for s, wins, rs in pend:
                cap = [o.outputs[0].text.strip() for o in outs[k:k + len(rs)]]; k += len(rs)
                f.write(json.dumps(dict(id=s["id"], windows=[dict(t0=round(a, 2), t1=round(b, 2), caption=c) for (a, b), c in zip(wins, cap)]),
                                   ensure_ascii=False) + "\n")
            f.flush(); pend.clear()
        for s in songs:
            if s["id"] in done: continue
            a = load16(s["audio_path"]); wins = windows(len(a) / SR, s.get("lines") or [])
            rs = [dict(prompt=prompt, multi_modal_data={"audio": (a[int(t0 * SR):int(t1 * SR)], SR)}) for t0, t1 in wins]
            pend.append((s, wins, rs))
            if sum(len(x[2]) for x in pend) >= args.batch: flush()
        flush()
    print("done", args.out)


if __name__ == "__main__":
    main()
