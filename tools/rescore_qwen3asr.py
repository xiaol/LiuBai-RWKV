#!/usr/bin/env python3
"""Independent ASR re-score of eval_cursor_lm.py gates with Qwen3-ASR-1.7B (the WildSongBench judge; the gates and the DPO
reward use Whisper / HeartTranscriptor, so a Whisper-specific gain would not survive here). This step only transcribes (run in
/root/envs/qwen-asr-venv, no uroman / torchaudio there); letter CER is scored by tools/score_qwen3asr.py in venvs/yue2.
usage: rescore_qwen3asr.py RUN_DIR [RUN_DIR ...]   (dirs under out/clm_eval; writes <dir>/qwen3asr.json)"""
import json, sys, numpy as np, soundfile as sf, torch
from math import gcd
from pathlib import Path
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
from qwen_asr import Qwen3ASRModel  # noqa: E402

asr = Qwen3ASRModel.from_pretrained("/models/rwkv-music/models/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=256)


def load16(p):
    y, sr = sf.read(p, dtype="float32", always_2d=True); y = y.mean(1); g = gcd(sr, 16000)
    return resample_poly(y, 16000 // g, sr // g).astype(np.float32)


def text(r):
    r = r[0] if isinstance(r, list) else r
    return (getattr(r, "text", None) or (r.get("text") if isinstance(r, dict) else str(r)) or "").strip()


for run in sys.argv[1:]:
    d = ROOT / "out/clm_eval" / run; ev = json.load(open(d / "eval.json")); rows = []
    for row in ev["rows"]:
        k = row["k"]; o = dict(k=k)
        for name, ref in (("A_original", row["own"]), ("C_own", row["own"]), ("C_swap", row["swap"])):
            p = d / f"{k:03d}" / f"{name}.flac"
            if not p.exists(): continue
            o[f"tr_{name}"] = text(asr.transcribe(audio=(load16(p), 16000))); o[f"ref_{name}"] = ref
        rows.append(o)
    json.dump(dict(rows=rows), open(d / "qwen3asr.json", "w"), ensure_ascii=False, indent=1); print(run, "transcribed", flush=True)
