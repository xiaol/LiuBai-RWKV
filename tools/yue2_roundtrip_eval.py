#!/usr/bin/env python3
"""Score the G0 round trips (PLAN §8.4) without listening.

For every song dir under --root with A_original.flac, B_vae_only.flac (optional) and C_roundtrip_*.flac:
  * lyric intelligibility: HeartTranscriptor (Whisper-large architecture, Apache-2.0) transcript of each file;
    WER of each transcript against the song's lyrics (section markers stripped) and of C against A's transcript.
  * style/identity: MERT-v2-FullSong recording-level embedding (mean of last hidden state over the song),
    cosine(A, C), cosine(A, B) and cosine(A, other song) as the chance level.
Writes <root>/eval_<tag>.json and prints a table. Run inside venvs/yue2 on one GPU.
"""
import argparse, json, re, sys, itertools
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[1]


def load_mono(path, sr_out):
    a, sr = sf.read(path, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != sr_out:
        g = gcd(sr, sr_out); a = resample_poly(a, sr_out // g, sr // g).astype(np.float32)
    return a


def norm_text(s):
    s = re.sub(r"\[[^\]]*\]", " ", s); s = re.sub(r"\([^)]*\)", " ", s)
    s = s.lower(); s = re.sub(r"[^a-z0-9' ]+", " ", s); return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "out/yue2_roundtrip"))
    ap.add_argument("--variants", default="lora_s32,base_s32", help="C_roundtrip_<variant>.flac suffixes to score")
    ap.add_argument("--transcriptor", default=str(ROOT / "models/HeartTranscriptor-oss"))
    ap.add_argument("--mert", default=str(ROOT / "models/yue2/MERT-v2-FullSong"))
    args = ap.parse_args(); dev = "cuda"; root = Path(args.root)
    import jiwer
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModel, AutoFeatureExtractor
    meta = {}
    for f in (ROOT / "data/meta/suno94k").glob("shard-0000*.jsonl"):
        for l in open(f):
            m = json.loads(l); meta[m["id"]] = m
    songs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "A_original.flac").exists())
    variants = [v for v in args.variants.split(",") if v]
    files = {}
    for d in songs:
        files[d.name] = {"A": d / "A_original.flac"}
        if (d / "B_vae_only.flac").exists(): files[d.name]["B"] = d / "B_vae_only.flac"
        for v in variants:
            p = d / f"C_roundtrip_{v}.flac"
            if p.exists(): files[d.name][f"C_{v}"] = p

    # ---- ASR ----
    proc = WhisperProcessor.from_pretrained(args.transcriptor)
    asr = WhisperForConditionalGeneration.from_pretrained(args.transcriptor, torch_dtype=torch.float16).to(dev).eval()
    cache = root / "transcripts.json"; trans = json.loads(cache.read_text()) if cache.exists() else {}

    def transcribe(path):
        key = str(path.relative_to(root))
        if key in trans: return trans[key]
        a = load_mono(path, 16000); parts = []
        for s in range(0, len(a), 16000 * 30):
            seg = a[s:s + 16000 * 30]
            if len(seg) < 16000: break
            feats = proc(seg, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
            with torch.inference_mode():
                ids = asr.generate(feats, language="en", task="transcribe", max_new_tokens=440)
            parts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        trans[key] = " ".join(parts); cache.write_text(json.dumps(trans, indent=1, ensure_ascii=False)); return trans[key]

    rows = []
    for sid, fl in files.items():
        lyr = norm_text(meta.get(sid, {}).get("lyrics", ""))
        tx = {k: norm_text(transcribe(p)) for k, p in fl.items()}
        r = dict(id=sid, title=meta.get(sid, {}).get("title", "")[:30], n_lyric_words=len(lyr.split()))
        for k, t in tx.items():
            r[f"wer_lyrics_{k}"] = jiwer.wer(lyr, t) if lyr and t else float("nan")
            if k != "A": r[f"wer_vsA_{k}"] = jiwer.wer(tx["A"], t) if tx["A"] and t else float("nan")
        rows.append(r); print(f"  ASR {sid[:8]} " + " ".join(f"{k}:{v:.2f}" for k, v in r.items() if k.startswith("wer")), flush=True)
    del asr; torch.cuda.empty_cache()

    # ---- MERT embeddings ----
    fe = AutoFeatureExtractor.from_pretrained(args.mert, trust_remote_code=True)
    mert = AutoModel.from_pretrained(args.mert, trust_remote_code=True).to(dev).eval()

    def embed(path):
        a = load_mono(path, 24000)[:24000 * 120]
        inp = {k: v.to(dev) for k, v in fe(a, sampling_rate=24000, return_tensors="pt").items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = mert(**inp).last_hidden_state[0].float().mean(0)
        return torch.nn.functional.normalize(h, dim=0).cpu()
    embs = {sid: {k: embed(p) for k, p in fl.items()} for sid, fl in files.items()}
    sids = list(files)
    for r in rows:
        e = embs[r["id"]]
        for k in e:
            if k != "A": r[f"cos_A_{k}"] = float(e["A"] @ e[k])
        others = [embs[o]["A"] for o in sids if o != r["id"]]
        r["cos_A_other"] = float(np.mean([float(e["A"] @ o) for o in others])) if others else float("nan")

    json.dump(rows, open(root / "eval.json", "w"), indent=1)
    keys = [k for k in rows[0] if k.startswith(("wer", "cos"))]
    print("\nmean over %d songs:" % len(rows))
    for k in keys:
        vals = [r[k] for r in rows if k in r and not np.isnan(r[k])]
        if vals: print(f"  {k:28s} {np.mean(vals):.3f}   (median {np.median(vals):.3f}, n={len(vals)})")


if __name__ == "__main__":
    main()
