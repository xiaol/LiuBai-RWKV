#!/usr/bin/env python3
"""Align Suno lyrics to audio time (PLAN §2.3 "section alignment later", §8.3 R3 lyric cursor; needed after the G3 gate).

ASR (HeartTranscriptor = Whisper-family) with timestamps -> ASR words with times -> sequence alignment to the song's own
lyric words (difflib on normalised words) -> per lyric line / per section: start, end, matched-word coverage.
Word timestamps are used when the model ships alignment heads; otherwise the chunk/segment timestamps are spread
linearly over the words of each segment (good to ~1-2 s, enough for section boundaries and a coarse lyric cursor).
Input audio: any file soundfile reads (opus/flac/mp3), resampled to 16 kHz mono.
Outputs <out>/<id>.json: sections [{marker, start, end, n_words, matched}], lines [...], asr text, coverage, seconds.
usage (venvs/yue2): lyric_align.py --shard 1 --n 6 [--audio-dir data/yue2_corpus/suno94k/shard-00001.audio] [--out data/lyric_align/shard-00001]
"""
import argparse, difflib, json, re, sys, time
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"^\s*\[([^\]]*)\]\s*$")
WORD = re.compile(r"[a-z0-9']+")


def norm_words(s):
    s = s.lower().replace("’", "'"); return WORD.findall(s)


def parse_lyrics(lyrics):
    """-> lines: [{section, marker, text, words}] ; section index increments at each [marker] line."""
    lines, sec, marker = [], -1, None
    for raw in lyrics.splitlines():
        m = MARKER.match(raw)
        if m: sec += 1; marker = m.group(1).strip(); continue
        w = norm_words(raw)
        if not w: continue
        if sec < 0: sec = 0; marker = marker or "intro"
        lines.append(dict(section=sec, marker=marker, text=raw.strip(), words=w))
    return lines


def load_mono16(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True); a = a.mean(1)
    if sr != 16000: g = gcd(sr, 16000); a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    return a


def asr_words(pipe, audio, word_level):
    out = pipe({"raw": audio, "sampling_rate": 16000}, return_timestamps="word" if word_level else True, chunk_length_s=30, batch_size=8)
    words = []
    for ch in out["chunks"]:
        t0, t1 = ch["timestamp"]; t1 = t1 if t1 is not None else t0 + 0.5; ws = norm_words(ch["text"])
        if not ws: continue
        if word_level: words.append(dict(w=ws[0], t0=t0, t1=t1)); continue
        step = (t1 - t0) / len(ws)
        words += [dict(w=w, t0=t0 + i * step, t1=t0 + (i + 1) * step) for i, w in enumerate(ws)]
    return out["text"], words


def align(lines, words):
    lyr = [w for l in lines for w in l["words"]]; owner = [(li, wi) for li, l in enumerate(lines) for wi in range(len(l["words"]))]
    asr = [w["w"] for w in words]; sm = difflib.SequenceMatcher(None, lyr, asr, autojunk=False)
    hit = {}                                                   # lyric word index -> asr word
    for a, b, n in sm.get_matching_blocks():
        for k in range(n): hit[a + k] = words[b + k]
    for l in lines: l["t"] = []
    for i, w in hit.items(): lines[owner[i][0]]["t"].append((w["t0"], w["t1"]))
    for l in lines:
        l["matched"] = len(l["t"]); l["n_words"] = len(l["words"])
        l["start"] = min(t[0] for t in l["t"]) if l["t"] else None; l["end"] = max(t[1] for t in l["t"]) if l["t"] else None; del l["t"]
    secs = {}
    for l in lines:
        s = secs.setdefault(l["section"], dict(section=l["section"], marker=l["marker"], n_words=0, matched=0, start=None, end=None))
        s["n_words"] += l["n_words"]; s["matched"] += l["matched"]
        if l["start"] is not None:
            s["start"] = l["start"] if s["start"] is None else min(s["start"], l["start"]); s["end"] = l["end"] if s["end"] is None else max(s["end"], l["end"])
    cov = len(hit) / max(1, len(lyr))
    return lines, [secs[k] for k in sorted(secs)], cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1); ap.add_argument("--n", type=int, default=6); ap.add_argument("--ids", default="")
    ap.add_argument("--audio-dir", default=""); ap.add_argument("--out", default=""); ap.add_argument("--model", default=str(ROOT / "models/HeartTranscriptor-oss"))
    ap.add_argument("--word", type=int, default=-1, help="1 = word timestamps, 0 = segment timestamps, -1 = auto (word if alignment heads exist)")
    a = ap.parse_args(); dev = "cuda"
    audio_dir = Path(a.audio_dir) if a.audio_dir else ROOT / f"data/yue2_corpus/suno94k/shard-{a.shard:05d}.audio"
    out = Path(a.out) if a.out else ROOT / f"data/lyric_align/shard-{a.shard:05d}"; out.mkdir(parents=True, exist_ok=True)
    meta = {m["id"]: m for m in (json.loads(l) for l in open(ROOT / f"data/meta/suno94k/shard-{a.shard:05d}.jsonl"))}
    ids = [i for i in a.ids.split(",") if i] or [i for i in sorted(p.stem for p in audio_dir.iterdir()) if i in meta and len((meta[i].get("lyrics") or "")) > 50][:a.n]
    from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
    gcfg = json.load(open(Path(a.model) / "generation_config.json")); word_level = (a.word == 1) or (a.word == -1 and bool(gcfg.get("alignment_heads")))
    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval(); proc = WhisperProcessor.from_pretrained(a.model)
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor, torch_dtype=torch.float16, device=0)
    print(f"{len(ids)} songs, word_level={word_level}", flush=True); tot_audio = tot_t = 0; covs = []
    for i, sid in enumerate(ids):
        m = meta[sid]; lines = parse_lyrics(m["lyrics"]); p = next(audio_dir.glob(f"{sid}.*"), None)
        if p is None or not lines: print(f"  skip {sid}"); continue
        t0 = time.time(); audio = load_mono16(p)
        try: text, words = asr_words(pipe, audio, word_level)
        except Exception as e:
            print(f"  ASR FAIL {sid}: {str(e)[:120]}", flush=True); continue
        lines, secs, cov = align(lines, words); dt = time.time() - t0; tot_audio += len(audio) / 16000; tot_t += dt; covs.append(cov)
        json.dump(dict(id=sid, duration=len(audio) / 16000, word_level=word_level, coverage=cov, n_lyric_words=sum(l["n_words"] for l in lines), n_asr_words=len(words),
                       sections=secs, lines=lines, asr_text=text, seconds=dt), open(out / f"{sid}.json", "w"), indent=1, ensure_ascii=False)
        sec_str = " | ".join(f"{s['marker'][:10]} {s['start']:.0f}-{s['end']:.0f}s {s['matched']}/{s['n_words']}" if s["start"] is not None else f"{s['marker'][:10]} --" for s in secs)
        print(f"  [{i + 1}/{len(ids)}] {sid[:8]} {len(audio) / 16000:.0f}s cov {cov:.2f} asr {len(words)}w {dt:.1f}s :: {sec_str[:200]}", flush=True)
    print(f"ALIGN DONE {len(covs)} songs, coverage mean {np.mean(covs):.2f} median {np.median(covs):.2f}, {tot_audio / max(tot_t, 1e-6):.0f}x realtime", flush=True)


if __name__ == "__main__":
    main()
