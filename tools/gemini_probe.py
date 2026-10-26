"""Probe an OpenAI-compatible router model on song clips (audio in, JSON out).

Usage: python tools/gemini_probe.py --model gemini-3.8-flash --n 3 [--t0 60 --dur 30]
Reads data/real_corpus/manifest/wangyiyun.jsonl, cuts a mono 16 kHz WAV clip,
sends it as input_audio, prints the reply next to the LRC lines of that window.
"""
import argparse, base64, io, json, os, random, time, urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PROMPT = (
    "You are annotating a song clip for a music-generation dataset. Listen and return JSON only:\n"
    '{"lyrics_heard": "<verbatim transcript of sung words, original language, \\n per line; empty if none>",'
    ' "language": "", "vocals": "<voice count, gender, timbre, technique>",'
    ' "genre": [], "mood": [], "instruments": [], "tempo_bpm": 0, "key": "",'
    ' "production": "<live/studio, mix, space>", "structure": "<section(s) in this clip>",'
    ' "description": "<2-3 sentences a producer would write>"}\n'
    "Do not guess lyrics you cannot hear; say so instead."
)


def clip_wav(path, t0, dur, sr_out=16000):
    info = sf.info(path)
    t0 = min(t0, max(0.0, info.duration - dur))
    x, sr = sf.read(path, start=int(t0 * info.samplerate), frames=int(dur * info.samplerate), dtype="float32", always_2d=True)
    x = resample_poly(x.mean(1), sr_out, sr).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr_out, format="WAV", subtype="PCM_16")
    return buf.getvalue(), t0


def ask(model, wav, base, token, timeout=180):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
        ]}],
        "max_tokens": 4000,
    }
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--manifest", default="data/real_corpus/manifest/wangyiyun.jsonl")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--t0", type=float, default=60.0)
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = os.environ.get("ROUTER_BASE_URL", os.environ.get("ANTHROPIC_BASE_URL", "https://next-token.cc")).rstrip("/")
    token = os.environ.get("ROUTER_TOKEN") or os.environ["ANTHROPIC_AUTH_TOKEN"]

    rows = []
    with open(a.manifest) as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            r = json.loads(line)
            if r.get("lines"):
                rows.append(r)
    random.Random(a.seed).shuffle(rows)
    out = open(a.out, "w") if a.out else None
    for r in rows[: a.n]:
        wav, t0 = clip_wav(r["audio_path"], a.t0, a.dur)
        ref = [l["text"] for l in r["lines"] if l["t1"] > t0 and l["t0"] < t0 + a.dur]
        try:
            d, dt = ask(a.model, wav, base, token)
            reply = d["choices"][0]["message"]["content"]
            usage = d.get("usage", {})
        except Exception as e:  # keep going on per-clip failures
            reply, usage, dt = f"ERROR {e!r} {getattr(e, 'read', lambda: b'')()[:500]!r}", {}, 0.0
        print(f"=== {r['id']} {r['title']} [{t0:.0f}-{t0 + a.dur:.0f}s] {dt:.1f}s "
              f"in={usage.get('prompt_tokens')} audio_tok={usage.get('prompt_tokens_details', {}).get('audio_tokens')} "
              f"out={usage.get('completion_tokens')} src={usage.get('usage_source')}")
        print("--- LRC ref:\n" + "\n".join(ref))
        print("--- model:\n" + reply + "\n")
        if out:
            out.write(json.dumps({"id": r["id"], "t0": t0, "dur": a.dur, "ref": ref, "reply": reply, "usage": usage,
                                  "sec": dt}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
