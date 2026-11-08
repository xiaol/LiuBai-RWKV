#!/usr/bin/env python3
"""Score whole songs from tools/gen_song.py: per-line lyric CER from Qwen3-ASR transcripts + loudness.

Step 1 (/root/envs/qwen-asr-venv): tools/transcribe_song_qwen.py DIR -> DIR/asr_qwen.json, one transcript per timed line
window [t0 - 0.3, min(t1 + 0.5, t0 + 30)]. Step 2 (this script, venvs/yue2): CER = letter edit distance
(vae.data.romanise, spaces dropped) / reference letters, summed over a song's lines (micro), then averaged over songs.
Lines without letters are skipped. Also reported: CER by song third (does binding fade?), per language, and loudness
(RMS dBFS per 1 s window: mean, fraction of windows below -45 dBFS = silence, mean over the last 20 % of the song).

usage: venvs/yue2/bin/python tools/score_song.py DIR [DIR ...]   (each DIR has gen*.json + flacs + asr_qwen.json; writes DIR/score.json)
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.data import romanise  # noqa: E402
from vae.train import edit  # noqa: E402


def letters(t): return romanise(t).replace(" ", "")


def loudness(y, sr):
    w = sr; n = len(y) // w
    if n == 0: return dict(db_mean=-120.0, silent=1.0, db_tail=-120.0)
    db = 10 * np.log10(np.maximum((y[:n * w].reshape(n, w) ** 2).mean(1), 1e-12))
    return dict(db_mean=float(db.mean()), silent=float((db < -45).mean()), db_tail=float(db[int(n * 0.8):].mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); args = ap.parse_args()
    for d in map(Path, args.dirs):
        songs = sorted((s for p in sorted(glob.glob(str(d / "gen*.json"))) for s in json.load(open(p, encoding="utf-8"))["songs"]), key=lambda s: s["k"])
        asr = {x["k"]: x["hyp"] for x in json.load(open(d / "asr_qwen.json", encoding="utf-8"))["songs"]}
        rows = []
        for s in songs:
            y, sr = sf.read(d / s["file"], dtype="float32", always_2d=True); y = y.mean(1)
            lines = []
            for l, h in zip(s["lines"], asr[s["k"]]):
                r = letters(l["text"])
                if r: lines.append(dict(t0=l["t0"], ref=l["text"], hyp=h, edits=int(edit(list(letters(h)), list(r))), n=len(r)))
            third = [[], [], []]
            for x in lines: third[min(2, int(3 * x["t0"] / s["dur"]))].append(x)
            cer = lambda xs: sum(x["edits"] for x in xs) / max(1, sum(x["n"] for x in xs)) if xs else float("nan")
            rows.append(dict(k=s["k"], name=s["name"], title=s.get("title"), lang=s.get("lang"), dur=s["dur"], n_lines=len(lines),
                             cer=cer(lines), cer_third=[cer(t) for t in third], **loudness(y, sr), lines=lines))
            print(f"{d.name} {s['k']:3d} {s.get('lang')} cer {rows[-1]['cer']:.3f} thirds " +
                  " ".join(f"{c:.2f}" for c in rows[-1]["cer_third"]) + f" dB {rows[-1]['db_mean']:.1f} silent {rows[-1]['silent']:.2f}", flush=True)
        nm = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
        summ = dict(asr="Qwen3-ASR-1.7B", n=len(rows), cer=nm([r["cer"] for r in rows]), cer_median=float(np.median([r["cer"] for r in rows])),
                    cer_third=[nm([r["cer_third"][i] for r in rows]) for i in range(3)],
                    db_mean=nm([r["db_mean"] for r in rows]), silent=nm([r["silent"] for r in rows]), db_tail=nm([r["db_tail"] for r in rows]),
                    by_lang={L: nm([r["cer"] for r in rows if r["lang"] == L]) for L in sorted({r["lang"] for r in rows})})
        json.dump(dict(summary=summ, songs=rows), open(d / "score.json", "w"), indent=1, ensure_ascii=False)
        print(f"== {d}: " + json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
