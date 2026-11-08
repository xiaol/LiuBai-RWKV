#!/usr/bin/env python3
"""Listening page for whole-song tests: one row per song, one column per scored dir (tools/score_song.py output).

usage: python3 tools/song_page.py OUT.html LABEL=DIR [LABEL=DIR ...]   (audio is linked relative to OUT.html)
"""
import html, json, os, sys
from pathlib import Path


def main():
    out = Path(sys.argv[1]); cols = [a.split("=", 1) for a in sys.argv[2:]]
    data = [(lab, Path(d), json.load(open(Path(d) / "score.json", encoding="utf-8"))) for lab, d in cols]
    e = html.escape
    h = ["<!doctype html><meta charset=utf-8><title>whole-song test</title><style>body{font:14px sans-serif;margin:20px}"
         "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}table{border-collapse:collapse}audio{width:260px}"
         "details{max-width:260px;font-size:12px}</style><h2>Whole-song test (16 held-out songs, prompt = metadata + timed lyrics)</h2>",
         "<p>CER = per-line letter CER (Qwen3-ASR-1.7B on each line's timed window), lower is better. "
         "thirds = CER in the first / middle / last third of the song. silent = fraction of 1 s windows below -45 dBFS.</p><table><tr><th></th>"]
    for lab, _, sc in data:
        s = sc["summary"]
        h.append(f"<th>{e(lab)}<br>CER {s['cer']:.3f} (median {s['cer_median']:.3f})<br>thirds " +
                 " / ".join(f"{c:.2f}" for c in s["cer_third"]) + f"<br>{s['db_mean']:.1f} dB, silent {s['silent']:.2f}</th>")
    h.append("</tr>")
    for i, s0 in enumerate(data[0][2]["songs"]):
        h.append(f"<tr><td><b>{e(s0.get('title') or s0['name'])}</b><br>{e(s0.get('lang') or '')}, {s0['dur']:.0f} s, {s0['n_lines']} lines</td>")
        for lab, d, sc in data:
            s = next((x for x in sc["songs"] if x["k"] == s0["k"]), None)
            if s is None: h.append("<td></td>"); continue
            src = os.path.relpath(d / f"{s['k']:03d}_{s['name']}.flac", out.parent)
            tr = "<br>".join(f"<b>{l['t0']:.1f}</b> {e(l['ref'])}<br>&rarr; {e(l['hyp'])}" for l in s["lines"])
            h.append(f"<td><audio controls preload=none src='{e(src)}'></audio><br>CER {s['cer']:.3f}, thirds " +
                     " / ".join(f"{c:.2f}" for c in s["cer_third"]) + f", {s['db_mean']:.1f} dB<details><summary>lines</summary>{tr}</details></td>")
        h.append("</tr>")
    out.write_text("".join(h) + "</table>", encoding="utf-8"); print(out)


if __name__ == "__main__":
    main()
