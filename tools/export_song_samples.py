#!/usr/bin/env python3
"""Export whole-song test runs (tools/gen_song.py + tools/score_song.py dirs) as MP3 + a browser player for GitHub.

Only generated audio is exported. Reference-only runs (--ref, e.g. the recon ceiling = the test songs' own latents decoded)
contribute their scores, never audio; song titles, artists and lyric text of the commercial test songs stay local.
usage: export_song_samples.py --out samples/song_s1_20k_20261109 --run "s1 20k=out/song_eval/s1_20k" --ref "recon ceiling=out/song_eval/recon"
"""
import argparse, html, json
from pathlib import Path
import soundfile as sf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="label=scored gen dir (audio exported)")
    ap.add_argument("--ref", action="append", default=[], help="label=scored dir (scores only, no audio)")
    ap.add_argument("--title", default="LiuBai whole-song generation (RWKV-7 5.1B on MSVAE latents)")
    ap.add_argument("--note", default="")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    load = lambda spec: (lambda l, d: (l, Path(d), json.load(open(Path(d) / "score.json", encoding="utf-8"))))(*spec.split("=", 1))
    runs = [load(s) for s in a.run]; refs = [load(s) for s in a.ref]
    slug = lambda s: "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")
    songs0 = runs[0][2]["songs"]; byk = lambda sc: {s["k"]: s for s in sc["songs"]}
    rows = []
    for s0 in songs0:
        k = s0["k"]; row = dict(k=k, lang=s0.get("lang"), dur=round(s0["dur"], 1), n_lines=s0["n_lines"], cells=[])
        for label, d, sc in runs + refs:
            s = byk(sc).get(k)
            if s is None: row["cells"].append(None); continue
            cell = dict(label=label, cer=round(s["cer"], 3), thirds=[round(c, 2) for c in s["cer_third"]], db=round(s["db_mean"], 1),
                        silent=round(s["silent"], 3))
            if (label, d, sc) in runs:
                y, sr = sf.read(d / f"{k:03d}_{s['name']}.flac"); name = f"{k:03d}_{slug(label)}.mp3"
                sf.write(out / name, y, sr, format="MP3"); cell["file"] = name
            row["cells"].append(cell)
        rows.append(row)
    summ = {label: sc["summary"] for label, _, sc in runs + refs}
    json.dump(dict(runs={l: str(d) for l, d, _ in runs}, refs={l: str(d) for l, d, _ in refs}, summary=summ, songs=rows),
              open(out / "index.json", "w"), indent=1, ensure_ascii=False)
    e = html.escape; refl = {l for l, _, _ in refs}
    H = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(a.title)}</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222;background:#fafafa}}
.clip{{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:.6rem 1rem;margin:.8rem 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:.5rem}} .grid div{{font-size:.9em}}
audio{{width:100%}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:.2rem .6rem;text-align:right}}
</style></head><body><h1>{e(a.title)}</h1>
<p>Whole songs generated from a text prompt only (song metadata + timed lyrics, chat template), no audio prompt, CFG 3, 24 flow steps,
decoded by MSVAE to 48 kHz stereo. 16 held-out songs (not cherry-picked). Number = per-line letter CER of Qwen3-ASR-1.7B
on each line's timed window against the lyrics the model was given (lower is better; &gt; 1 = many inserted words).
<b>recon ceiling</b> = the same scorer on the real songs' own MSVAE latents decoded (scores only).
Only generated audio is published; titles, lyrics and the original recordings are commercial and stay local.{(" " + e(a.note)) if a.note else ""}</p>
<table><tr><th>run</th><th>CER mean</th><th>median</th><th>thirds</th><th>dB</th><th>silent</th></tr>"""]
    for label, s in summ.items():
        H.append(f"<tr><th>{e(label)}</th><td>{s['cer']:.3f}</td><td>{s['cer_median']:.3f}</td><td>" +
                 " / ".join(f"{c:.2f}" for c in s["cer_third"]) + f"</td><td>{s['db_mean']:.1f}</td><td>{s['silent']:.2f}</td></tr>")
    H.append("</table>")
    for r in rows:
        H.append(f'<div class="clip"><b>song {r["k"]:03d}</b> ({e(r["lang"] or "?")}, {r["dur"]:.0f} s, {r["n_lines"]} lines)<div class="grid">')
        for c in r["cells"]:
            if c is None: continue
            txt = f'{e(c["label"])}: CER {c["cer"]:.2f} (thirds ' + " / ".join(f"{x:.2f}" for x in c["thirds"]) + f'), {c["db"]:.1f} dB'
            H.append(f'<div>{txt}' + (f'<audio controls preload="none" src="{c["file"]}"></audio>' if "file" in c else "") + "</div>")
        H.append("</div></div>")
    H.append("</body></html>"); (out / "player.html").write_text("\n".join(H), encoding="utf-8")
    md = [f"# {a.title}", "", f"**Play in the browser:** https://raw.githack.com/xiaol/LiuBai-RWKV/main/{out.as_posix()}/player.html", "",
          "Whole songs from a text prompt only (metadata + timed lyrics), CFG 3, 24 flow steps, MSVAE decode 48 kHz stereo. "
          "16 held-out songs, not cherry-picked. CER = Qwen3-ASR-1.7B per-line letter CER vs the given lyrics; recon ceiling = real "
          "songs' own latents decoded, same scorer (scores only). Only generated audio is published." + ((" " + a.note) if a.note else ""), "",
          "| run | CER mean | median | thirds | dB | silent |", "|---|---|---|---|---|---|"]
    for label, s in summ.items():
        md.append(f"| {label} | {s['cer']:.3f} | {s['cer_median']:.3f} | " + " / ".join(f"{c:.2f}" for c in s["cer_third"]) +
                  f" | {s['db_mean']:.1f} | {s['silent']:.2f} |")
    labs = [l for l, _, _ in runs + refs]
    md += ["", "| song | lang | dur | " + " | ".join(labs) + " |", "|---|---|---|" + "---|" * len(labs)]
    for r in rows:
        cs = [("-" if c is None else (f"[{c['cer']:.2f}]({c['file']})" if "file" in c else f"{c['cer']:.2f}")) for c in r["cells"]]
        md.append(f"| {r['k']:03d} | {r['lang']} | {r['dur']:.0f} s | " + " | ".join(cs) + " |")
    (out / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print(out, len(rows), "songs")


if __name__ == "__main__":
    main()
