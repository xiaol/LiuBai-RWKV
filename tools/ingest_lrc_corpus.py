#!/usr/bin/env python3
"""Real-song corpus ingest: NetEase-style bulk packs (<song_id>/{.info-<id>.json, <name>.flac, <name>.lrc[, .lrc.trans]}).

Per song -> one manifest row (same keys the suno94k G1 jsonl uses where they overlap: id, title, duration, lyrics, caption):
  id            "ne<identifier>" (NetEase song id; falls back to the pack song_id)
  audio_path    absolute path of the flac
  lyrics        cleaned plain lyrics, one line per LRC line, blank line at long gaps (stanza breaks for build_yue2_sec_binidx)
  lines         [{t0, t1, text}] seconds; t1 = next line's start (or song end)
  credits       {role: name} from the LRC header/footer (原唱, 词曲, 编曲, ...); never goes into the caption
  meta          title / artists / album / is_live / has_trans / source
  caption       "" (filled later by the caption stage)
LRC cleaning: credit lines ("role : name" in the leading/trailing block) removed; translation lines removed (second line with the
same timestamp, or a line found in the .lrc.trans file); "纯音乐" lyrics -> instrumental.
usage: ingest_lrc_corpus.py --src data/raw/bulk/bulk_000022 [--src ...] --out data/real_corpus/manifest/bulk_000022.jsonl
"""
import argparse, glob, json, os, re, sys
from pathlib import Path
import soundfile as sf

TS = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")
CREDIT = re.compile(r"^\s*([^:：\[\]]{1,24}?)\s*[:：]\s*(\S.*)$")
GAP_S = 6.0  # silence between lines that starts a new stanza
INSTRUMENTAL = re.compile(r"纯音乐|此歌曲为没有填词的纯音乐|instrumental", re.I)


def parse_lrc(text):
    """-> [(t, line)] in file order; a line with several time tags is repeated at each time."""
    out = []
    for raw in text.splitlines():
        tags = list(TS.finditer(raw))
        if not tags: continue
        body = raw[tags[-1].end():].strip()
        for m in tags:
            sec = m.group(2).replace(":", ".")
            out.append((int(m.group(1)) * 60 + float(sec), body))
    return out


def strip_credits(rows):
    """Credit lines sit in a leading block (and sometimes a trailing one); a colon inside the sung text is left alone."""
    credits = {}
    def take(i):
        m = CREDIT.match(rows[i][1])
        if not m: return False
        credits.setdefault(m.group(1).strip(), m.group(2).strip()); return True
    lo = 0
    while lo < len(rows) and (not rows[lo][1] or take(lo)): lo += 1
    hi = len(rows)
    while hi > lo and (not rows[hi - 1][1] or take(hi - 1)): hi -= 1
    return rows[lo:hi], credits


def drop_translations(rows, trans_text):
    trans = {b for _, b in parse_lrc(trans_text)} if trans_text else set()
    out, seen_t = [], set()
    for t, body in rows:
        key = round(t, 2)
        if body and (key in seen_t or body in trans): continue
        if body: seen_t.add(key)
        out.append((t, body))
    return out


def ingest_song(d):
    d = Path(d)
    # pack layout: .info-<id>.json; live musicdl layout (/models/musicdl/wangyiyun): Info.json, written after the audio,
    # so a dir without it is still downloading. Two lyric copies there ("name.lrc", "name - <netease id>.lrc"): prefer the first.
    info_f = next(iter(glob.glob(str(d / ".info-*.json")) + glob.glob(str(d / "Info.json"))), None)
    if info_f is None: return None, "incomplete (no info json yet)"
    # the downloader leaves 0-byte "name - <id>.flac" stubs next to the real file: take the largest audio file
    audio = [p for p in list(d.glob("*.flac")) + list(d.glob("*.mp3")) if p.stat().st_size > 100_000]
    flac = max(audio, key=lambda p: p.stat().st_size) if audio else None
    lrcs = sorted(d.glob("*.lrc"), key=lambda p: (bool(re.search(r" - \d+\.lrc$", p.name)), p.name))
    lrc_f = lrcs[0] if lrcs else None
    if flac is None: return None, "no audio"
    info = json.load(open(info_f, encoding="utf-8")) if info_f else {}
    si = sf.info(str(flac)); dur = si.frames / si.samplerate
    rows, credits = [], {}
    if lrc_f is not None:
        tr = d / (lrc_f.name + ".trans")
        rows = parse_lrc(lrc_f.read_text(encoding="utf-8", errors="replace"))
        rows.sort(key=lambda r: r[0])
        rows, credits = strip_credits(rows)
        rows = drop_translations(rows, tr.read_text(encoding="utf-8", errors="replace") if tr.exists() else "")
    lines = []
    for i, (t, body) in enumerate(rows):
        if not body: continue
        nxt = next((rows[j][0] for j in range(i + 1, len(rows))), dur)
        lines.append(dict(t0=round(t, 2), t1=round(min(nxt, dur), 2), text=body))
    instrumental = not lines or (len(lines) <= 2 and any(INSTRUMENTAL.search(l["text"]) for l in lines))
    if instrumental: lines = []
    parts = []
    for i, l in enumerate(lines):
        if i and l["t0"] - lines[i - 1]["t1"] >= GAP_S or (i and l["t0"] - lines[i - 1]["t0"] >= GAP_S + 4): parts.append("")
        parts.append(l["text"])
    name = info.get("name") or flac.stem
    rec = dict(id=f"ne{info['identifier']}" if info.get("identifier") else f"pack{info.get('song_id', d.name)}",
               audio_path=str(flac.resolve()), title=name, duration=round(dur, 2), sample_rate=si.samplerate, channels=si.channels,
               lyrics="[Instrumental]" if instrumental else "\n".join(parts), lines=lines, credits=credits, caption="",
               meta=dict(artists=info.get("artists", ""), album=info.get("album", ""), source=info.get("source", ""),
                         is_live=bool(re.search(r"live", name, re.I)), has_trans=any(d.glob("*.lrc.trans")),
                         lyric_status=info.get("lyric_status", "missing" if lrc_f is None else "")))
    return rec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="extracted bulk pack dir (contains <song_id>/ dirs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true", help="incremental: keep --out, add only song dirs not ingested yet (live download dir)")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.append and os.path.exists(args.out):
        seen = {str(Path(json.loads(l)["audio_path"]).parent) for l in open(args.out, encoding="utf-8")}
    n = bad = 0; skips = {}
    with open(args.out, "a" if args.append else "w", encoding="utf-8") as f:
        for src in args.src:
            for d in sorted(p for p in Path(src).iterdir() if p.is_dir()):
                if str(d.resolve()) in seen: continue
                try: rec, err = ingest_song(d)
                except Exception as e: rec, err = None, f"error {type(e).__name__}: {e}"
                if rec is None:
                    skips[err.split(":")[0]] = skips.get(err.split(":")[0], 0) + 1; bad += 1
                    if not err.startswith("incomplete"): print(f"skip {d}: {err}", file=sys.stderr)
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} new songs -> {args.out} ({bad} skipped: {skips})")


if __name__ == "__main__":
    main()
