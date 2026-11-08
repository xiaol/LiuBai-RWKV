#!/usr/bin/env python3
"""Clean corpus index over the v2 ingest manifests (tools/ingest_lrc_corpus.py -> data/real_corpus/manifest/v2/<source>.jsonl).
Nothing is deleted or copied: every song gets one index row with flags, a split and a sampling weight; training manifests are
filtered views of it.

pass 1 (parallel)  lines cleaned with tools/clean_lyrics.clean_row (credits, notices / copyright lines, section words, glosses,
                   singer labels, titles, translations, furigana); t1 capped at the plausible end of singing:
                   t1 = min(next line start, t0 + 3 s + 0.6 s per sung unit) (units = CJK / kana / Hangul chars + Latin
                   syllables); the uncapped value is kept as t1n. Per-song lyric flags.
pass 2             language (fastText NLLB LID-218 on the cleaned text; Han / kana / Hangul decided by script: cmn, yue by
                   Cantonese particles, jpn, kor), duplicates, splits, weights.
  duplicate        same normalised title + first artist + live flag, durations within 2 s -> one kept (timed > untimed >
                   instrumental > none lyrics, lossless, >= 48 kHz, source ne > kg > jm > sd > yt), the rest dup_of = kept id
  split            by md5 of the normalised title (all versions / covers of a title share a split, so no lyric leak):
                   test 0.5 %, dev 0.5 %, train; titles of the old held-out rows (slim_val, v5_val, gate20) are forced to dev
                   (they were used for model selection). in_v5 = the row is in VAE v5's training manifest.
  weight w         min(1, 300 / songs of the first artist, 100 / songs of the album) over kept rows (Crops honours "w")
audio (--scan)     bad_audio (unreadable), silent (> 50 % of frames < -60 dBFS): excluded; band_lt16k / band_lt19k (lossy
                   cliff), clipped (> 0.1 % samples at full scale), mono: flags only, cut_hz stored
flags              junk (audiobook / lesson / white noise / ringtone / ASMR ...), short (< 60 s), long (> 600 s), acc (accompaniment
                   / karaoke / instrumental version), live, dup; lyric: lyr_after_end (a line starts after the audio ends),
                   lyr_zero (> 10 % zero-length lines), lyr_fast (median > 10 units/s), lyr_partial (lines span < 30 % of a
                   > 4-line song), lyr_few (< 4 lines left), acc_lyrics (timed lyrics on an accompaniment)
outputs (data/real_corpus/clean/)
  index.jsonl              all rows: id source audio_path duration sr lyric_kind lang lang_p split in_v5 w flags dup_of n_lines
                           title artists album is_live (the meta for a meta-only prompt)
  lyric_{train,dev,test}.jsonl   lyric_kind timed, no lyric / junk / acc / dup flag, 60-600 s: audio_path duration lines id lang w
  audio_{train,dev,test}.jsonl   every non-junk, non-dup row 30-900 s (instrumental and untimed too; lines only when lyric-clean)
  stats.json
usage: build_clean_index.py [--src data/real_corpus/manifest/v2] [--out data/real_corpus/clean] [--procs 32] [--scan SCAN.jsonl ...]
"""
import argparse, hashlib, json, math, re, statistics, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "tools"))
from clean_lyrics import clean_row, HAN, KANA, HANG
from audit_corpus import norm_title

SRC_RANK = {"ne": 5, "kg": 4, "jm": 3, "sd": 2, "yt": 1}
KIND_RANK = {"timed": 3, "untimed": 2, "instrumental": 1, "none": 0}
JUNK = re.compile(r"有声书|有声小说|听力|朗读|讲座|课程|第.{1,3}课|相声|评书|小品|广播剧(?!.{0,6}(主题曲|插曲|片尾|片头曲|曲))|单词|"
                  r"audiobook|\blessons?\b|vocabulary|"
                  r"白噪音|白噪声|white noise|pink noise|brown noise|助眠|胎教|冥想引导|guided meditation|asmr|铃声|ringtone|"
                  r"sound effect|音效|闹钟", re.I)
ACC = re.compile(r"伴奏|karaoke|off ?vocal|instrumental|inst\.|\(inst\)|纯音乐版|backing track|カラオケ|minus one|mr\b", re.I)
YUE = re.compile(r"[嘅咗唔喺係冇佢啲嚟咁嘢乜噉睇揾俾哋嗰]")
LOSSLESS = (".flac", ".wav")
OLD_HELDOUT = ["data/real_corpus/manifest/slim_val.jsonl", "data/real_corpus/manifest/v5_val_clean.jsonl",
               "data/real_corpus/manifest/gate20_shard0.jsonl"]
V5_TRAIN = "data/real_corpus/manifest/v5_train_clean.jsonl"


def units(t):
    """sung units: CJK / kana / Hangul characters + Latin syllables (vowel groups per word, >= 1)"""
    n = len(re.findall(f"[{HAN}{KANA}{HANG}]", t))
    for w in re.findall(r"[A-Za-zÀ-ÿ']+", t): n += max(1, len(re.findall(r"[aeiouyàâäéèêëîïôöùûüÿ]+", w.lower())))
    return n


def pass1(line):
    r = json.loads(line); m = r.get("meta") or {}; kind = m.get("lyric_kind") or ("timed" if r.get("lines") else "none")
    st = Counter(); dur = float(r.get("duration") or 0); flags = []
    L = []
    if kind == "timed":
        L = clean_row(r, st)["lines"]
        for i, l in enumerate(L):
            nxt = L[i + 1]["t0"] if i + 1 < len(L) else l["t1"]
            t1n = round(min(max(nxt, l["t0"]), dur), 2)
            l["t1n"] = t1n; l["t1"] = round(min(t1n, l["t0"] + 3.0 + 0.6 * units(l["text"])), 2)
        if len(L) < 4: flags.append("lyr_few")
        if L:
            if any(l["t0"] > dur + 1 for l in L): flags.append("lyr_after_end")
            if sum(l["t1n"] - l["t0"] <= 0.05 for l in L) > 0.1 * len(L): flags.append("lyr_zero")
            rate = [units(l["text"]) / max(l["t1n"] - l["t0"], 0.1) for l in L if l["t1n"] > l["t0"]]
            if rate and statistics.median(rate) > 10: flags.append("lyr_fast")
            if len(L) > 4 and L[-1]["t0"] - L[0]["t0"] < 0.3 * dur: flags.append("lyr_partial")
    text = "\n".join(l["text"] for l in L) if kind == "timed" else (r.get("lyrics") or "") if kind == "untimed" else ""
    title, album = r.get("title") or "", m.get("album") or ""
    if JUNK.search(title) or JUNK.search(album): flags.append("junk")
    if dur < 60: flags.append("short")
    if dur > 600: flags.append("long")
    if ACC.search(title):
        flags.append("acc")
        if kind == "timed": flags.append("acc_lyrics")
    if m.get("is_live"): flags.append("live")
    artist = re.split(r"[,/&、，]", m.get("artists") or "")[0].strip().lower()
    return dict(id=r["id"], source=r["id"][:2], audio_path=r["audio_path"], duration=dur, sr=r.get("sample_rate"),
                lyric_kind=kind, flags=flags, n_lines=len(L), title=title, artists=m.get("artists") or "", album=album,
                is_live=bool(m.get("is_live")), _lines=json.dumps(L, ensure_ascii=False), _text=text[:3000], _nt=norm_title(title) or r["id"], _artist=artist,
                _clean=dict(st))


def lang_of(text, lid):
    han, kana, hang = (len(re.findall(f"[{p}]", text)) for p in (HAN, KANA, HANG))
    letters = len(re.findall(r"[^\W\d_]", text))
    if not letters: return "", 0.0
    if hang > 0.3 * letters: return "kor", 1.0
    if kana > 0.05 * (han + kana) and kana + han > 0.3 * letters: return "jpn", 1.0
    if han > 0.3 * letters: return ("yue" if len(YUE.findall(text)) >= max(2, 0.01 * han) else "cmn"), 1.0
    (p, lab), = lid.f.predict(re.sub(r"\s+", " ", text)[:2000], 1, 0.0, "strict")
    return (lab.replace("__label__", "").split("_")[0] if p >= 0.5 else "und"), round(float(p), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default=str(ROOT / "data/real_corpus/manifest/v2"))
    ap.add_argument("--out", default=str(ROOT / "data/real_corpus/clean")); ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--lid", default=str(ROOT / "models/lid/lid218e.bin"))
    ap.add_argument("--scan", action="append", default=[], help="tools/audio_scan.py output(s) to join")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import fasttext; lid = fasttext.load_model(a.lid)

    rows, clean_st = [], Counter()
    files = sorted(Path(a.src).glob("*.jsonl")); print("sources", [f.name for f in files], flush=True)
    with Pool(a.procs) as pool:
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for r in pool.imap(pass1, fh, chunksize=256):
                    clean_st.update(r.pop("_clean")); rows.append(r)
            print(f"  pass1 {f.name}: {len(rows)} rows", flush=True)

    for r in rows:
        r["lang"], r["lang_p"] = lang_of(r.pop("_text"), lid)
    print("  languages done", flush=True)

    # audio scan (tools/audio_scan.py): unreadable -> bad_audio; band-limited / clipped / mono / silent -> flags, kept
    if a.scan:
        sc = {}
        for f in a.scan:
            for l in open(f, encoding="utf-8"):
                s = json.loads(l); sc[s["audio_path"]] = s
        for r in rows:
            s = sc.get(r["audio_path"])
            if s is None: r["flags"].append("unscanned"); continue
            if "err" in s:
                if not s["err"].startswith("ValueError: short"): r["flags"].append("bad_audio")
                continue
            r["cut_hz"] = s["cut_hz"]
            if s["cut_hz"] < 16000: r["flags"].append("band_lt16k")
            elif s["cut_hz"] < 19000: r["flags"].append("band_lt19k")
            if s["clip"] > 1e-3: r["flags"].append("clipped")
            if s["lr_corr"] >= 0.999: r["flags"].append("mono")
            if s["silence"] > 0.5: r["flags"].append("silent")
        print("  audio scan joined", len(sc), flush=True)

    # duplicates
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[(r["_nt"], r["_artist"], r["is_live"])].append(i)
    score = lambda r: (KIND_RANK[r["lyric_kind"]], "junk" not in r["flags"], r["audio_path"].lower().endswith(LOSSLESS),
                       (r["sr"] or 0) >= 48000, SRC_RANK.get(r["source"], 0), r["n_lines"])
    for g in groups.values():
        if len(g) < 2: continue
        g = sorted(g, key=lambda i: rows[i]["duration"]); clusters = [[g[0]]]
        for i in g[1:]:
            if rows[i]["duration"] - rows[clusters[-1][-1]]["duration"] <= 2.0: clusters[-1].append(i)
            else: clusters.append([i])
        for c in clusters:
            if len(c) < 2: continue
            keep = max(c, key=lambda i: score(rows[i]))
            for i in c:
                if i != keep: rows[i]["flags"].append("dup"); rows[i]["dup_of"] = rows[keep]["id"]

    # splits
    held = set()
    for p in OLD_HELDOUT:
        if (ROOT / p).exists():
            for l in open(ROOT / p, encoding="utf-8"): held.add(json.loads(l)["audio_path"])
    v5 = {json.loads(l)["audio_path"] for l in open(ROOT / V5_TRAIN, encoding="utf-8")} if (ROOT / V5_TRAIN).exists() else set()
    forced_dev = {r["_nt"] for r in rows if r["audio_path"] in held}
    for r in rows:
        h = int(hashlib.md5(r["_nt"].encode()).hexdigest()[:8], 16) % 1000
        r["split"] = "dev" if r["_nt"] in forced_dev else "test" if h < 5 else "dev" if h < 10 else "train"
        r["in_v5"] = r["audio_path"] in v5

    # weights
    kept = [r for r in rows if "dup" not in r["flags"]]
    na = Counter(r["_artist"] for r in kept if r["_artist"]); nb = Counter(r["album"].strip().lower() for r in kept if r["album"].strip())
    for r in rows:
        w = 1.0
        if r["_artist"]: w = min(w, 300 / na[r["_artist"]]) if na[r["_artist"]] else w
        b = r["album"].strip().lower()
        if b and nb[b]: w = min(w, 100 / nb[b])
        r["w"] = round(w, 4)

    LYR_BAD = {"bad_audio", "silent", "lyr_few", "lyr_after_end", "lyr_zero", "lyr_fast", "lyr_partial", "acc_lyrics", "acc", "junk", "dup", "short", "long"}
    files = {f"{k}_{s}": open(out / f"{k}_{s}.jsonl", "w", encoding="utf-8") for k in ("lyric", "audio") for s in ("train", "dev", "test")}
    st = Counter(); hours = Counter()
    with open(out / "index.jsonl", "w", encoding="utf-8") as fi:
        for r in rows:
            lyric_ok = r["lyric_kind"] == "timed" and not (set(r["flags"]) & LYR_BAD)
            audio_ok = not ({"junk", "dup", "bad_audio", "silent"} & set(r["flags"])) and 30 <= r["duration"] <= 900
            rec = {k: v for k, v in r.items() if not k.startswith("_")}
            rec["lyric_ok"], rec["audio_ok"] = lyric_ok, audio_ok
            fi.write(json.dumps(rec, ensure_ascii=False) + "\n")
            slim = json.dumps(dict(id=r["id"], audio_path=r["audio_path"], duration=r["duration"], lang=r["lang"], w=r["w"]),
                              ensure_ascii=False)[:-1] + ', "lines": ' + (r["_lines"] if lyric_ok else "[]") + "}\n"
            if lyric_ok: files[f"lyric_{r['split']}"].write(slim)
            if audio_ok: files[f"audio_{r['split']}"].write(slim)
            h = r["duration"] / 3600; st["rows"] += 1; hours["rows"] += h
            for k in [f"kind:{r['lyric_kind']}", f"src:{r['source']}"] + [f"flag:{x}" for x in r["flags"]] + \
                     ([f"lyric_ok:{r['split']}"] if lyric_ok else []) + ([f"audio_ok:{r['split']}"] if audio_ok else []):
                st[k] += 1; hours[k] += h
            if lyric_ok: st[f"lyric_lang:{r['lang']}"] += 1; hours[f"lyric_lang:{r['lang']}"] += h
    for f in files.values(): f.close()
    stats = {k: dict(n=st[k], hours=round(hours[k])) for k in sorted(st)}; stats["clean_lyrics"] = dict(clean_st)
    stats["test_in_v5"] = sum(1 for r in rows if r["split"] == "test" and r["in_v5"])
    json.dump(stats, open(out / "stats.json", "w"), indent=1, ensure_ascii=False)
    for k, v in stats.items(): print(k, v)


if __name__ == "__main__":
    main()
