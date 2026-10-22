#!/usr/bin/env python3
"""Diversity / quality audit of the ingested corpus (data/real_corpus/manifest/wangyiyun.jsonl).

part meta  (all rows, CPU): durations, declared sample rate / channels, lyrics & line coverage, lyric-language mix,
           normalized-title duplicate groups, artist spread.
part audio (stratified sample, CPU parallel): reads the header and ~60 s from the middle of each file at its native rate and
           measures true sample rate, subtype, bandwidth (LTAS cutoff: a transcode of a lossy source usually stops at
           16/19/20 kHz even when the container says 44.1/48 kHz), HF step, rolloff, silence, clipping, DC, loudness, and a
           72-dim LTAS signature for near-duplicate detection inside title groups.
part report: markdown summary from the two JSON/CSV outputs.

usage: audit_corpus.py --part meta|audio|report [--n 20000] [--workers 48] [--out out/audit]
"""
import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"

LIVE = re.compile(r"[\(\（\[【](live|演唱会|现场|remaster(ed)?|版|demo|acoustic|instrumental|伴奏|纯音乐|karaoke|翻唱|cover|dj|remix|sped up|slowed)[^\)\）\]】]*[\)\）\]】]", re.I)


def norm_title(t):
    t = (t or "").lower()
    t = LIVE.sub("", t)
    t = re.sub(r"[\s\-_·.,'\"!?，。！？、:：;；/\\()（）\[\]【】]+", "", t)
    return t


def lang_of(text):
    if not text: return "none"
    c = Counter()
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF: c["cjk"] += 1
        elif 0xAC00 <= o <= 0xD7AF: c["hangul"] += 1
        elif 0x3040 <= o <= 0x30FF: c["kana"] += 1
        elif 0x0400 <= o <= 0x04FF: c["cyrillic"] += 1
        elif ("a" <= ch.lower() <= "z"): c["latin"] += 1
        elif 0x0E00 <= o <= 0x0E7F: c["thai"] += 1
        elif 0x0600 <= o <= 0x06FF: c["arabic"] += 1
    if not c: return "other"
    return c.most_common(1)[0][0]


def part_meta(args):
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    durs, srs, chs, langs, artists, groups, inst = [], Counter(), Counter(), Counter(), Counter(), defaultdict(list), 0
    with_lyr = with_lines = lines_n = 0
    for r in rows:
        durs.append(float(r.get("duration") or 0)); srs[r.get("sample_rate")] += 1; chs[r.get("channels")] += 1
        lyr = r.get("lyrics") or ""; L = r.get("lines") or []
        with_lyr += bool(lyr.strip()); with_lines += bool(L); lines_n += len(L)
        langs[lang_of(lyr)] += 1
        art = (r.get("meta") or {}).get("artists") or ""
        for a in re.split(r"[,/&、]", art):
            a = a.strip()
            if a: artists[a] += 1
        groups[norm_title(r.get("title"))].append(r["id"])
        if not L and not lyr.strip(): inst += 1
    durs.sort(); n = len(rows)
    q = lambda p: durs[min(n - 1, int(p * n))]
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1 and k}
    import math
    def entropy(c):
        tot = sum(c.values()); return -sum((v / tot) * math.log2(v / tot) for v in c.values() if v)
    out = dict(
        n=n, hours=sum(durs) / 3600, dur_p10=q(.1), dur_p50=q(.5), dur_p90=q(.9), dur_max=durs[-1],
        sample_rate=dict(srs.most_common()), channels=dict(chs.most_common()),
        pct_with_lyrics=with_lyr / n, pct_with_lines=with_lines / n, lines_per_song=lines_n / max(with_lines, 1),
        lyric_language=dict(langs.most_common()), pct_no_lyrics_no_lines=inst / n,
        artists_unique=len(artists), artist_entropy_bits=entropy(artists),
        top_artists=artists.most_common(20), top10_artist_share=sum(v for _, v in artists.most_common(10)) / max(sum(artists.values()), 1),
        unique_norm_titles=len(groups), dup_title_groups=len(dup_groups), songs_in_dup_groups=sum(len(v) for v in dup_groups.values()),
        top_dup_titles=sorted(((k, len(v)) for k, v in dup_groups.items()), key=lambda x: -x[1])[:20])
    out["artist_entropy_bits"] = round(out["artist_entropy_bits"], 2)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(args.out) / "metadata.json", "w"), indent=1, ensure_ascii=False)
    for k, v in out.items(): print(k, "=", v)


def _audio_features(path):
    import numpy as np, soundfile as sf
    rec = dict(path=path, ok=False, err="")
    try:
        info = sf.info(path); sr, ch, frames = info.samplerate, info.channels, info.frames
        rec.update(sr=sr, ch=ch, fmt=info.format, subtype=info.subtype, dur=frames / sr)
        take = min(int(60 * sr), frames); start = max(0, (frames - take) // 2)
        x, _ = sf.read(path, start=start, frames=take, dtype="float32", always_2d=True)
        m = x.mean(1)
        rec["rms_db"] = float(20 * np.log10(np.sqrt((m ** 2).mean()) + 1e-9))
        rec["peak"] = float(np.abs(x).max()); rec["clip"] = float((np.abs(x) > 0.999).mean()); rec["dc"] = float(m.mean())
        win = 2048; hop = 1024; nf = max(1, (len(m) - win) // hop)
        fr = np.lib.stride_tricks.sliding_window_view(m, win)[::hop][:nf]
        e = (fr ** 2).mean(1); rec["silence_frac"] = float((e < 1e-6).mean())
        n = 4096; h = 2048
        frames_ = np.lib.stride_tricks.sliding_window_view(m, n)[::h][: max(1, (len(m) - n) // h)]
        S = np.abs(np.fft.rfft(frames_ * np.hanning(n), axis=1)) ** 2
        f = np.fft.rfftfreq(n, 1 / sr); ltas = S.mean(0) + 1e-12
        cum = np.cumsum(ltas); roll = f[np.searchsorted(cum, 0.95 * cum[-1])]
        thr = ltas.max() * 1e-4                                     # -40 dB: where the audible band really ends
        rec["rolloff95"] = float(roll); rec["bandwidth"] = float(f[np.nonzero(ltas > thr)[0][-1]])
        band = lambda a, b: float(ltas[(f >= a) & (f < b)].mean()) if ((f >= a) & (f < b)).any() else 0.0
        rec["hf_step"] = band(15000, 20000) / max(band(10000, 15000), 1e-12)
        rec["band_levels"] = {f"{a//1000}-{b//1000}k": round(np.log10(band(a, b) + 1e-12), 2) for a, b in ((0, 500), (500, 2000), (2000, 8000), (8000, 12000), (12000, 16000), (16000, 20000), (20000, 24000))}
        edges = np.logspace(np.log10(40), np.log10(min(22000, sr / 2 * 0.99)), 25)
        seg = np.stack([p.mean(0) for p in np.array_split(S, 3, axis=0)])          # frame count need not divide by 3
        vec = np.stack([np.array([np.log10(lt[((f >= edges[i]) & (f < edges[i + 1]))].mean() + 1e-12) for i in range(len(edges) - 1)]) for lt in seg]).ravel()
        rec["sig"] = np.round((vec - vec.mean()) / (vec.std() + 1e-9), 3).tolist()
        rec["ok"] = True
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {str(e)[:120]}"
    return rec


def part_audio(args):
    import multiprocessing as mp
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    rows = [r for r in rows if r.get("audio_path") and Path(r["audio_path"]).suffix.lower() in (".flac", ".mp3", ".wav", ".m4a", ".ogg", ".aac")]
    rng = random.Random(0)
    # stratified by declared sample rate so a rare-but-wanted bucket (48 kHz) is not drowned by 44.1 kHz
    by = defaultdict(list)
    for r in rows: by[r.get("sample_rate")].append(r)
    n = args.n; picked = []
    tot = sum(len(v) for v in by.values())
    for sr, v in by.items():
        k = max(1, int(round(n * len(v) / tot)))
        picked += rng.sample(v, min(k, len(v)))
    picked = picked[: int(n * 1.05)]
    print(f"audio sample: {len(picked)} files from {len(rows)} rows, {len(by)} declared sample rates", flush=True)
    with mp.Pool(args.workers) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(_audio_features, [r["audio_path"] for r in picked], chunksize=8), 1):
            recs.append(r)
            if i % 500 == 0: print(f"  {i}/{len(picked)}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "audio.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    ok = [r for r in recs if r["ok"]]
    import statistics as st
    srs, fmts = Counter(r["sr"] for r in ok), Counter((r["fmt"], r["subtype"]) for r in ok)
    # a container-declared lossless file whose content ends well below Nyquist is a lossy transcode in disguise
    fake = [r for r in ok if r["sr"] >= 44100 and r["bandwidth"] < 0.82 * r["sr"] / 2]
    dead_top = [r for r in ok if r["sr"] >= 44100 and r["bandwidth"] < 18000]
    silent = [r for r in ok if r["silence_frac"] > 0.5]
    clipped = [r for r in ok if r["clip"] > 1e-4]
    lvl = defaultdict(list)
    for r in ok:
        for k, v in r["band_levels"].items(): lvl[k].append(v)
    summ = dict(n=len(recs), ok=len(ok), failed=len(recs) - len(ok), sample_rate=dict(srs.most_common()), format=dict((f"{a}/{b}", c) for (a, b), c in fmts.most_common(8)),
                bandwidth_p10=st.quantiles([r["bandwidth"] for r in ok], n=10)[0], bandwidth_median=st.median(r["bandwidth"] for r in ok),
                pct_bw_below_wideband=len(fake) / len(ok), pct_bw_below_18k=len(dead_top) / len(ok),
                pct_silence_over_half=len(silent) / len(ok), pct_clipped=len(clipped) / len(ok),
                rms_db_median=st.median(r["rms_db"] for r in ok), dur_median=st.median(r["dur"] for r in ok),
                mean_band_level_log10={k: round(st.mean(v), 2) for k, v in lvl.items()},
                example_fake_lossless=[r["path"] for r in fake[:8]], example_failed=[(r["path"], r["err"]) for r in recs if not r["ok"]][:8])
    # near-duplicates: cosine inside normalized-title groups and a coarse global signature bucket
    idx = {r["path"]: r for r in ok}
    groups = defaultdict(list)
    for r in rows:
        if r.get("audio_path") in idx: groups[norm_title(r.get("title"))].append(r["audio_path"])
    import numpy as np
    pairs = 0
    for k, ps in groups.items():
        if len(ps) < 2 or len(ps) > 40: continue
        V = np.array([idx[p]["sig"] for p in ps]); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        C = V @ V.T; pairs += int(((C > 0.97).sum() - len(ps)) // 2)
    summ["near_dup_pairs_in_title_groups"] = pairs
    summ["pct_likely_near_dup"] = pairs / max(len(ok), 1)
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    for k, v in summ.items(): print(k, "=", v)


def part_report(args):
    o = Path(args.out); m = json.load(open(o / "metadata.json")); s = json.load(open(o / "summary.json"))
    L = [f"# Corpus audit: {MANIFEST.name}\n", f"songs {m['n']}, hours {m['hours']:.0f}, duration p10/p50/p90 {m['dur_p10']:.0f}/{m['dur_p50']:.0f}/{m['dur_p90']:.0f} s\n",
         "## Structure and labels", f"- declared sample rate {m['sample_rate']}", f"- channels {m['channels']}",
         f"- lyrics present {m['pct_with_lyrics']:.1%}, timed lines {m['pct_with_lines']:.1%} ({m['lines_per_song']:.0f} lines/song on average)",
         f"- lyric language {m['lyric_language']}", f"- no lyrics and no lines {m['pct_no_lyrics_no_lines']:.1%}",
         f"- artists {m['artists_unique']} unique, entropy {m['artist_entropy_bits']} bits, top-10 share {m['top10_artist_share']:.1%}",
         f"- normalized titles {m['unique_norm_titles']}, duplicate groups {m['dup_title_groups']} covering {m['songs_in_dup_groups']} songs", "",
         "## Audio (sample)", f"- decoded {s['ok']}/{s['n']} (failed {s['failed']})", f"- true sample rate {s['sample_rate']}", f"- formats {s['format']}",
         f"- bandwidth median {s['bandwidth_median']:.0f} Hz, p10 {s['bandwidth_p10']:.0f} Hz",
         f"- content ends below 0.82·Nyquist {s["pct_bw_below_wideband"]:.1%} (suspected lossy transcode in a lossless container)",
         f"- bandwidth below 18 kHz {s["pct_bw_below_18k"]:.1%}",
         f"- >50% silence {s["pct_silence_over_half"]:.1%}, clipped {s['pct_clipped']:.1%}, median RMS {s['rms_db_median']:.1f} dB",
         f"- mean band levels (log10) {s['mean_band_level_log10']}",
         f"- near-duplicate pairs inside title groups {s['near_dup_pairs_in_title_groups']} ({s['pct_likely_near_dup']:.2%} of sample)", ""]
    tx = o / "report.md"; tx.write_text("\n".join(L)); print(tx.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["meta", "audio", "report"])
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--workers", type=int, default=48); ap.add_argument("--out", default=str(ROOT / "out/audit"))
    a = ap.parse_args(); {"meta": part_meta, "audio": part_audio, "report": part_report}[a.part](a)
