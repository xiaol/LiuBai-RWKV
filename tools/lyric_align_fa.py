#!/usr/bin/env python3
"""Lyric -> time alignment by CTC forced alignment (MMS_FA, torchaudio) for R3.1 section-interleaved training (PLAN §8.6, §9 2026-09-18).

Why not ASR (tools/lyric_align.py): HeartTranscriptor word timestamps (cross-attention DTW) run at 3-10x realtime and its
segment timestamps are unusable (no timestamp tokens); the corpus is 40k songs = 2,200 h. Forced alignment uses the known
lyrics as the transcript, so it is both faster (>300x realtime) and directly gives per-word times; quality comes out as a
per-word CTC posterior (score) that we keep per line / section, so bad songs (wrong lyrics, buried vocals) can be filtered.

Text side: lyrics -> sections at every [marker] line (marker-only sections are kept, with no words), lines -> words;
words romanised (NFKD strip + uroman for non-Latin) to the MMS_FA alphabet a-z '.  Audio: opus/flac -> 16 kHz mono.
Optional <star> tokens (--star) between sections absorb untranscribed vocals (ad-libs, backing vocals).

Output: <out>/shard-NNNNN.jsonl, one line per song:
  {id, duration, n_words, n_align_words, score (mean word posterior), sections: [{section, marker, n_words, start, end, score}],
   lines: [{section, text, n_words, start, end, score}], seconds}
Resume-safe (skips ids already in the jsonl). Multi-GPU: --part k --nparts n partitions the selected shards (one writer per shard file).
usage (venvs/yue2): CUDA_VISIBLE_DEVICES=g python tools/lyric_align_fa.py --shards 1-18,43-60 --part g --nparts 4
"""
import argparse, json, re, sys, time, unicodedata
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/yue2_corpus/suno94k"
MARKER = re.compile(r"^\s*\[([^\]]*)\]\s*$")
ASCII_WORD = re.compile(r"[a-z']+")
SR = 16000
_uroman = None


def romanize(s):
    """Lower-case a-z' words for MMS_FA; NFKD-strip diacritics, uroman for anything still non-ASCII (CJK, Cyrillic, Arabic...)."""
    global _uroman
    s = s.replace("’", "'").replace("`", "'")
    t = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    if re.search(r"[^\x00-\x7f]", t):
        if _uroman is None:
            import uroman as ur; _uroman = ur.Uroman()
        t = _uroman.romanize_string(t)
        t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return t.lower()


def parse_lyrics(lyrics):
    """-> sections: [{section, marker, lines: [{text, words(original tokens), roman: [a-z' strings]}]}] (marker-only sections kept)."""
    secs, cur = [], None
    for raw in lyrics.splitlines():
        m = MARKER.match(raw)
        if m:
            cur = dict(section=len(secs), marker=m.group(1).strip().lower(), lines=[]); secs.append(cur); continue
        if not raw.strip(): continue
        if cur is None: cur = dict(section=0, marker="intro", lines=[]); secs.append(cur)
        toks = raw.split()
        roman = [ASCII_WORD.findall(romanize(w)) for w in toks]
        roman = ["".join(r) for r in roman]
        cur["lines"].append(dict(text=raw.strip(), n_words=len(toks), roman=[r for r in roman if r]))
    return secs


def load_mono16(path):
    a, sr = sf.read(str(path), dtype="float32", always_2d=True); a = a.mean(1)
    if sr != SR: g = gcd(sr, SR); a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a


class Songs(torch.utils.data.Dataset):
    def __init__(s, items): s.items = items
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        sid, path, rec = s.items[i]
        try: audio = load_mono16(path)
        except Exception as e: return sid, None, rec, str(e)[:120]
        return sid, audio, rec, None


@torch.no_grad()
def emissions(model, audio, dev):
    x = torch.as_tensor(audio).to(dev)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        em, _ = model(x)
    return torch.log_softmax(em.float(), dim=-1)          # [1, T, C]


def align_song(model, dictionary, star_id, audio, secs, dev, use_star):
    """Forced-align all romanised words of the song; returns per-word (start_s, end_s, score) in lyric order (None for songs w/o words)."""
    import torchaudio.functional as AF
    words = []                                              # (sec_idx, line_idx, roman)
    for s in secs:
        for li, l in enumerate(s["lines"]):
            for r in l["roman"]: words.append((s["section"], li, r))
    if not words: return None, None
    targets, spans_len = [], []                             # token ids; per-word token counts (star tokens are separate "words" with sec=-1)
    units = []                                              # parallel to spans_len: ('w', sec, li) or ('*',)
    last_sec = None
    for sec, li, r in words:
        if use_star and sec != last_sec:
            targets.append(star_id); spans_len.append(1); units.append(("*",)); last_sec = sec
        ids = [dictionary[c] for c in r if c in dictionary]
        if not ids: continue
        targets += ids; spans_len.append(len(ids)); units.append(("w", sec, li))
    if use_star: targets.append(star_id); spans_len.append(1); units.append(("*",))
    em = emissions(model, audio, dev); T = em.shape[1]
    if len(targets) >= T // 2:                              # CTC needs T >= L (+ repeats); refuse hopeless cases
        return None, "too_many_tokens"
    tg = torch.tensor([targets], dtype=torch.int32, device=dev)
    ali, sc = AF.forced_align(em, tg, blank=0)
    spans = AF.merge_tokens(ali[0], sc[0].exp())
    assert len(spans) == len(targets), (len(spans), len(targets))
    ratio = int(audio.shape[0]) / T / SR                         # seconds per emission frame
    out, k = [], 0
    for n, u in zip(spans_len, units):
        ws = spans[k:k + n]; k += n
        if u[0] == "*": continue
        out.append((u[1], u[2], ws[0].start * ratio, ws[-1].end * ratio, float(np.mean([w.score for w in ws]))))
    return out, None


def summarize(secs, word_times, duration):
    """Fold word times into lines and sections. Sections/lines without aligned words get start=end=None."""
    by_line = {}
    for sec, li, t0, t1, sc in word_times: by_line.setdefault((sec, li), []).append((t0, t1, sc))
    lines_out, secs_out, all_sc = [], [], []
    for s in secs:
        so = dict(section=s["section"], marker=s["marker"], n_words=sum(l["n_words"] for l in s["lines"]), n_align=0, start=None, end=None, score=None); scs = []
        for li, l in enumerate(s["lines"]):
            w = by_line.get((s["section"], li), [])
            lo = dict(section=s["section"], text=l["text"], n_words=l["n_words"], n_align=len(w), start=None, end=None, score=None)
            if w:
                lo["start"] = round(min(x[0] for x in w), 2); lo["end"] = round(max(x[1] for x in w), 2); lo["score"] = round(float(np.mean([x[2] for x in w])), 3)
                so["start"] = lo["start"] if so["start"] is None else min(so["start"], lo["start"]); so["end"] = lo["end"] if so["end"] is None else max(so["end"], lo["end"])
                so["n_align"] += len(w); scs += [x[2] for x in w]
            lines_out.append(lo)
        if scs: so["score"] = round(float(np.mean(scs)), 3); all_sc += scs
        secs_out.append(so)
    return lines_out, secs_out, (float(np.mean(all_sc)) if all_sc else None)


def parse_shards(spec):
    out = []
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part: out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="1"); ap.add_argument("--ids", default=""); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "data/lyric_align/fa")); ap.add_argument("--star", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args(); dev = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from torchaudio.pipelines import MMS_FA
    model = MMS_FA.get_model(with_star=bool(a.star)).to(dev).eval()
    labels = MMS_FA.get_labels(star="*" if a.star else None); dictionary = {c: i for i, c in enumerate(labels)}; star_id = dictionary.get("*")
    want = set(i for i in a.ids.split(",") if i)
    items, files = [], {}
    shards = [sh for j, sh in enumerate(parse_shards(a.shards)) if j % a.nparts == a.part]   # partition by shard: one writer per jsonl
    for sh in shards:
        jl = CORPUS / f"shard-{sh:05d}.jsonl"; ad = CORPUS / f"shard-{sh:05d}.audio"
        if not jl.exists() or not ad.exists(): print(f"skip shard {sh}: no jsonl/audio", flush=True); continue
        outp = out / f"shard-{sh:05d}.jsonl"; done = set()
        if outp.exists(): done = {json.loads(l)["id"] for l in open(outp) if l.rstrip().endswith("}")}   # tolerate a truncated last line
        files[sh] = open(outp, "a", encoding="utf-8")
        for l in open(jl, encoding="utf-8"):
            r = json.loads(l); sid = r["id"]
            if "error" in r or sid in done or (want and sid not in want): continue
            p = next(ad.glob(f"{sid}.*"), None)
            if p is None: continue
            items.append((sid, p, dict(shard=sh, lyrics=r.get("lyrics") or "", duration=r.get("duration"), n_frames=r.get("n_frames"))))
    if a.n: items = items[:a.n]
    print(f"{len(items)} songs to align (shards {shards}, part {a.part}/{a.nparts}, star={bool(a.star)})", flush=True)
    dl = torch.utils.data.DataLoader(Songs(items), batch_size=None, num_workers=a.workers, prefetch_factor=4 if a.workers else None)
    t_start = time.time(); tot_audio = 0.0; n_done = n_fail = 0; scores = []; t_align = 0.0
    for sid, audio, rec, err in dl:
        t0 = time.time(); secs = parse_lyrics(rec["lyrics"])
        row = dict(id=sid, shard=rec["shard"], n_words=sum(l["n_words"] for s in secs for l in s["lines"]), n_sections=len(secs))
        if err is not None or audio is None: row["error"] = err or "decode"; n_fail += 1
        else:
            audio = np.asarray(audio, dtype=np.float32); row["duration"] = round(audio.shape[0] / SR, 2); tot_audio += audio.shape[0] / SR
            try:
                wt, why = align_song(model, dictionary, star_id, audio, secs, dev, bool(a.star))
                if wt is None: row["error"] = why or "no_words"; n_fail += 1
                else:
                    lines, so, score = summarize(secs, wt, row["duration"])
                    row.update(n_align_words=len(wt), score=None if score is None else round(score, 3), sections=so, lines=lines); scores.append(score)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); row["error"] = "oom"; n_fail += 1
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:100]}"; n_fail += 1
        row["seconds"] = round(time.time() - t0, 2); t_align += row["seconds"]; n_done += 1
        files[rec["shard"]].write(json.dumps(row, ensure_ascii=False) + "\n"); files[rec["shard"]].flush()
        if n_done % a.log_every == 0 or n_done == len(items):
            el = time.time() - t_start
            print(f"  [{n_done}/{len(items)}] fail {n_fail} | score mean {np.mean(scores):.3f} med {np.median(scores):.3f} | {tot_audio / el:.0f}x realtime wall "
                  f"({tot_audio / max(t_align, 1e-6):.0f}x gpu) | {el / 60:.1f} min, eta {(len(items) - n_done) * el / n_done / 60:.0f} min", flush=True)
    print(f"ALIGN DONE {n_done} songs, {n_fail} failed, {tot_audio / 3600:.1f} h audio in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
