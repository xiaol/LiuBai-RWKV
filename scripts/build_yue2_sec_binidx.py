#!/usr/bin/env python3
"""R3.1 corpus: section-interleaved documents in YuE2 token space (PLAN §8.6 / §9 2026-09-18 G3 gate decision).

The G3 model learned an unconditional code LM because each lyric line sits thousands of tokens before the frames that sing
it. Here every aligned song becomes

    world_tokens("[Genre] caption\n[Lyrics]\nfull lyrics\n")  EOD
    ( SOS  world_tokens("[marker]\nsection lines\n")  SOA  YUE2CODEC  codes[b_{i-1}:b_i] + YUE2_BASE  EOA )*   0

so the text of a section is at most a few hundred tokens before its own audio (YuE1's CoT format). Songs without a usable
alignment (no opus audio, no words, low CTC score, squeezed sections) keep the G3 whole-song layout
(header EOD SOA YUE2CODEC codes EOA 0), so the model still sees the whole corpus.

Inputs : data/yue2_corpus/suno94k/shard-NNNNN.{jsonl,tokens.npz} (G1, R1-a tokens, 25 Hz)
         data/lyric_align/fa/shard-NNNNN.jsonl (tools/lyric_align_fa.py)
Outputs: data/binidx/yue2_sec_stage1_{train,val}.{bin,idx}, data/binidx/yue2_sec_stage1_binidx.json, data/manifests/yue2_sec_stage1_*.jsonl
Sections = [marker] blocks split further at blank lines (stanzas), so marker-less lyrics still give short sections.
Section boundaries (seconds -> frames * 25): between two aligned sections the midpoint of (end_i, start_{i+1}); next to an
unaligned (marker-only) section 0.5 s outside the aligned one; remaining boundaries interpolated linearly; monotone; the
intro audio before the first words belongs to the first section, the outro to the last. Sections that end up with < 1 s of
audio are merged into the following section's text.
Unaligned songs are kept as whole-song docs only for a --whole-frac (default 0.3) hash-selected subset, so most of the
token budget goes to the section format while the G3 prompt style stays usable.
usage: build_yue2_sec_binidx.py [--ctx 8192] [--min-score 0.04] [--whole-frac 0.3] [--test-shards 2] [--shards all] [--tag yue2_sec]
"""
import argparse, collections, json, random, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools" / "RWKV-LM" / "RWKV-v7" / "train_temp"))
sys.path.insert(0, str(ROOT / "tools" / "RWKV-LM" / "RWKV-v5" / "tokenizer"))
from yue2_layout import EOD, SOA, EOA, YUE2CODEC, SOS, YUE2_BASE, N_CODES, VOCAB_SIZE  # noqa: E402
from build_stage1_manifest import prompt_text, is_instrumental, lyric_hash, creator_bucket, VOCAB_TXT  # noqa: E402
from build_binidx import Builder, magic_prime, DTYPE  # noqa: E402
from build_yue2_binidx import parse_shards  # noqa: E402
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402
from src.binidx import MMapIndexedDataset  # noqa: E402

CORPUS = ROOT / "data/yue2_corpus/suno94k"
ALIGN = ROOT / "data/lyric_align/fa"
FPS = 25


MARKER = re.compile(r"^\s*\[([^\]]*)\]\s*$")


def line_units(al):
    """Line-level units: one unit per lyric line (the [marker] is prepended to the first line of its section); marker-only
    sections become text-less units. Times come straight from the aligner's per-line start/end."""
    units, last_sec = [], None
    for s in al["sections"]:
        ls = [l for l in al["lines"] if l["section"] == s["section"]]
        if not ls: units.append(dict(marker=s["marker"], texts=[], start=None, end=None, n_align=0, head=True)); continue
        for j, l in enumerate(ls):
            units.append(dict(marker=s["marker"], texts=[l["text"]], start=l["start"], end=l["end"], n_align=l["n_align"], head=(j == 0)))
    return units


def units_from(al, raw_lyrics):
    """Alignment units = stanzas: a section (at each [marker]) is split further at blank lines, so marker-less lyrics
    (half of Suno songs) still become several short sections. Text lines of the raw lyrics correspond 1:1, in order, to
    al["lines"] (tools/lyric_align_fa.parse_lyrics keeps every non-blank non-marker line); if the counts disagree we fall
    back to the aligner's sections. -> [{marker, texts, start, end, n_align}] in song order (marker-only units have no texts)."""
    lines = al["lines"]; secs = al["sections"]
    raw = raw_lyrics.splitlines(); n_text = sum(1 for r in raw if r.strip() and not MARKER.match(r))
    if n_text != len(lines):
        return [dict(marker=s["marker"], texts=[l["text"] for l in lines if l["section"] == s["section"]], start=s["start"], end=s["end"], n_align=s["n_align"]) for s in secs]
    units, li, cur, marker, blank = [], 0, None, None, False
    for r in raw:
        m = MARKER.match(r)
        if m: marker = m.group(1).strip().lower(); cur = dict(marker=marker, texts=[], start=None, end=None, n_align=0); units.append(cur); blank = False; continue
        if not r.strip(): blank = True; continue
        if cur is None or (blank and cur["texts"]): cur = dict(marker=marker or "verse", texts=[], start=None, end=None, n_align=0); units.append(cur)
        blank = False; l = lines[li]; li += 1; cur["texts"].append(l["text"])
        if l["start"] is not None:
            cur["start"] = l["start"] if cur["start"] is None else min(cur["start"], l["start"]); cur["end"] = l["end"] if cur["end"] is None else max(cur["end"], l["end"]); cur["n_align"] += l["n_align"]
    return units


def unit_text(u):
    head = f"[{u['marker']}]\n" if u.get("head", True) else ""
    return head + ("\n".join(u["texts"]) + "\n" if u["texts"] else "")


def text_units(lyrics, unit="stanza"):
    """Inference-time counterpart of units_from / line_units: section texts from raw lyrics alone ([marker] blocks split at blank
    lines, or one unit per line with the marker on the first line of its section)."""
    if unit == "line":
        out, marker, first = [], None, True
        for r in lyrics.splitlines():
            m = MARKER.match(r)
            if m: marker = m.group(1).strip().lower(); first = True; continue
            if not r.strip(): continue
            out.append((f"[{marker or 'verse'}]\n" if first else "") + r.strip() + "\n"); first = False
        return out or ["[verse]\n"]
    units, cur, marker, blank = [], None, None, False
    for r in lyrics.splitlines():
        m = MARKER.match(r)
        if m: marker = m.group(1).strip().lower(); cur = dict(marker=marker, texts=[]); units.append(cur); blank = False; continue
        if not r.strip(): blank = True; continue
        if cur is None or (blank and cur["texts"]): cur = dict(marker=marker or "verse", texts=[]); units.append(cur)
        blank = False; cur["texts"].append(r.strip())
    return [unit_text(u) for u in units] or ["[verse]\n"]


def alignment_ok(al, min_score, min_frac, min_spw):
    """Song-level acceptance of a forced alignment; returns (ok, reason)."""
    if "error" in al or al.get("score") is None: return False, al.get("error", "no_score")
    if al["score"] < min_score: return False, "low_score"
    if al["n_align_words"] < min_frac * max(1, al["n_words"]): return False, "unromanized"
    for s in al["sections"]:
        if s["start"] is not None and s["n_align"] >= 8 and (s["end"] - s["start"]) < min_spw * s["n_align"]: return False, "squeezed"
    if not any(s["start"] is not None for s in al["sections"]): return False, "no_aligned_section"
    return True, "ok"


def boundaries(secs, duration, margin=0.5):
    """Frame-domain boundaries b_0..b_n for n sections (b_0 = 0, b_n = duration), see module doc."""
    n = len(secs); b = [None] * (n + 1); b[0] = 0.0; b[n] = duration
    for k in range(1, n):
        p, q = secs[k - 1], secs[k]
        if p["start"] is not None and q["start"] is not None: b[k] = (p["end"] + q["start"]) / 2
        elif p["start"] is not None: b[k] = p["end"] + margin
        elif q["start"] is not None: b[k] = q["start"] - margin
    known = [i for i in range(n + 1) if b[i] is not None]
    for i0, i1 in zip(known[:-1], known[1:]):
        for k in range(i0 + 1, i1): b[k] = b[i0] + (b[i1] - b[i0]) * (k - i0) / (i1 - i0)
    out, cur = [], 0.0
    for x in b: cur = min(max(cur, x), duration); out.append(cur)
    return out


def build_sections(al, raw_lyrics, n_frames, min_sec_frames=FPS, unit="stanza"):
    """-> list of (text, frame_start, frame_end) covering [0, n_frames) exactly; short sections merged forward."""
    secs = line_units(al) if unit == "line" else units_from(al, raw_lyrics); dur = n_frames / FPS
    b = [int(round(x * FPS)) for x in boundaries(secs, dur)]; b[0] = 0; b[-1] = n_frames
    out, pending = [], ""
    for k, s in enumerate(secs):
        txt = pending + unit_text(s); f0, f1 = b[k], b[k + 1]
        if f1 - f0 < min_sec_frames and k < len(secs) - 1: pending = txt; b[k + 1] = f0; continue
        pending = ""; out.append((txt, f0, f1))
    if out and out[-1][2] - out[-1][1] < 1 and len(out) > 1:                     # last section empty: give it the previous one's tail second
        txt, f0, f1 = out.pop(); ptxt, pf0, pf1 = out.pop(); cut = max(pf0 + 1, pf1 - FPS); out += [(ptxt, pf0, cut), (txt, cut, n_frames)]
    if pending and out: txt, f0, f1 = out[-1]; out[-1] = (txt + pending, f0, f1)
    assert out[0][1] == 0 and out[-1][2] == n_frames and all(a[2] == c[1] for a, c in zip(out[:-1], out[1:])), (b, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192); ap.add_argument("--min-seconds", type=float, default=30); ap.add_argument("--max-seconds", type=float, default=300)
    ap.add_argument("--test-shards", default="2"); ap.add_argument("--shards", default="all"); ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--min-score", type=float, default=0.04); ap.add_argument("--min-frac", type=float, default=0.8); ap.add_argument("--min-spw", type=float, default=0.12)
    ap.add_argument("--shuffle-seed", type=int, default=0); ap.add_argument("--tag", default="yue2_sec"); ap.add_argument("--whole-song", type=int, default=1, help="1 = keep unaligned songs as whole-song docs")
    ap.add_argument("--whole-frac", type=float, default=0.3, help="fraction of the unaligned songs kept as whole-song docs (by id hash; 1.0 = all)")
    ap.add_argument("--unit", default="stanza", choices=["stanza", "line"], help="section granularity: stanza (blank-line blocks) or line (one lyric line per SOS block)")
    ap.add_argument("--min-sec-seconds", type=float, default=1.0, help="sections shorter than this are merged into the next one")
    args = ap.parse_args(); t0 = time.time()
    test = parse_shards(args.test_shards) or set(); only = parse_shards(args.shards)
    tok = TRIE_TOKENIZER(str(VOCAB_TXT))
    align = {}
    for jl in sorted(ALIGN.glob("shard-*.jsonl")):
        for l in open(jl, encoding="utf-8"):
            if l.rstrip().endswith("}"): r = json.loads(l); align[r["id"]] = r
    recs = []
    for jl in sorted(CORPUS.glob("shard-*.jsonl")):
        n = int(jl.stem.split("-")[1])
        if n in test or (only is not None and n not in only) or not jl.with_name(f"{jl.stem}.tokens.npz").exists(): continue
        for l in open(jl, encoding="utf-8"): r = json.loads(l); r["shard"] = n; recs.append(r)
    drop, why = collections.Counter(), collections.Counter(); kept = []
    for r in recs:
        if "error" in r or "n_frames" not in r: drop["encode_error"] += 1; continue
        d = r["n_frames"] / FPS
        if d < args.min_seconds: drop["too_short"] += 1; continue
        if d > args.max_seconds: drop["too_long"] += 1; continue
        lyrics = r.get("lyrics", "") or ""; inst = is_instrumental(lyrics)
        header = prompt_text(r.get("caption", "") or "", "[Instrumental]" if inst else lyrics); n_header = len(tok.encode(header))
        al = align.get(r["id"]); ok, reason = alignment_ok(al, args.min_score, args.min_frac, args.min_spw) if al and not inst else (False, "no_alignment" if not inst else "instrumental")
        why[reason] += 1
        rec = dict(id=r["id"], shard=r["shard"], n_frames=r["n_frames"], n_header=n_header, instrumental=inst, creator=r.get("creator"),
                   lyric_hash=None if inst else lyric_hash(lyrics), audio_seconds=d, header=header, layout="whole", align_reason=reason)
        if ok:
            secs = build_sections(al, lyrics, r["n_frames"], min_sec_frames=int(args.min_sec_seconds * FPS), unit=args.unit); sec_tok = [tok.encode(t) for t, _, _ in secs]
            n_doc = n_header + 1 + sum(1 + len(st) + 2 + (f1 - f0) + 1 for st, (_, f0, f1) in zip(sec_tok, secs)) + 1
            if n_doc > args.ctx: drop["over_ctx_sec"] += 1; ok = False; why["over_ctx_sec"] += 1
            else: rec.update(layout="sections", n_doc=n_doc, sections=[dict(text=t, f0=f0, f1=f1, n_text=len(st)) for (t, f0, f1), st in zip(secs, sec_tok)], align_score=al["score"])
        if not ok:
            if not args.whole_song or creator_bucket(None, r["id"] + "whole") >= args.whole_frac: drop["whole_subsampled"] += 1; continue
            rec["n_doc"] = n_header + 5 + r["n_frames"]
            if rec["n_doc"] > args.ctx: drop["over_ctx"] += 1; continue
        kept.append(rec)
    val = [k for k in kept if creator_bucket(k["creator"], k["id"]) < args.val_frac]; val_ids = {k["id"] for k in val}
    train = [k for k in kept if k["id"] not in val_ids]; train_hashes = {k["lyric_hash"] for k in train if k["lyric_hash"]}
    leak = [k for k in val if k["lyric_hash"] in train_hashes]; val = [k for k in val if k["lyric_hash"] not in train_hashes]
    man = ROOT / "data/manifests"; man.mkdir(parents=True, exist_ok=True); (ROOT / "data/binidx").mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(man / f"{args.tag}_stage1_{name}.jsonl", "w", encoding="utf-8") as f:
            for k in rows: f.write(json.dumps(k, ensure_ascii=False) + "\n")

    def encode(r, codes):
        head = np.array(tok.encode(r["header"]) + [EOD], dtype=np.int64)
        if r["layout"] == "whole":
            body = np.concatenate([np.array([SOA, YUE2CODEC], dtype=np.int64), codes + YUE2_BASE, np.array([EOA, 0], dtype=np.int64)])
        else:
            parts = []
            for s in r["sections"]:
                parts += [np.array([SOS] + tok.encode(s["text"]) + [SOA, YUE2CODEC], dtype=np.int64), codes[s["f0"]:s["f1"]] + YUE2_BASE, np.array([EOA], dtype=np.int64)]
            body = np.concatenate(parts + [np.array([0], dtype=np.int64)])
        doc = np.concatenate([head, body]).astype(DTYPE); assert doc.size == r["n_doc"] and doc.max() < VOCAB_SIZE, (r["id"], doc.size, r["n_doc"])
        return doc

    def build(split, rows):
        rows = list(rows); random.Random(args.shuffle_seed).shuffle(rows); order = {r["id"]: i for i, r in enumerate(rows)}; docs = [None] * len(rows)
        by_shard = collections.defaultdict(list)
        for r in rows: by_shard[r["shard"]].append(r)
        for shard, rs in sorted(by_shard.items()):
            z = np.load(CORPUS / f"shard-{shard:05d}.tokens.npz")
            for r in rs:
                codes = z[r["id"]].astype(np.int64); assert codes.shape[0] == r["n_frames"] and codes.min() >= 0 and codes.max() < N_CODES, r["id"]
                docs[order[r["id"]]] = encode(r, codes)
        prefix = str(ROOT / "data/binidx" / f"{args.tag}_stage1_{split}"); b = Builder(prefix)
        for d in docs: b.add_document(d)
        b.finalize(); ds = MMapIndexedDataset(prefix)
        j = next(i for i, r in enumerate(rows) if r["layout"] == "sections"); d0 = np.asarray(ds[j]); te = int(np.where(d0 == EOD)[0][0])
        return dict(documents=len(docs), tokens=b.n_tokens, magic_prime=magic_prime(b.n_tokens, args.ctx) if b.n_tokens > args.ctx * 20 else None, bin=prefix + ".bin",
                    section_docs=sum(r["layout"] == "sections" for r in rows), section_tokens=int(sum(r["n_doc"] for r in rows if r["layout"] == "sections")),
                    verify=dict(doc=j, len=int(d0.size), n_header=te, after_eod=d0[te + 1:te + 4].tolist(), n_sos=int((d0 == SOS).sum()), n_eoa=int((d0 == EOA).sum()), tail=d0[-3:].tolist()))

    sec_kept = [k for k in kept if k["layout"] == "sections"]
    info = dict(tag=args.tag, unit=args.unit, ctx=args.ctx, vocab_size=VOCAB_SIZE, layout="header EOD (SOS text SOA YUE2CODEC codes EOA)* 0 | whole: header EOD SOA YUE2CODEC codes EOA 0",
                ids=dict(EOD=EOD, SOA=SOA, EOA=EOA, YUE2CODEC=YUE2CODEC, SOS=SOS, YUE2_BASE=YUE2_BASE), thresholds=dict(min_score=args.min_score, min_frac=args.min_frac, min_spw=args.min_spw),
                shards_used=sorted({r["shard"] for r in recs}), test_shards=sorted(test), n_meta=len(recs), dropped=dict(drop), alignment=dict(why), kept=len(kept), aligned=len(sec_kept),
                sections_per_song_p50_p90=[float(np.percentile([len(k["sections"]) for k in sec_kept], q)) for q in (50, 90)] if sec_kept else None,
                section_seconds_p10_p50_p90=[float(np.percentile([(s["f1"] - s["f0"]) / FPS for k in sec_kept for s in k["sections"]], q)) for q in (10, 50, 90)] if sec_kept else None,
                val_removed_lyric_leak=len(leak), train_hours=round(sum(k["audio_seconds"] for k in train) / 3600, 1), train_creators=len({k["creator"] for k in train}))
    for split, rows in (("train", train), ("val", val)): info[split] = build(split, rows)
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(man / f"{args.tag}_stage1_summary.json", "w"), indent=2); json.dump(info, open(ROOT / "data/binidx" / f"{args.tag}_stage1_binidx.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
