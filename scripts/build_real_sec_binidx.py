#!/usr/bin/env python3
"""Stage-1 corpus for the real-song run: section-interleaved docs (R3.1 layout) from scripts/prepare_real_yue2.py shards,
optionally mixed with the existing suno94k section manifest (data/manifests/yue2_sec_stage1_train.jsonl).

    world_tokens("[Genre] caption\\n[Lyrics]\\nfull lyrics\\n")  EOD  ( SOS world_tokens("[marker]\\nstanza\\n") SOA YUE2CODEC codes EOA )*  0

Sections come straight from the LRC line timestamps (ingest_lrc_corpus.py: lines [{t0, t1, text}]), grouped at >= 3 s gaps or
~20 s / 6 lines (see stanzas), so no forced alignment is needed; credit lines anywhere in the LRC are dropped. LRC carries no section names; marker = "chorus" for a stanza whose
normalised text occurs >= 2 times in the song, else "verse" (SongFormer labels can replace this later).
Caption, per song: tools/caption_fuse.py output when present (tags, or tags + description, hash-chosen; section directives
appended to the marker as "[chorus | directive]" for half of those songs), else cheap tags from metadata (language from the lyric
script, "live recording" from the title). Artists / albums / shows never enter the text.
Songs: 30 s .. --max-seconds, docs over --ctx dropped; test shard(s) excluded; 1 % val by id hash.
usage: build_real_sec_binidx.py [--ctx 12288] [--max-seconds 470] [--with-suno 1] [--tag real_sec]
"""
import argparse, collections, glob, hashlib, json, random, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools" / "RWKV-LM" / "RWKV-v7" / "train_temp"))
sys.path.insert(0, str(ROOT / "tools" / "RWKV-LM" / "RWKV-v5" / "tokenizer"))
from yue2_layout import EOD, SOA, EOA, YUE2CODEC, SOS, YUE2_BASE, N_CODES, VOCAB_SIZE  # noqa: E402
from build_stage1_manifest import prompt_text, VOCAB_TXT  # noqa: E402
from build_binidx import Builder, magic_prime, DTYPE  # noqa: E402
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402
from src.binidx import MMapIndexedDataset  # noqa: E402

REAL = Path("/models/rwkv-music/real_corpus/wangyiyun")
SUNO = ROOT / "data/yue2_corpus/suno94k"
CAPS = ROOT / "data/real_corpus/caption"
FPS = 25
HAN, KANA, HANGUL, LATIN = re.compile(r"[一-鿿]"), re.compile(r"[぀-ヿ]"), re.compile(r"[가-힯]"), re.compile(r"[A-Za-z]")


def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 2**32


def language(text):
    n = {k: len(r.findall(text)) for k, r in (("zh", HAN), ("ja", KANA), ("ko", HANGUL), ("en", LATIN))}
    if n["ja"] > 20: return "japanese"
    if n["ko"] > 20: return "korean"
    zh, en = n["zh"], n["en"] / 4                     # ~4 latin letters per word vs 1 han per syllable
    if zh + en == 0: return ""
    if zh > 3 * en: return "mandarin"
    if en > 3 * zh: return "english"
    return "mandarin, english"


def cheap_caption(r):
    tags = [t for t in (language(r["lyrics"]), "live recording" if r.get("meta", {}).get("is_live") else "") if t]
    return ", ".join(tags)


CREDIT_LINE = re.compile(r"^\s*[^:：]{1,16}\s*[:：]\s*\S")        # "编曲 : 某某", "Vocal Producer: X" (credits mid-LRC)
NOTICE_LINE = re.compile(r"^\s*[【\[(（].*(版权|授权|翻唱|商用|未经|著作权|监制|出品).*[】\])）]\s*$|版权所有|未经许可|未经授权|不得翻唱")
GAP_S, MAX_SEC_S, MAX_LINES = 3.0, 20.0, 6
ROLE = re.compile(r"词|曲|编|混|母带|录|鼓|吉他|贝斯|键盘|钢琴|弦|和声|和音|合音|监制|制作|统筹|总监|策划|出品|发行|顾问|宣传|PGM|OP|SP|舞|摄|灯光|音响|"
                  r"Program|Producer|Mix|Master|Engineer|Lyric|Compos|Arrang|Guitar|Bass|Drum|Piano|Keyboard|Strings|Vocal|Studio|Record", re.I)
SPEAKER = re.compile(r"^\s*([^:：]{1,40}?)\s*[:：]\s*(.*)$")


def clean_line(t):
    """None = drop (credit, notice, bare 'Hook:' header); else the sung text with a leading singer label ('合：', 'A+B：') removed."""
    if NOTICE_LINE.search(t): return None
    m = SPEAKER.match(t)
    if not m: return t
    label, rest = m.groups()
    if not rest.strip() or ROLE.search(label): return None
    if len(label) <= 6 or "+" in label or "合" in label: return rest.strip()      # singer / part label: keep the sung text
    return t


noise = lambda t: clean_line(t) is None


def stanzas(r):
    """-> [(marker, [lines], t_start)]: timed LRC lines (credit lines dropped) grouped into sections; a new section starts at a
    >= GAP_S silence, or once the current one is MAX_SEC_S long / MAX_LINES lines. marker = chorus if the text repeats in the song."""
    lines = [dict(l, text=clean_line(l["text"])) for l in (r.get("lines") or []) if not noise(l["text"])]
    out = []
    for l in lines:
        if out:
            ls, t0, t1 = out[-1]
            if l["t0"] - t1 < GAP_S and l["t0"] - t0 < MAX_SEC_S and len(ls) < MAX_LINES: ls.append(l["text"]); out[-1][2] = l["t1"]; continue
        out.append([[l["text"]], l["t0"], l["t1"]])
    norm = lambda ls: re.sub(r"\W+", "", "".join(ls)).lower()
    cnt = collections.Counter(norm(ls) for ls, _, _ in out)
    return [("chorus" if cnt[norm(ls)] >= 2 and len(norm(ls)) >= 6 else "verse", ls, t0) for ls, t0, _ in out]


def sections_of(r, n_frames, directives=None):
    """-> [(text, f0, f1)] covering [0, n_frames); boundary k = start of stanza k (intro audio goes to stanza 0)."""
    st = stanzas(r)
    if len(st) < 2: return None
    b = [0] + [min(n_frames, max(0, int(round(t0 * FPS)))) for _, _, t0 in st[1:]] + [n_frames]
    out, pend = [], ""
    for k, (marker, ls, _) in enumerate(st):
        d = directives(st[k][2]) if directives else ""
        txt = pend + f"[{marker}{(' | ' + d) if d else ''}]\n" + "\n".join(ls) + "\n"; f0, f1 = b[k], b[k + 1]
        if f1 - f0 < FPS and k < len(st) - 1: pend = txt; b[k + 1] = f0; continue
        pend = ""; out.append((txt, f0, f1))
    if pend and out: t, f0, f1 = out[-1]; out[-1] = (t + pend, f0, f1)
    if not out or out[0][1] != 0 or out[-1][2] != n_frames or any(out[i][2] != out[i + 1][1] for i in range(len(out) - 1)): return None
    if any(f1 <= f0 for _, f0, f1 in out): return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=12288); ap.add_argument("--min-seconds", type=float, default=30); ap.add_argument("--max-seconds", type=float, default=470)
    ap.add_argument("--test-shards", default="0"); ap.add_argument("--with-suno", type=int, default=1); ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--tag", default="real_sec"); ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "data/binidx"))
    args = ap.parse_args(); t0 = time.time(); tok = TRIE_TOKENIZER(str(VOCAB_TXT))
    test = {int(x) for x in args.test_shards.split(",") if x}
    caps = {}
    for f in glob.glob(str(CAPS / "*.captions.jsonl")):
        for l in open(f, encoding="utf-8"): c = json.loads(l); caps[c["id"]] = c
    drop = collections.Counter(); kept = []
    for jl in sorted(REAL.glob("shard-*.jsonl")):
        n = int(jl.stem.split("-")[1])
        if n in test or not jl.with_name(f"{jl.stem}.tokens.npz").exists(): continue
        for l in open(jl, encoding="utf-8"):
            r = json.loads(l)
            if "n_frames" not in r: drop["encode_error"] += 1; continue
            d = r["n_frames"] / FPS
            if d < args.min_seconds: drop["too_short"] += 1; continue
            if d > args.max_seconds: drop["too_long"] += 1; continue
            inst = r["lyrics"] == "[Instrumental]"
            c = caps.get(r["id"]); directives = None
            if c:
                cap = c["tags"] if h01(r["id"] + "cap") < 0.5 else c["tags"] + ". " + c["description"]
                if h01(r["id"] + "dir") < 0.5:
                    secs_c = c["sections"]
                    directives = lambda t, secs_c=secs_c: next((s["directive"] for s in secs_c if s["start"] <= t + 0.5 < s["end"] and s["directive"]), "")
            else: cap = cheap_caption(r)
            lyr = "\n".join(x if not x.strip() else clean_line(x) for x in r["lyrics"].splitlines() if not x.strip() or clean_line(x) is not None)
            header = prompt_text(cap, "[Instrumental]" if inst else lyr); nh = len(tok.encode(header))
            secs = None if inst else sections_of(r, r["n_frames"], directives)
            rec = dict(src="real", id=r["id"], shard=n, n_frames=r["n_frames"], header=header, n_header=nh, captioned=bool(c), audio_seconds=d)
            if secs:
                st = [tok.encode(t) for t, _, _ in secs]
                n_doc = nh + 1 + sum(1 + len(s) + 2 + (f1 - f0) + 1 for s, (_, f0, f1) in zip(st, secs)) + 1
                rec.update(layout="sections", sections=[dict(text=t, f0=f0, f1=f1) for t, f0, f1 in secs], n_doc=n_doc)
            else:
                rec.update(layout="whole", n_doc=nh + 5 + r["n_frames"]); drop["whole_layout"] += 1
            if rec["n_doc"] > args.ctx: drop["over_ctx"] += 1; continue
            kept.append(rec)
    n_real = len(kept)
    if args.with_suno:
        for l in open(ROOT / "data/manifests/yue2_sec_stage1_train.jsonl", encoding="utf-8"):
            r = json.loads(l); r["src"] = "suno"
            if r["n_doc"] <= args.ctx: kept.append(r)
    val = [k for k in kept if k["src"] == "real" and h01(k["id"] + "val") < args.val_frac]; vid = {k["id"] for k in val}
    train = [k for k in kept if k["id"] not in vid]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True); man = ROOT / "data/manifests"
    for name, rows in (("train", train), ("val", val)):
        with open(man / f"{args.tag}_stage1_{name}.jsonl", "w", encoding="utf-8") as f:
            for k in rows: f.write(json.dumps({x: v for x, v in k.items() if x != "sections"}, ensure_ascii=False) + "\n")

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
        rows = list(rows); random.Random(args.shuffle_seed).shuffle(rows); prefix = str(Path(args.out_dir) / f"{args.tag}_stage1_{split}")
        by = collections.defaultdict(list)
        for i, r in enumerate(rows): by[(r["src"], r["shard"])].append(i)
        docs = [None] * len(rows)                    # int32 docs held in RAM: ~4 B/token (fine up to a few B tokens on 250 GB)
        for (src, shard), idx in sorted(by.items()):
            z = np.load((REAL if src == "real" else SUNO) / f"shard-{shard:05d}.tokens.npz")
            for i in idx:
                r = rows[i]; codes = z[r["id"]].astype(np.int64)
                assert codes.shape[0] == r["n_frames"] and codes.min() >= 0 and codes.max() < N_CODES, r["id"]
                docs[i] = encode(r, codes)
        b = Builder(prefix)
        for d in docs: b.add_document(d)
        b.finalize(); ds = MMapIndexedDataset(prefix); d0 = np.asarray(ds[0])
        return dict(documents=len(docs), tokens=b.n_tokens, magic_prime=magic_prime(b.n_tokens, args.ctx) if b.n_tokens > args.ctx * 20 else None,
                    bin=prefix + ".bin", real_docs=sum(r["src"] == "real" for r in rows), section_docs=sum(r["layout"] == "sections" for r in rows),
                    captioned=sum(bool(r.get("captioned")) for r in rows), verify=dict(len=int(d0.size), n_sos=int((d0 == SOS).sum()), tail=d0[-3:].tolist()))

    info = dict(tag=args.tag, ctx=args.ctx, vocab_size=VOCAB_SIZE, test_shards=sorted(test), dropped=dict(drop), real_kept=n_real,
                suno_kept=len(kept) - n_real, real_train_hours=round(sum(k["audio_seconds"] for k in train if k["src"] == "real") / 3600, 1))
    for split, rows in (("train", train), ("val", val)): info[split] = build(split, rows)
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(Path(args.out_dir) / f"{args.tag}_stage1_binidx.json", "w"), indent=2); print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
