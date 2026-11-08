#!/usr/bin/env python3
"""NetEase YRC (word-timed karaoke lyrics, `<name>.lrc.yrc` next to the audio in /models/musicdl/ncm_vip/<lang>/<id>/).

Format per line:  [line_start_ms,line_dur_ms](w_start_ms,w_dur_ms,0)word(w_start_ms,w_dur_ms,0)word ...
Words carry their own spaces / punctuation; CJK lines are one character per word. Some lines are machine-filled (every word
the same duration: credits, untimed imports) -> flagged `uniform`, not to be trusted as timing.

parse(path) -> [dict(t0, t1, text, words=[(t0, t1, w)], uniform)] in seconds, credit lines ("作曲 : ...") dropped.
"""
import re, sys

LINE = re.compile(r"^\[(\d+),(\d+)\](.*)$")
WORD = re.compile(r"\((\d+),(\d+),\d+\)([^(]*)")
CREDIT = re.compile(r"^\s*(作曲|作词|编曲|制作人|和声|混音|母带|录音|吉他|贝斯|鼓|键盘|弦乐|监制|出品|发行|词|曲|Lyrics|Composer|Producer|"
                    r"Arranged|Arranger|Written|Mixed|Mastered|OP|SP)\b.*[:：]", re.I)


def parse(path):
    out = []
    for raw in open(path, encoding="utf-8", errors="replace"):
        m = LINE.match(raw.strip())
        if not m: continue
        ws = [(int(a) / 1000, (int(a) + int(b)) / 1000, w) for a, b, w in WORD.findall(m.group(3))]
        text = "".join(w for _, _, w in ws).strip()
        if not text or CREDIT.match(text): continue
        durs = [round(b - a, 3) for a, b, w in ws if w.strip()]
        uniform = len(durs) >= 3 and len(set(durs)) == 1
        out.append(dict(t0=int(m.group(1)) / 1000, t1=(int(m.group(1)) + int(m.group(2))) / 1000, text=text, words=ws, uniform=uniform))
    return out


if __name__ == "__main__":
    for l in parse(sys.argv[1])[:int(sys.argv[2]) if len(sys.argv) > 2 else 8]:
        print(f"{l['t0']:7.2f}-{l['t1']:7.2f} {'U' if l['uniform'] else ' '} {l['text'][:60]}  |  " +
              " ".join(f"{w.strip()}@{a:.2f}" for a, b, w in l["words"][:6]))


def load_index(path=None, only_good=True):
    """-> {id: yrc path} from tools/yrc_index.py output"""
    import json
    from pathlib import Path
    p = path or Path(__file__).resolve().parents[1] / "data/real_corpus/yrc/yrc_index.jsonl"
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        if not only_good or r.get("good"): out[r["id"]] = r["path"]
    return out


def overlay_starts(ids, st, tab, ylines, rate, space=1, tol=1.0, max_word=2.0):
    """Replace MMS letter starts (precompute shards: ids, st [frames], tab rows (f0, f1raw, f1c, i0, i1)) by YRC timing, line by
    line. A shard line takes the nearest non-uniform YRC line whose start is within tol s; its romanised words must match the
    shard line's words one to one (else the line keeps MMS). Letters of a word are spread over the YRC word's [start, end),
    capped at max_word s (some YRC files hold one word for 10-40 s); spaces and the separator before a line take the next
    letter's start. A line whose YRC letters start past the shard line's end f1c + tol keeps MMS (YRC from another version
    of the song: 2026-11-08 check, 7 % of YRC songs had letters 2-37 s late, so the line clamps that followed corrupted the
    cursor targets). -> (new st float32 [frames], lines replaced)"""
    import numpy as np
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from vae.data import romanise
    st = np.asarray(st, np.float32).copy(); n_rep = 0
    good = [y for y in ylines if not y["uniform"]]
    for t in tab:
        f0 = t[0] / rate; i0, i1 = int(t[3]), int(t[4])
        cand = [y for y in good if abs(y["t0"] - f0) < tol]
        if not cand or i1 <= i0: continue
        y = min(cand, key=lambda y: abs(y["t0"] - f0))
        toks = []                                                         # (start, end) per romanised token (CJK char = 1)
        for a, b, w in y["words"]:
            r = romanise(w).split()
            for k in range(len(r)): toks.append((a + (b - a) * k / len(r), a + (b - a) * (k + 1) / len(r)))
        words, cur = [], None                                             # shard words = runs of non-space letter indices
        for k in range(i0, i1):
            if ids[k] == space: cur = None; continue
            if cur is None: cur = []; words.append(cur)
            cur.append(k)
        if len(words) != len(toks): continue
        new = np.full(i1 - i0, np.nan)
        for wk, (a, b) in zip(words, toks):
            b = min(max(b, a + 0.02), a + max_word)
            for j, k in enumerate(wk): new[k - i0] = (a + (b - a) * j / len(wk)) * rate
        if np.nanmax(new) > t[2] + tol * rate: continue
        nxt = np.nan
        for k in range(len(new) - 1, -1, -1):
            if np.isnan(new[k]): new[k] = nxt
            else: nxt = new[k]
        st[i0:i1] = new; n_rep += 1
        if i0 > 0 and ids[i0 - 1] == space: st[i0 - 1] = new[0]
    return st, n_rep
