#!/usr/bin/env python3
"""Strip LRC text that is never sung from the cursor-LM manifests (the letter aligner forces every letter onto the audio).

Per line, in order:
  tags        【singer】 [C+I+G] segments                               -> removed
  furigana    心（こころ） (kana reading after kanji)                    -> the reading
  credits     "作词：…", "混音 Mixed by …", "bgm原曲：《…》"             -> line dropped
  roles       short CJK prefix + colon ("张生：小姐…")                  -> prefix removed, the sung text kept
  split       "<original> / <translation>" (/ ｜ |), one side Han-only -> the Han-only side removed
  paren       (Han) after a line with no Han outside                    -> removed ("Oh No（哦不）"); all-paren lines kept (backing vocals)
  korean      Han runs inside Hangul lines                              -> removed
  trailing    kana / Latin line + trailing Han run (>= 3 after a space, >= 6 glued) -> removed, but only in songs where
              that pattern is common (>= 25 % of lines): a translated song, not one Chinese song with an English hook
  title       first 2 lines with " - ", 《》 or equal to the file name  -> dropped
Val rows whose first 12 cleaned lines also occur in the train manifest are dropped (same song uploaded twice).
usage: clean_lyrics.py --train data/real_corpus/manifest/pilot_clm_train.jsonl --val data/real_corpus/manifest/slim_val.jsonl
       (writes *_clean.jsonl next to them and prints counts + random before/after examples)
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path

HAN = r"一-鿿㐀-䶿"; KANA = r"぀-ヿ"; HANG = r"가-힯"
TAG = re.compile(r"【[^】]*】|\[[^\]]*\]")
CREDIT_KW = re.compile(r"作词|作曲|编曲|作編曲|詞|曲：|填词|词：|曲:|词:|混音|缩混|母带|录音|錄音|制作|製作|监制|監製|和声|和音|出品|发行|發行|企划|策划|"
                       r"统筹|統籌|宣传|封面|美工|后期|後期|原唱|翻唱|演唱|吉他|贝斯|貝斯|鼓：|键盘|鍵盤|弦乐|弦樂|配唱|调音|修音|人声|版权|版權|"
                       r"原曲|原作|改编|改編|作曲家|歌手|配音|策劃|助理|出版|OP|SP|ISRC|CV|[Pp]roduc|[Aa]rrang|[Cc]ompos|[Ll]yric|[Mm]ix|"
                       r"[Mm]aster|[Rr]ecord|[Vv]ocal|[Gg]uitar|[Bb]ass|[Dd]rum|[Ss]trings|[Pp]ublish|[Ss]upervis|[Ee]ngineer")
COLON = re.compile(r"^\s*([^:：\[\]]{1,24}?)\s*[:：]\s*(.*)$")
LEAD_CREDIT = re.compile(r"^\s*(作词|作曲|编曲|作编曲|填词|混音|缩混|母带|录音|制作人|制作|监制|和声|出品|发行|企划|策划|统筹|原唱|演唱|吉他|贝斯|"
                         r"弦乐|配唱|人声|版权|原曲|改编|OP|SP|ISRC)")
SHORT_CREDIT = set("词 曲 编 混 唱 歌 词作 曲作 演唱 原唱 主唱 鼓 弦 词曲 作 编曲 监制 制作 和声 吉他 贝斯 键盘 统筹 文案 海报 视频 剪辑 PV MV 翻 调 修".split())
FURI = re.compile(rf"([{HAN}々]+)[(（]([{KANA}]+)[)）]")
PAREN = re.compile(r"[(（][^()（）]*[)）]")
SPLIT = re.compile(r"\s*(?:\s/\s|/|｜|\|)\s*")
TRAIL_SP = re.compile(rf"(?<=\S)[\s　,，.。!！?？~]+[{HAN}，。、！？\s]{{3,}}$")
TRAIL_GLUED = re.compile(rf"(?<=[{KANA}a-zA-Z])[{HAN}]{{6,}}[，。、！？]*$")
has = lambda pat, t: re.search(f"[{pat}]", t) is not None
norm = lambda t: re.sub(r"[\W_]+", "", t.lower())


def han_only(t): return has(HAN, t) and not has(KANA, t) and not has(HANG, t) and not re.search(r"[a-zA-Z]", t)


def clean_line(t, stats):
    t0 = t; t = TAG.sub(" ", t).strip()
    t2 = FURI.sub(r"\2", t)                                   # 心（こころ） -> こころ: the reading is what is sung
    if t2 != t: stats["furigana"] += 1; t = t2
    m = COLON.match(t)
    if m:
        if CREDIT_KW.search(m.group(1)) or m.group(1).strip() in SHORT_CREDIT or not m.group(2).strip(): stats["credit"] += 1; return ""
        if len(m.group(1)) <= 6 and re.fullmatch(rf"[{HAN}{KANA}{HANG}\w·]+", m.group(1)) and not re.search(r"[a-zA-Z]", m.group(1)):
            stats["role"] += 1; t = m.group(2).strip()
    if LEAD_CREDIT.match(t): stats["credit"] += 1; return ""
    parts = [p for p in SPLIT.split(t) if p.strip()]
    if len(parts) == 2 and han_only(parts[0]) != han_only(parts[1]):
        stats["split"] += 1; t = parts[1] if han_only(parts[0]) else parts[0]
    outside = PAREN.sub("", t)
    if outside.strip() and not has(HAN, outside):
        t2 = PAREN.sub(lambda p: "" if has(HAN, p.group(0)) else p.group(0), t)
        if t2 != t: stats["paren"] += 1; t = t2
    if has(HANG, t) and has(HAN, t):
        stats["korean"] += 1; t = re.sub(f"[{HAN}]+", " ", t)
    if t != t0: stats["lines_changed"] += 1
    return re.sub(r"\s+", " ", t).strip()


def clean_row(r, stats):
    L = r.get("lines") or []
    if not L: return r
    stem = norm(Path(r["audio_path"]).stem); out = []
    trailing = sum(bool((has(KANA, l["text"]) or re.search(r"[a-zA-Z]{2,}", l["text"]) or has(HANG, l["text"])) and
                        (TRAIL_SP.search(l["text"]) or TRAIL_GLUED.search(l["text"]))) for l in L) >= 0.25 * len(L)
    for i, l in enumerate(L):
        t = l["text"]
        if i < 2 and (" - " in t or "《" in t or (stem and norm(t) == stem)): stats["title"] += 1; continue
        t = clean_line(t, stats)
        if trailing and t and not han_only(t):
            t2 = TRAIL_GLUED.sub("", TRAIL_SP.sub("", t)).strip()
            if t2 and t2 != t: stats["trailing"] += 1; t = t2
        if t: out.append(dict(l, text=t))
        else: stats["dropped_lines"] += 1
    stats["songs_changed"] += out != L
    return dict(r, lines=out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--examples", type=int, default=40)
    args = ap.parse_args(); random.seed(0); keys = set(); ex = []
    for which in ("train", "val"):
        src = Path(getattr(args, which)); dst = src.with_name(src.stem + "_clean.jsonl"); stats = Counter(); n = kept = 0
        with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
            for line in f:
                r = json.loads(line); n += 1; c = clean_row(r, stats)
                key = tuple(norm(l["text"]) for l in c.get("lines", [])[:12]) if len(c.get("lines") or []) >= 4 else None
                if which == "train" and key: keys.add(key)
                if which == "val" and key in keys: stats["val_dup_dropped"] += 1; continue
                if which == "train" and c["lines"] != (r.get("lines") or []) and random.random() < 0.002:
                    a = [l["text"] for l in r["lines"]]; b = [l["text"] for l in c["lines"]]
                    ex.append([x for x in a if x not in b][:3] + ["->"] + [y for y in b if y not in a][:3])
                g.write(json.dumps(c, ensure_ascii=False) + "\n"); kept += 1
        print(which, f"{n} rows -> {kept} {dst}", dict(stats), flush=True)
    for e in random.sample(ex, min(args.examples, len(ex))): print("  ", " | ".join(e))


if __name__ == "__main__":
    main()
