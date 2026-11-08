"""Text prompt for LiuBai-H whole-song generation (user 2026-11-06/07: keep the 3B, prompt = music metadata + lyrics, RWKV g1
chat template, the request in any language / wording, field names = the metadata attribute names).

The prompt is RWKV world-vocab text in front of the audio rows (LiuBaiH txt prefix: causal, the A stream reads it through the
WKV state; the F rows see it through S). Every field optional:

    User: Make a song.                     <- request: one of REQUESTS (many languages / wordings), or any text at inference
    title: Invisible Boy
    artists: Tori Amos
    album: Unrepentant Geraldines
    lang: eng
    is_live: false
    dur: 280.2
    lines:
    [00:10.4] You are, are not just
    [00:16.7] An invisible boy
    ...
    <blank line>
    Assistant: <think></think>

Field names and values are those of the precompute meta (tools/precompute_msvae_full.py .meta.jsonl): title, artists, album,
lang (ISO 639-3), is_live, dur (s), lines ([mm:ss.s] text, or text only). RWKV g1 chat template: "User: ...", one blank line,
"Assistant: <think></think>" (empty think = non-thinking mode, as g1 was trained); the user turn has single newlines only;
the audio rows follow the template. Training drops fields at random (each meta field p_field, timestamps p_ts, the lyric
text p_lyr_text) and draws the request wording at random (English half the time, else the song's language or another one),
so any subset / wording works at inference. null_prompt(request) = the request alone = the CFG null. The lyric cursor
(letters) is separate and unchanged.
"""
import random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "tools/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt"
_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        sys.path.insert(0, str(ROOT / "tools/RWKV-LM/RWKV-v5/tokenizer"))
        from rwkv_tokenizer import TRIE_TOKENIZER
        _tok = TRIE_TOKENIZER(str(VOCAB))
    return _tok


def ts(t):
    m, s = divmod(max(0.0, t), 60); return f"[{int(m):02d}:{s:04.1f}]"


REQUESTS = dict(
    eng=["Make a song.", "Write and sing a song.", "Create a song.", "Please make me a song.", "Compose a song with these lyrics.",
         "Generate a song.", "Sing this song.", "Can you make a song for me?", "Produce a full song.", "Make music for these lyrics."],
    cmn=["做一首歌。", "帮我写一首歌。", "生成一首歌曲。", "请为这些歌词作曲并演唱。", "来一首歌吧。", "创作一首歌。", "唱这首歌。"],
    yue=["整一首歌。", "幫我寫首歌。", "唱呢首歌。"],
    jpn=["曲を作って。", "歌を作ってください。", "この歌詞で曲を作って。", "一曲お願いします。", "歌って。"],
    kor=["노래를 만들어 줘.", "이 가사로 노래를 만들어 주세요.", "노래 한 곡 부탁해.", "노래해 줘."],
    spa=["Haz una canción.", "Crea una canción.", "Canta esta canción."],
    fra=["Fais une chanson.", "Crée une chanson.", "Chante cette chanson."],
    deu=["Mach ein Lied.", "Schreib einen Song.", "Sing dieses Lied."],
    ita=["Fai una canzone.", "Canta questa canzone."],
    por=["Faça uma música.", "Cante esta música."],
    rus=["Сделай песню.", "Спой эту песню."],
    tha=["แต่งเพลงให้หน่อย"], vie=["Hãy làm một bài hát."])
THINK = "Assistant: <think></think>"


def request(rng=None, lang=None):
    """training: English half the time, else the song's language (if listed) with p 0.6, else any listed language"""
    if rng is None: return REQUESTS["eng"][0]
    if rng.random() < 0.5: L = "eng"
    elif lang in REQUESTS and rng.random() < 0.6: L = lang
    else: L = rng.choice(sorted(REQUESTS))
    return rng.choice(REQUESTS[L])


def build(meta, lines, dur, rng=None, p_field=0.0, p_ts=0.0, p_lyr_text=0.0, max_chars=6000, req=None):
    """meta: dict (title, artists, album, lang, is_live); lines: [(t0 seconds relative to the audio start, text)] in order;
    dur: seconds; req: request text (None: random in training, "Make a song." otherwise).
    rng None = deterministic (keep everything, timestamps iff p_ts < 1)."""
    keep = (lambda p: True) if rng is None else (lambda p: rng.random() >= p)
    one = lambda v: " ".join(str(v).split())                              # no newlines inside a field
    out = ["User: " + one(req if req is not None else request(rng, meta.get("lang")))]
    for key in ("title", "artists", "album", "lang"):
        v = one(meta.get(key) or "")
        if v and keep(p_field): out.append(f"{key}: {v}")
    if meta.get("is_live") is not None and keep(p_field): out.append(f"is_live: {'true' if meta['is_live'] else 'false'}")
    if dur and keep(p_field): out.append(f"dur: {dur:.1f}")
    if lines and keep(p_lyr_text):
        with_ts = (p_ts < 1) if rng is None else rng.random() >= p_ts
        out.append("lines:"); n = sum(len(x) + 1 for x in out)
        for t0, text in lines:
            row = f"{ts(t0)} {one(text)}" if with_ts else one(text)
            if n + len(row) > max_chars: break
            out.append(row); n += len(row) + 1
    return "\n".join(out) + "\n\n" + THINK


def null_prompt(req=None, rng=None, lang=None):
    return "User: " + " ".join((req if req is not None else request(rng, lang)).split()) + "\n\n" + THINK


def encode(text): return tokenizer().encode(text)
