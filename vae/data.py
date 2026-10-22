"""Training crops for the VAE pilot from the NetEase manifest (tools/ingest_lrc_corpus.py rows: audio_path, duration, lines).

Crop = SEG seconds of 24 kHz stereo. 80 % of crops start just before a random LRC line (so most crops contain singing); the CTC
target is the romanised text of every line that starts inside the crop and ends (next line's start) before the crop end + 0.3 s,
i.e. lines that are sung completely inside it. Crops whose window has partial lines still train the codec; `ctc_ok` marks the
ones with >= 1 complete line; the CTC loss only covers frames up to the end of the last complete line (`ctc_frames`, 50 Hz).
Manifest rows [0, holdout) (= real-corpus shard 0, the test shard) are never used.
"""
import json, random, re, unicodedata
from math import gcd
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from .model import LETTERS

SR = 24000
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_ur, _cache = None, {}
L2I = {c: i + 1 for i, c in enumerate(LETTERS)}


def roman_char(c):
    global _ur
    if c not in _cache:
        if _ur is None:
            import uroman; _ur = uroman.Uroman()
        t = _ur.romanize_string(c); t = "".join(x for x in unicodedata.normalize("NFKD", t) if not unicodedata.combining(x))
        _cache[c] = re.sub(r"[^a-z']", "", t.lower())
    return _cache[c]


def romanise(text):
    """CJK: one syllable per character, space-separated (wo ai ni); Latin: lower-case words; everything else dropped."""
    out = []
    for tok in re.findall(r"[぀-ヿ㐀-䶿一-鿿가-힯]|[^\s぀-ヿ㐀-䶿一-鿿가-힯]+", text):
        if CJK.match(tok): r = roman_char(tok)
        else:
            r = "".join(x for x in unicodedata.normalize("NFKD", tok.replace("’", "'")) if not unicodedata.combining(x)).lower()
            r = re.sub(r"[^a-z']", "", r)
        if r: out.append(r)
    return " ".join(out)


def encode_text(s): return [L2I[c] for c in s if c in L2I]


def read_crop(path, t0, seg):
    info = sf.info(path); sr = info.samplerate
    a0 = int(t0 * sr); n = int(seg * sr) + 64
    x, _ = sf.read(path, start=a0, stop=min(a0 + n, info.frames), dtype="float32", always_2d=True)
    x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]
    if sr != SR: g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    S = int(seg * SR); x = x[:S]
    if len(x) < S: x = np.pad(x, ((0, S - len(x)), (0, 0)))
    return x.T


class Crops(torch.utils.data.IterableDataset):
    def __init__(s, manifest, seg=8.0, holdout=1000, p_line=0.8, seed=0, max_rows=0, with_lines=False, rate=12.5):
        rows = [json.loads(l) for l in open(manifest, encoding="utf-8")][holdout:]
        if max_rows: rows = rows[:max_rows]
        s.rows = [dict(audio_path=r["audio_path"], duration=r["duration"], lines=[(l["t0"], l["t1"], l["text"]) for l in r.get("lines") or []])
                  for r in rows if r.get("duration", 0) > seg + 2]
        s.seg, s.p_line, s.seed, s.with_lines, s.rate = seg, p_line, seed, with_lines, rate

    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); rng = random.Random(s.seed * 1000 + (wi.id if wi else 0) + random.randrange(1 << 30))
        while True:
            r = rng.choice(s.rows); L = r["lines"]
            if L and rng.random() < s.p_line:
                l0 = rng.choice(L); t0 = max(0.0, min(l0[0] - rng.uniform(0.1, 1.2), r["duration"] - s.seg - 0.1))
            else: t0 = rng.uniform(0, r["duration"] - s.seg - 0.1)
            t1 = t0 + s.seg
            full = []                                                     # consecutive complete lines from the first one after t0
            for l in L:
                if l[0] < t0: continue
                if l[1] > t1 + 0.3: break
                full.append(l)
            text = romanise(" ".join(l[2] for l in full)); tgt = encode_text(text)
            ctc_frames = int(min(s.seg, (full[-1][1] if full else t0) - t0 + 0.3) * 50)   # CTC sees only [t0, end of last full line] (50 Hz)
            try: wav = read_crop(r["audio_path"], t0, s.seg)
            except Exception: continue
            if np.abs(wav).max() < 1e-4: continue
            ok = bool(full) and len(tgt) > 0 and len(tgt) <= ctc_frames // 2
            if not s.with_lines:
                yield torch.from_numpy(wav), torch.tensor(tgt, dtype=torch.long), ok, ctc_frames; continue
            # renderer conditioning: every line overlapping the crop (cut ones too: the renderer must sing them),
            # as (start frame at the latent rate, clamped to 0, letter ids)
            lines = []
            for l in L:
                if l[1] <= t0 + 0.05 or l[0] >= t1 - 0.05: continue
                ids = encode_text(romanise(l[2]))
                if ids: lines.append((int(round((l[0] - t0) * s.rate)), ids, int(round((l[1] - t0) * s.rate))))   # raw start/end frame:
                                                                         # < 0 / > T marks a cut line (vae/align_words.py)
            yield torch.from_numpy(wav), torch.tensor(tgt, dtype=torch.long), ok, ctc_frames, lines


def collate(batch):
    if len(batch[0]) == 5:                             # with_lines: per-sample line lists stay plain Python
        return (*collate([b[:4] for b in batch]), [b[4] for b in batch])
    wav = torch.stack([b[0] for b in batch]); tg = [b[1] for b in batch]
    lens = torch.tensor([len(t) for t in tg]); ok = torch.tensor([b[2] for b in batch]); cf = torch.tensor([b[3] for b in batch])
    flat = torch.cat(tg) if lens.sum() else torch.zeros(0, dtype=torch.long)
    return wav, flat, lens, ok, cf
