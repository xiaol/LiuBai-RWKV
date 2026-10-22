"""Word-level lyric timing for the lyric renderer (Test 2, 2026-10-22: does the renderer use lyrics when the timing is given?).

MMS_FA (torchaudio, with <star>) forced-aligns the romanised words of every line that lies completely inside the window to the
window's own audio; each word then enters the renderer sequence at its aligned start frame (12.5 Hz) instead of the whole line
at its LRC start. Lines cut by the window edge keep line-level placement (their audio is incomplete, aligning them would
squeeze the missing words into the edge). <star> tokens between lines and at both ends absorb ad-libs, backing vocals and
the cut lines. CJK is one syllable per "word" (vae.data.romanise), so Chinese gets per-character timing.

lines in : per sample [(f0, ids, f1)] at the latent rate, f0 may be < 0 and f1 > n_frames (cut lines); ids = vae.data letters
           with id 1 (' ') between words
lines out: per sample [(frame, ids)] with one entry per aligned word + the cut lines (start clamped to 0)
"""
import torch
import torchaudio
import torchaudio.functional as AF
from .model import LETTERS

SPACE = 1                                                   # vae.data.encode_text id of ' '


class WordAligner:
    def __init__(s, dev="cuda", sr=24000, rate=12.5):
        from torchaudio.pipelines import MMS_FA
        # torchaudio's with_star wrapper only supports batch 1 (it appends a [1,T,1] zero logit); do the same per batch here
        s.model = MMS_FA.get_model(with_star=False).to(dev).eval().requires_grad_(False)
        labels = MMS_FA.get_labels(star="*"); d = {c: i for i, c in enumerate(labels)}
        s.star = d["*"]; assert s.star == len(labels) - 1, "star must be the last label"; s.map = {i + 1: d[c] for i, c in enumerate(LETTERS) if c in d}   # our letter id -> MMS id (space has none)
        s.res = torchaudio.transforms.Resample(sr, 16000).to(dev); s.rate = rate; s.dev = dev
        s.stats = dict(windows=0, aligned_words=0, fallback=0)

    @staticmethod
    def words(ids):
        out, cur = [], []
        for i in ids:
            if i == SPACE:
                if cur: out.append(cur); cur = []
            else: cur.append(i)
        if cur: out.append(cur)
        return out

    @torch.no_grad()
    def __call__(s, wav, lines, n_frames):
        x = s.res(wav.float().mean(1))                      # [B, S16]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            em, _ = s.model(x)
        em = em.float(); em = torch.cat([em, torch.zeros_like(em[..., :1])], -1)   # <star> logit = 0, as in torchaudio
        em = torch.log_softmax(em, -1); T = em.shape[1]; sec_per_em = x.shape[1] / 16000 / T
        out = []
        for b, ln in enumerate(lines):
            s.stats["windows"] += 1
            full = [l for l in ln if len(l) > 2 and l[0] >= 0 and l[2] <= n_frames]
            cut = [(max(0, min(l[0], n_frames - 1)), l[1]) for l in ln if not (len(l) > 2 and l[0] >= 0 and l[2] <= n_frames)]
            targets, units = [s.star], []                   # units: (our word ids, n MMS tokens)
            for l in full:
                for w in s.words(l[1]):
                    m = [s.map[c] for c in w if c in s.map]
                    if m: targets += m; units.append((w, len(m)))
                targets.append(s.star)
            if not units or len(targets) >= T // 2:
                out.append(cut + [(max(0, min(l[0], n_frames - 1)), l[1]) for l in full]); s.stats["fallback"] += bool(units); continue
            try:
                ali, sc = AF.forced_align(em[b:b + 1], torch.tensor([targets], dtype=torch.int32, device=em.device), blank=0)
                spans = AF.merge_tokens(ali[0], sc[0].exp())
            except Exception:
                out.append(cut + [(max(0, min(l[0], n_frames - 1)), l[1]) for l in full]); s.stats["fallback"] += 1; continue
            k = 1; res = list(cut); ui = 0                 # spans[0] = leading star; walk targets in order: per word n tokens,
                                                            # then one star after each line
            for l in full:
                n_w = sum(1 for w in s.words(l[1]) if any(c in s.map for c in w))
                for _ in range(n_w):
                    w, n = units[ui]; ui += 1
                    f = int(spans[k].start * sec_per_em * s.rate); k += n
                    res.append((max(0, min(f, n_frames - 1)), w))
                k += 1                                      # the star after the line
            s.stats["aligned_words"] += len(units)
            out.append(res)
        return out
