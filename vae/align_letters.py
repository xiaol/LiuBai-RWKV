"""Letter-level lyric timing for the cursor LM (vae/cursor_lm.py): MMS_FA forced alignment, one start time per letter.

Each complete line is aligned on its own, inside its LRC span widened by --pre / --post seconds, with <star> at both ends
(absorbs ad-libs, backing vocals, neighbouring lines). Per-line alignment matters: aligning a whole window at once matched
repeated lyrics ("won't you wait ..." x4) to an earlier repeat sung by a cut line. Letters without an MMS label and the spaces
between words / lines take the start of the next aligned letter.

out per sample: (ids [letter ids incl. spaces, complete lines only], starts float frames at the latent rate [len(ids)],
                 mask bool [n_frames]: False on frames sung by lines cut by the window edge - their text is not in ids, so the
                 flow / advance losses skip them)
Line spans are capped at 2 s + 0.2 s per letter (an LRC line ends where the next one starts, so a line before an
instrumental break would otherwise span the break). A line whose alignment fails gets its letters spread uniformly over its LRC span (stats["fallback"] counts lines).
"""
import numpy as np
import torch
import torchaudio.functional as AF
from .align_words import WordAligner, SPACE


class LetterAligner(WordAligner):
    def __init__(s, dev="cuda", pre=0.6, post=0.4, **kw):
        super().__init__(dev, **kw); s.pre, s.post = pre, post
        s.stats = dict(windows=0, lines=0, aligned_words=0, fallback=0)

    @torch.no_grad()
    def __call__(s, wav, lines, n_frames):
        x = s.res(wav.float().mean(1))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            em, _ = s.model(x)
        em = em.float(); em = torch.cat([em, torch.zeros_like(em[..., :1])], -1)
        em = torch.log_softmax(em, -1); T = em.shape[1]; em_per_fr = T / (x.shape[1] / 16000 * s.rate)
        out = []
        for b, ln in enumerate(lines):
            s.stats["windows"] += 1                                     # LRC end = next line's start: cap long spans (breaks)
            ln = [(l[0], l[1], min(l[2], l[0] + int(s.rate * (2.0 + 0.2 * len(l[1]))))) if len(l) > 2 else l for l in ln]
            is_full = lambda l: len(l) > 2 and l[0] >= 0 and l[2] <= n_frames
            full = sorted((l for l in ln if is_full(l)), key=lambda l: l[0])
            mask = np.ones(n_frames, bool)
            for l in ln:
                if not is_full(l): mask[max(0, l[0]):max(0, min(n_frames, l[2] if len(l) > 2 else n_frames))] = False
            ids, st = [], []
            for l in full:
                s.stats["lines"] += 1
                lid, targets, slots = [], [s.star], []
                for wi, w in enumerate(s.words(l[1])):
                    if wi: lid.append(SPACE)
                    for c in w:
                        lid.append(c)
                        if c in s.map: targets.append(s.map[c]); slots.append(len(lid) - 1)
                targets.append(s.star)
                lst = np.full(len(lid), np.nan)
                e0 = max(0, int((l[0] - s.pre * s.rate) * em_per_fr)); e1 = min(T, int((l[2] + s.post * s.rate) * em_per_fr) + 1)
                ok = bool(slots) and len(targets) < (e1 - e0) // 2
                if ok:
                    try:
                        ali, sc = AF.forced_align(em[b:b + 1, e0:e1], torch.tensor([targets], dtype=torch.int32, device=em.device), blank=0)
                        spans = AF.merge_tokens(ali[0], sc[0].exp()); ok = len(spans) == len(targets)
                    except Exception:
                        ok = False
                if ok:
                    for k, p in enumerate(slots): lst[p] = (e0 + spans[k + 1].start) / em_per_fr
                    s.stats["aligned_words"] += len(s.words(l[1]))
                else:
                    s.stats["fallback"] += 1; lst = l[0] + (l[2] - l[0]) * np.arange(len(lid)) / max(len(lid), 1)
                if ids: ids.append(SPACE); st.append(np.nan)            # line separator, timed by the next letter
                ids += lid; st += list(lst)
            st = np.asarray(st, np.float64)
            nxt = np.nan                                                # back-fill spaces / unmapped letters from the next letter
            for i in range(len(st) - 1, -1, -1):
                if np.isnan(st[i]): st[i] = nxt
                else: nxt = st[i]
            last = 0.0
            for i in range(len(st)):
                if np.isnan(st[i]): st[i] = last
                last = st[i]
            out.append((ids, np.maximum.accumulate(np.clip(st, 0, n_frames - 1e-3)) if len(st) else st, mask))
        return out
