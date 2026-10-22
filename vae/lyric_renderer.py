"""Phase C v2: lyric-conditioned RWKV renderer (our NAR), tokens + lyrics -> VAE latent by rectified flow.

Why: the token-only renderer (vae/renderer.py) sings fluent but invented words (gate20 CER 0.98) although the words are
readable from its input tokens (reader CER 0.39-0.50) - nothing made it render them. YuE2's NAR attends to the lyrics text;
ours gets them the RWKV way, in the sequence:

    sequence = frames with, at the frame where each LRC line starts, that line's romanised letters inserted just before it
    frame position  : x_t (noisy latent) + token embedding + time embedding   -> velocity head (only frames are scored)
    letter position : letter embedding + time embedding                       (no output)
    bidirectional RWKV-7 blocks carry each line in the state across the frames that sing it (state-native binding, no
    attention, unbounded length)

Training adds a lyric loss on the renderer's clean-latent estimate x1_hat = x_t + (1 - t) v_hat, read by the frozen lyric CTC
reader trained with the VAE (vae/model.LyricCTC, CER 0.49 on real latents): rendered latents must be readable as the words.
Null-lyrics / null-token dropout gives classifier-free guidance on each condition separately.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rwkv import BiBlock
from .renderer import t_embed
from .model import N_CTC

PAD_LETTER = 0                                         # letter ids from vae.data.encode_text are 1..N_CTC-1


def interleave(n_frames, lines, max_letters):
    """-> (pos_type [L] 0=frame 1=letter, frame_index [L] (-1 for letters), letter_id [L] (0 for frames))
    lines: [(start_frame, [letter ids])], letters truncated so the sequence stays <= n_frames + max_letters."""
    by_frame = {}
    budget = max_letters
    for l in sorted(lines, key=lambda l: l[0]):                                  # (f0, ids) or (f0, ids, f1); f0 may be < 0 (cut line)
        f0, ids = l[0], l[1]
        if budget <= 0: break
        ids = ids[:budget]; budget -= len(ids) + 1
        by_frame.setdefault(max(0, min(f0, n_frames - 1)), []).extend(ids + [1])  # id 1 = ' ' (LETTERS[0]): line separator
    pt, fi, li = [], [], []
    for f in range(n_frames):
        for c in by_frame.get(f, []): pt.append(1); fi.append(-1); li.append(c)
        pt.append(0); fi.append(f); li.append(0)
    return torch.tensor(pt), torch.tensor(fi), torch.tensor(li)


class LyricRenderer(nn.Module):
    def __init__(s, latent=64, d=1024, n=16, vocab=15360, n_letters=N_CTC):
        super().__init__(); s.d = d; s.latent = latent
        s.tok = nn.Embedding(vocab + 1, d); s.letter = nn.Embedding(n_letters + 1, d)   # last id = null
        s.null_tok, s.null_letter = vocab, n_letters
        s.kind = nn.Embedding(2, d)                                                    # frame / letter position
        s.xin = nn.Linear(latent, d); s.tmlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList(BiBlock(d) for _ in range(n)); s.norm = nn.LayerNorm(d); s.out = nn.Linear(d, latent)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)

    def build(s, codes, lines, max_letters=256, drop_tok=None, drop_lyr=None):
        """codes [B,T] long; lines: list (B) of [(start_frame, ids)] -> padded layout tensors for forward().
        Every sequence is padded to the SAME fixed length T + max_letters + 1 (null letters after the last frame): the backward
        RWKV pass reads the right padding, so a batch-dependent pad length would let one sample's lyrics change another's
        output (caught by vae/test_lyric_renderer.py) and make train / CFG layouts differ."""
        B, T = codes.shape; dev = codes.device; seqs = []
        for b in range(B):
            ln = [] if (drop_lyr is not None and drop_lyr[b]) else lines[b]
            seqs.append(interleave(T, ln, max_letters))
        L = T + max_letters + 1
        pt = torch.ones(B, L, dtype=torch.long); fi = torch.full((B, L), -1, dtype=torch.long); li = torch.zeros(B, L, dtype=torch.long)
        for b, (p, f, l) in enumerate(seqs):
            n = len(p); pt[b, :n] = p; fi[b, :n] = f; li[b, :n] = l
            pt[b, n:] = 1; li[b, n:] = s.null_letter                                   # right padding = null letters
        c = codes.clone()
        if drop_tok is not None: c[drop_tok] = s.null_tok
        return dict(pt=pt.to(dev), fi=fi.to(dev), li=li.to(dev), codes=c)

    def forward(s, x, t, lay):
        """x [B,T,64] noisy latent, t [B], lay from build() -> velocity [B,T,64]"""
        B, T, _ = x.shape; pt, fi, li = lay["pt"], lay["fi"], lay["li"]; L = pt.shape[1]
        isf = (pt == 0); fidx = fi.clamp_min(0)
        xf = torch.gather(x, 1, fidx[..., None].expand(B, L, x.shape[-1]))               # frame latent at frame positions
        cf = torch.gather(lay["codes"], 1, fidx)
        h = torch.where(isf[..., None], s.xin(xf) + s.tok(cf), s.letter(li)) + s.kind(pt) + s.tmlp(t_embed(t, 256))[:, None]
        for b in s.blocks: h = b(h)
        v_all = s.out(s.norm(h)).float()                                               # [B,L,64]
        # every frame index 0..T-1 appears exactly once among the frame positions: gather them back in frame order
        order = torch.where(isf, fi, T + torch.arange(L, device=x.device)[None].expand(B, L))    # letters sorted after frames
        idx = order.argsort(1)[:, :T]
        return torch.gather(v_all, 1, idx[..., None].expand(B, T, x.shape[-1]))

    @torch.no_grad()
    def sample(s, codes, lines, steps=24, cfg_tok=1.0, cfg_lyr=1.0, seed=0, max_letters=256):
        B, T = codes.shape; dev = codes.device; g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(B, T, s.latent, device=dev, generator=g)
        full = s.build(codes, lines, max_letters)
        no_lyr = s.build(codes, lines, max_letters, drop_lyr=torch.ones(B, dtype=torch.bool)) if cfg_lyr != 1.0 else None
        ts = torch.linspace(0, 1, steps + 1, device=dev)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]; tt = t0.expand(B); v = s(x, tt, full)
            if no_lyr is not None: vu = s(x, tt, no_lyr); v = vu + cfg_lyr * (v - vu)  # guidance towards the lyrics
            x = x + (t1 - t0) * v.float()
        return x
