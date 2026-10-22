"""Phase B: lyric-aware semantic tokens on top of the LiuBai VAE latent (docs/VAE_PLAN.md).

  latent mu [B,T,64] @12.5 Hz (frozen VAE encoder)
    -> BiRWKV encoder (d 512, 6 blocks) -> FSQ (levels 8,8,8,6,5 = 15,360 codes)          tokens the RWKV LM generates
    -> heads trained only through the quantised code:
         * lyric CTC  (romanised letters, 50 Hz via 4x upsample, same LETTERS as the VAE probe)
         * latent reconstruction (BiRWKV decoder -> mu)  — the tokens must still describe the music, not only the words
FSQ (Mentzer et al. 2023) has no codebook collapse and no commitment loss; straight-through rounding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rwkv import BiBlock
from .model import N_CTC


class FSQ(nn.Module):
    def __init__(s, levels=(8, 8, 8, 6, 5)):
        super().__init__(); s.register_buffer("L", torch.tensor(levels, dtype=torch.float32))
        basis = torch.cumprod(torch.tensor([1] + list(levels[:-1])), 0); s.register_buffer("basis", basis)
        s.n_codes = int(torch.tensor(levels).prod())

    def forward(s, h):                                  # h [..., len(levels)] -> quantised (same shape, in [-1,1]), code ids
        half = (s.L - 1) / 2; off = (s.L % 2 == 0).float() * 0.5
        z = torch.tanh(h) * half - off                  # even levels: shifted grid so rounding gives exactly L values
        q = z + (torch.round(z) - z).detach()
        idx = ((torch.round(z) + off + half).long() * s.basis).sum(-1)
        return (q + off) / half, idx

    def decode(s, idx):                                 # code ids -> quantised vector
        half = (s.L - 1) / 2; off = (s.L % 2 == 0).float() * 0.5
        digits = (idx[..., None] // s.basis) % s.L.long()
        return (digits.float() - half) / half


class SemTokenizer(nn.Module):
    """n_q > 1: n_q independent FSQ codes per latent frame (product quantisation, 2026-10-22 capacity test): n_q x the bits and
    n_q x the LM tokens per second. ids are [B,T] for n_q = 1 (unchanged), [B,T,n_q] otherwise; q is always [B,T,n_q*len(levels)]."""
    def __init__(s, latent=64, d=512, n_enc=6, n_dec=4, levels=(8, 8, 8, 6, 5), ctc_up=4, n_q=1):
        super().__init__(); s.fsq = FSQ(levels); s.up = ctc_up; s.n_q = n_q; qd = n_q * len(levels)
        s.inp = nn.Linear(latent, d); s.enc = nn.ModuleList(BiBlock(d) for _ in range(n_enc)); s.to_q = nn.Linear(d, qd)
        s.from_q = nn.Linear(qd, d); s.dec = nn.ModuleList(BiBlock(d) for _ in range(n_dec)); s.out = nn.Linear(d, latent)
        s.ctc_in = nn.Linear(qd, 256 * ctc_up); s.ctc = nn.ModuleList(BiBlock(256) for _ in range(2)); s.ctc_out = nn.Linear(256, N_CTC)

    def encode(s, mu):
        h = s.inp(mu)
        for b in s.enc: h = b(h)
        h = s.to_q(h)
        if s.n_q == 1: return s.fsq(h)                  # q [B,T,5], ids [B,T]
        B, T, _ = h.shape; q, ids = s.fsq(h.view(B, T, s.n_q, -1))
        return q.reshape(B, T, -1), ids                 # q [B,T,n_q*L], ids [B,T,n_q]

    def decode(s, q):
        h = s.from_q(q)
        for b in s.dec: h = b(h)
        return s.out(h)

    def lyric_logp(s, q):
        B, T, _ = q.shape; h = s.ctc_in(q).view(B, T * s.up, -1)
        for b in s.ctc: h = b(h)
        return F.log_softmax(s.ctc_out(h).float(), -1)

    def forward(s, mu):
        q, ids = s.encode(mu)
        return s.decode(q), s.lyric_logp(q), ids
