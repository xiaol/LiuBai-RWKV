"""Gate-1 pilot of LiuBai-RWKV: block-flow RWKV LM on the continuous VAE latent with a monotonic lyric cursor.

No tokenizer and no separate renderer: a causal RWKV-7 stack steps once per block of `blk` latent frames (4 x 80 ms) and a
small flow head (MLP with adaLN) samples the next block from noise, conditioned on the stack's hidden state (block-wise
autoregression with a per-block flow, as DiffRhythm 2 / MAR; here the RNN state replaces the attention cache).

Lyrics, mode "cursor" (ours): the model never has to recall the lyrics. A cursor c_i points into the romanised letter string;
block i's input carries the letters [c_i - wb, c_i + wf) (absolute ids + relative-position embeddings), and an advance head
predicts a_i = how many letters block i sings (c_{i+1} = c_i + a_i). The flow head gets h_i + emb(a_i), i.e. exactly which
letters to sing now. Training labels come from letter-level forced alignment (vae/align_letters.py); at inference the cursor
is sampled from the advance head, so the model chooses its own timing.
Lyrics, mode "prefix" (baseline, DiffRhythm-2 style untimed conditioning): the whole letter string precedes the blocks and
must be recalled from the RWKV state.

Token i (predicts block i) = inp(block i-1) (BOS for i = 0) + lyric input for block i.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rwkv import CausalBlock
from .renderer import t_embed
from .model import N_CTC

PAD, END, NULL = 0, N_CTC, N_CTC + 1                     # letter ids 1..N_CTC-1 come from vae.data.encode_text


def cursor_arrays(starts, n_blocks, blk):
    """starts: letter start frames (non-decreasing) -> c [NB] letters sung before block i, a [NB] letters starting in block i"""
    c = np.searchsorted(np.asarray(starts, np.float64), np.arange(n_blocks + 1) * blk, side="left")
    return c[:-1], np.diff(c)


def windows(ids, c, wb, wf):
    """-> [NB, wb + wf] letter ids at string positions c_i - wb .. c_i + wf - 1 (PAD before the string, END after it)"""
    arr = np.concatenate([np.full(wb, PAD), np.asarray(ids, np.int64), np.full(1, END)])
    idx = np.minimum(np.asarray(c)[:, None] + np.arange(wb + wf)[None], len(arr) - 1)
    return arr[idx]


class AdaMLP(nn.Module):
    def __init__(s, d):
        super().__init__(); s.ln = nn.LayerNorm(d, elementwise_affine=False)
        s.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)); s.mod = nn.Linear(d, 3 * d)
        nn.init.zeros_(s.mod.weight); nn.init.zeros_(s.mod.bias)

    def forward(s, x, c):
        sh, sc, g = s.mod(F.silu(c)).chunk(3, -1)
        return x + g * s.mlp(s.ln(x) * (1 + sc) + sh)


class FlowHead(nn.Module):
    """velocity of one flattened block [.., dim] given flow time t [..] and condition c [.., cond]"""
    def __init__(s, dim, cond, d=1024, n=6):
        super().__init__()
        s.xin = nn.Linear(dim, d); s.cin = nn.Linear(cond, d); s.tm = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList(AdaMLP(d) for _ in range(n)); s.lnf = nn.LayerNorm(d, elementwise_affine=False)
        s.modf = nn.Linear(d, 2 * d); s.out = nn.Linear(d, dim)
        for m in (s.modf, s.out): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(s, x, t, c):
        cc = s.tm(t_embed(t.reshape(-1).float(), 256)).view(*t.shape, -1) + s.cin(c)
        h = s.xin(x)
        for b in s.blocks: h = b(h, cc)
        sh, sc = s.modf(F.silu(cc)).chunk(2, -1)
        return s.out(s.lnf(h) * (1 + sc) + sh)


class CursorLM(nn.Module):
    def __init__(s, latent=64, blk=4, d=1024, n=12, mode="cursor", wb=4, wf=20, max_adv=24, max_prefix=512, head_d=1024, head_n=6):
        super().__init__(); assert mode in ("cursor", "prefix")
        s.latent, s.blk, s.mode, s.wb, s.wf, s.max_adv, s.max_prefix = latent, blk, mode, wb, wf, max_adv, max_prefix
        D = s.D = latent * blk
        s.inp = nn.Linear(D, d); s.bos = nn.Parameter(torch.zeros(d)); s.kind = nn.Embedding(3, d)   # block / letter / pad
        if mode == "cursor":
            s.wl = nn.Embedding(N_CTC + 2, 64); s.wpos = nn.Parameter(torch.randn(wb + wf, 64) * 0.02); s.win = nn.Linear((wb + wf) * 64, d)
            s.adv = nn.Linear(d, max_adv); s.adv_emb = nn.Embedding(max_adv + 1, d)                    # id max_adv = null
        else:
            s.pl = nn.Embedding(N_CTC + 2, d)
        s.blocks = nn.ModuleList(CausalBlock(d) for _ in range(n)); s.norm = nn.LayerNorm(d)
        s.head = FlowHead(D, d, head_d, head_n)

    def hidden(s, z, lyr):
        """z [B,NB,D] blocks (only z[:, :-1] is read: token i sees blocks < i); lyr = windows [B,NB,W] (cursor) or prefix ids
        [B,P] (prefix, left-padded with PAD) -> h [B,NB,d] (h_i predicts block i)"""
        B, NB, _ = z.shape
        h = torch.cat([s.bos.expand(B, 1, -1), s.inp(z[:, :-1])], 1) + s.kind.weight[0]
        if s.mode == "cursor":
            h = h + s.win((s.wl(lyr) + s.wpos).flatten(2))
        else:
            h = torch.cat([s.pl(lyr) + s.kind(torch.where(lyr == PAD, 2, 1)), h], 1)
        for b in s.blocks: h = b(h)
        return s.norm(h)[:, -NB:]

    def cond(s, h, a=None):
        return h + s.adv_emb(a) if s.mode == "cursor" else h

    @torch.no_grad()
    def sample_block(s, c, steps=24, gen=None):
        """c [B,d] condition -> one block [B,D] (Euler, t: 0 noise -> 1 data)"""
        x = torch.randn(c.shape[0], s.D, device=c.device, generator=gen)
        ts = torch.linspace(0, 1, steps + 1, device=c.device)
        for i in range(steps):
            x = x + (ts[i + 1] - ts[i]) * s.head(x, ts[i].expand(c.shape[0]), c).float()
        return x
