"""Phase C: generative token -> latent renderer (replaces the MSE regressor that collapsed to the conditional mean).

Rectified flow on the VAE latent mu [B,T,64] @12.5 Hz, conditioned frame-by-frame on a token stream:
    x_t = (1 - t) * noise + t * mu,   target velocity v = mu - noise,   loss = |v_hat - v|^2
Network: per-frame condition embedding (token ids via nn.Embedding, or continuous features via Linear) + noisy latent + t
-> N bidirectional RWKV-7 blocks (vae/rwkv.py) -> velocity. Condition dropout (p 0.1) enables classifier-free guidance.
Sampling: Euler / midpoint ODE, 16-32 steps, optional CFG. Conditions at 25 Hz (MERT units) are average-pooled 2:1 to 12.5 Hz.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rwkv import BiBlock


def t_embed(t, dim):
    half = dim // 2; f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000 * f[None]; return torch.cat([a.sin(), a.cos()], -1)


class Renderer(nn.Module):
    def __init__(s, latent=64, d=768, n=12, vocab=0, cond_dim=0, cond_rate=1):
        super().__init__(); s.d = d; s.cond_rate = cond_rate
        s.emb = nn.Embedding(vocab + 1, d) if vocab else None                  # +1: null token for CFG
        s.cproj = nn.Linear(cond_dim, d) if cond_dim else None
        s.null = nn.Parameter(torch.zeros(d))
        s.xin = nn.Linear(latent, d); s.tmlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList(BiBlock(d) for _ in range(n)); s.norm = nn.LayerNorm(d); s.out = nn.Linear(d, latent)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)

    def cond(s, c, drop=None):                                                  # -> [B,T,d] at the latent rate
        h = s.emb(c) if s.emb is not None else s.cproj(c.float())
        if s.cond_rate > 1:                                                     # e.g. 25 Hz units -> 12.5 Hz
            B, T, D = h.shape; T2 = T // s.cond_rate; h = h[:, :T2 * s.cond_rate].view(B, T2, s.cond_rate, D).mean(2)
        if drop is not None: h = torch.where(drop[:, None, None], s.null.to(h.dtype).expand_as(h), h)
        return h

    def forward(s, x, t, ch):                                                   # x [B,T,64], t [B], ch [B,T,d]
        h = s.xin(x) + ch[:, :x.shape[1]] + s.tmlp(t_embed(t, 256))[:, None]
        for b in s.blocks: h = b(h)
        return s.out(s.norm(h))

    def loss(s, mu, c, p_drop=0.1):
        B = mu.shape[0]; t = torch.sigmoid(torch.randn(B, device=mu.device))                      # logit-normal t
        noise = torch.randn_like(mu); xt = (1 - t[:, None, None]) * noise + t[:, None, None] * mu
        drop = torch.rand(B, device=mu.device) < p_drop
        ch = s.cond(c, drop); T = min(ch.shape[1], mu.shape[1])
        v = s(xt[:, :T], t, ch[:, :T]); return F.mse_loss(v.float(), (mu - noise)[:, :T].float())

    @torch.no_grad()
    def sample(s, c, T=None, steps=24, cfg=1.0, seed=0):
        ch = s.cond(c); T = T or ch.shape[1]; ch = ch[:, :T]
        g = torch.Generator(device=ch.device).manual_seed(seed); x = torch.randn(ch.shape[0], T, s.xin.in_features, device=ch.device, generator=g)
        null = s.null.to(ch.dtype).expand_as(ch); ts = torch.linspace(0, 1, steps + 1, device=ch.device)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]; tt = t0.expand(x.shape[0])
            v = s(x, tt, ch)
            if cfg != 1.0: vu = s(x, tt, null); v = vu + cfg * (v - vu)
            x = x + (t1 - t0) * v.float()
        return x
