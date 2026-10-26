"""Frozen semantic targets for the MSVAE-style partitioned latent of VAE v3 (vae/train.py --align-w / --chroma-w).

lyric stream      : MERT-v2-FullSong layer 12 (the most lyric-rich feature we measured: reader CER 0.399), online from the
                    crop (24 kHz mono, raw waveform input), instance-normalised over time per channel, resampled to 2 frames
                    per 12.5 Hz latent frame
composition stream: 12-bin chroma (Gaussian pitch-class filterbank on a 4096-point STFT, 55 Hz - 5 kHz), hop 960 = 2 frames
                    per latent frame, L2-normalised per frame
Both return [B, 2T, C] for a crop whose latent has T frames. MERT-v2 is CC BY-NC 4.0: first (non-commercial) version only.
"""
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class MertTarget(nn.Module):
    def __init__(s, layer=12, path=str(ROOT / "models/yue2/MERT-v2-FullSong")):
        super().__init__()
        from transformers import AutoModel
        s.mert = AutoModel.from_pretrained(path, trust_remote_code=True).eval().requires_grad_(False); s.layer = layer

    @torch.no_grad()
    def forward(s, wav, T):
        x = wav.float().mean(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = s.mert(input_values=x, output_hidden_states=True).hidden_states[s.layer].float()     # [B, ~25 Hz, 1024]
        h = F.interpolate(h.transpose(1, 2), size=2 * T, mode="linear", align_corners=False).transpose(1, 2)
        return (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-4)


class ChromaTarget(nn.Module):
    def __init__(s, sr=24000, n_fft=4096, hop=960, fmin=55.0, fmax=5000.0, width=0.5):
        super().__init__(); s.n_fft, s.hop = n_fft, hop
        f = torch.arange(n_fft // 2 + 1) * sr / n_fft
        midi = 12 * torch.log2(f.clamp_min(1e-3) / 440.0) + 69
        d = ((midi[None] - torch.arange(12)[:, None] + 6) % 12) - 6                                    # distance to pitch class
        fb = torch.exp(-0.5 * (d / width) ** 2) * ((f >= fmin) & (f <= fmax))[None]
        s.register_buffer("fb", fb / fb.sum(1, keepdim=True)); s.register_buffer("win", torch.hann_window(n_fft))

    @torch.no_grad()
    def forward(s, wav, T):
        x = wav.float().mean(1)
        with torch.autocast("cuda", enabled=False):
            X = torch.stft(x, s.n_fft, s.hop, window=s.win, return_complex=True).abs() ** 2        # [B, F, frames]
            c = torch.einsum("cf,bft->btc", s.fb, X).clamp_min(1e-10).sqrt()
        c = c[:, :2 * T]
        if c.shape[1] < 2 * T: c = F.pad(c, (0, 0, 0, 2 * T - c.shape[1]), mode="replicate")
        return F.normalize(c, dim=-1)


class AlignHead(nn.Module):
    """latent partition [B,T,k] -> [B,2T,C] prediction of a 2x-rate target"""
    def __init__(s, k, C, hid=512):
        super().__init__(); s.C = C; s.net = nn.Sequential(nn.Linear(k, hid), nn.GELU(), nn.Linear(hid, 2 * C))

    def forward(s, z):
        B, T, _ = z.shape; return s.net(z).view(B, 2 * T, s.C)


def cos_loss(pred, tgt):
    return (1 - F.cosine_similarity(pred.float(), tgt.float(), dim=-1)).mean()
