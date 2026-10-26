"""Our own singing ASR encoder (replaces MMS_FA / MERT / FireRedASR in the LiuBai stack; no YuE2 or NC weights).

Mixture audio (no source separation) -> 128-bin log-mel at 100 Hz -> conv stride 2 -> 50 Hz -> bidirectional RWKV-7 blocks
-> CTC over the romanised letters of vae.model.LETTERS (uroman: pinyin-like for CJK, same alphabet as every other lyric
reader here). Trained on the LRC-timed NetEase songs (vae/train_sing_asr.py). Three uses later:
  1. alignment teacher for VAE v3 (MSVAE-style ASR-aligned latent subspace): `features()` = last hidden states, 50 Hz
  2. forced aligner for the cursor labels (CTC posteriors)
  3. lyric reward for RL (an independent ASR - Whisper / Qwen3-ASR - stays the judge)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from .rwkv import BiBlock
from .model import N_CTC


class SingASR(nn.Module):
    def __init__(s, d=768, n=12, n_mels=128, sr=24000):
        super().__init__()
        s.mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, win_length=960, hop_length=sr // 100, n_mels=n_mels, power=2.0)
        s.inorm = nn.LayerNorm(n_mels)
        s.front = nn.Sequential(nn.Conv1d(n_mels, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 4, stride=2, padding=1), nn.GELU())
        s.blocks = nn.ModuleList(BiBlock(d) for _ in range(n)); s.norm = nn.LayerNorm(d); s.out = nn.Linear(d, N_CTC)

    def logmel(s, wav):
        """wav [B,2,S] or [B,S] at 24 kHz -> [B,T100,n_mels]"""
        x = wav.float().mean(1) if wav.dim() == 3 else wav.float()
        with torch.autocast("cuda", enabled=False):
            m = s.mel(x).clamp_min(1e-8).log()
        return m.transpose(1, 2)

    def encode(s, m):
        h = s.front(s.inorm(m).transpose(1, 2)).transpose(1, 2)           # 50 Hz
        for b in s.blocks: h = b(h)
        return s.norm(h)

    def forward(s, x, is_mel=False):
        """wav [B,2,S] (or log-mel [B,T100,n_mels] with is_mel) -> CTC log-probs [B, T50, N_CTC]"""
        return F.log_softmax(s.out(s.encode(x if is_mel else s.logmel(x))).float(), -1)

    def features(s, wav):
        return s.encode(s.logmel(wav))


def spec_augment(m, n_time=4, max_t=40, n_freq=2, max_f=20):
    """m [B,T,F] log-mel, in place on a copy: time / frequency masks filled with the per-sample mean"""
    m = m.clone(); B, T, Fq = m.shape; fill = m.mean((1, 2), keepdim=True)
    for _ in range(n_time):
        w = torch.randint(0, max_t + 1, (B,)); t0 = (torch.rand(B) * (T - w).clamp_min(1)).long()
        ar = torch.arange(T)[None]; mk = (ar >= t0[:, None]) & (ar < (t0 + w)[:, None])
        m = torch.where(mk[..., None].to(m.device), fill, m)
    for _ in range(n_freq):
        w = torch.randint(0, max_f + 1, (B,)); f0 = (torch.rand(B) * (Fq - w).clamp_min(1)).long()
        ar = torch.arange(Fq)[None]; mk = (ar >= f0[:, None]) & (ar < (f0 + w)[:, None])
        m = torch.where(mk[:, None].to(m.device), fill, m)
    return m
