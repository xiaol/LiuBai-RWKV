"""LiuBai VAE: Oobleck-style conv codec (Stable Audio / YuE2-Vae family) with bidirectional RWKV blocks at the latent rate,
plus the two RWKV-specific training signals of the pilot (docs/VAE_PLAN.md):

  * lyric CTC head   latent mean -> 4x upsample (50 Hz) -> small BiRWKV -> romanised-lyric letters (CTC). Gradients reach the
                     encoder, so the latent is pushed to carry what is sung, not only how it sounds.
  * latent prior     a small causal RWKV predicts z_{t+1} from z_<=t; its normalised error is added to the encoder loss, so
                     the latent stays predictable for an autoregressive LM (no frame-level noise the LM could never learn).

Pilot rate: 24 kHz stereo, hop 1920 -> 12.5 Hz, latent 64.  Strides 2*2*4*4*5*6 = 1920 (same as YuE2-Vae at 48 kHz).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint
from .rwkv import BiBlock, CausalBlock

LETTERS = " abcdefghijklmnopqrstuvwxyz'"          # CTC vocab: 0 = blank, 1.. = these
N_CTC = len(LETTERS) + 1


def WNConv1d(*a, **k): return weight_norm(nn.Conv1d(*a, **k))
def WNConvT1d(*a, **k): return weight_norm(nn.ConvTranspose1d(*a, **k))


class Snake(nn.Module):
    def __init__(s, c):
        super().__init__(); s.alpha = nn.Parameter(torch.zeros(1, c, 1)); s.beta = nn.Parameter(torch.zeros(1, c, 1))
    def forward(s, x):
        a, b = s.alpha.exp(), s.beta.exp()
        return x + torch.sin(x * a).pow(2) / (b + 1e-9)


class ResUnit(nn.Module):
    def __init__(s, c, dil):
        super().__init__(); s.net = nn.Sequential(Snake(c), WNConv1d(c, c, 7, dilation=dil, padding=3 * dil), Snake(c), WNConv1d(c, c, 1))
    def forward(s, x): return x + (checkpoint(s.net, x, use_reentrant=False) if s.training else s.net(x))


class EncBlock(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__(); s.net = nn.Sequential(ResUnit(cin, 1), ResUnit(cin, 3), ResUnit(cin, 9), Snake(cin),
                                                  WNConv1d(cin, cout, 2 * stride, stride=stride, padding=math.ceil(stride / 2)))
    def forward(s, x): return s.net(x)


class DecBlock(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__(); s.net = nn.Sequential(Snake(cin), WNConvT1d(cin, cout, 2 * stride, stride=stride, padding=math.ceil(stride / 2)),
                                                  ResUnit(cout, 1), ResUnit(cout, 3), ResUnit(cout, 9))
    def forward(s, x): return s.net(x)


class VAE(nn.Module):
    def __init__(s, ch=32, mults=(1, 2, 4, 8, 16, 16), strides=(2, 2, 4, 4, 5, 6), latent=64, d=512, n_rwkv=4, audio_ch=2):
        super().__init__(); s.hop = math.prod(strides); s.latent = latent
        m = [1] + list(mults)
        enc = [WNConv1d(audio_ch, ch, 7, padding=3)] + [EncBlock(m[i] * ch, m[i + 1] * ch, strides[i]) for i in range(len(strides))]
        s.enc_conv = nn.Sequential(*enc); s.enc_in = nn.Conv1d(m[-1] * ch, d, 1)
        s.enc_rwkv = nn.ModuleList(BiBlock(d) for _ in range(n_rwkv)); s.enc_out = nn.Linear(d, 2 * latent)
        s.dec_in = nn.Linear(latent, d); s.dec_rwkv = nn.ModuleList(BiBlock(d) for _ in range(n_rwkv)); s.dec_proj = nn.Conv1d(d, m[-1] * ch, 1)
        dec = [DecBlock(m[i + 1] * ch, m[i] * ch, strides[i]) for i in reversed(range(len(strides)))]
        s.dec_conv = nn.Sequential(*dec, Snake(ch), WNConv1d(ch, audio_ch, 7, padding=3, bias=False))

    def encode(s, wav):                     # wav [B,2,S] -> mean, logvar [B,T,latent]
        h = s.enc_in(s.enc_conv(wav)).transpose(1, 2)
        for b in s.enc_rwkv: h = b(h)
        mu, logvar = s.enc_out(h).chunk(2, -1)
        return mu, logvar.clamp(-12, 6)

    def decode(s, z):                       # z [B,T,latent] -> wav [B,2,T*hop]
        h = s.dec_in(z)
        for b in s.dec_rwkv: h = b(h)
        return s.dec_conv(s.dec_proj(h.transpose(1, 2)))

    def forward(s, wav, sample=True, s2=None, blur=None):
        """s2 [B] relative noise power, blur [B] side weight a of a [a, 1-2a, a] time kernel: generator-error augmentation
        (v2). Noise is scaled by each sample's own per-channel latent variance (detached), so the encoder cannot escape it by
        growing the latent; it must make the information robust instead."""
        mu, logvar = s.encode(wav)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        if s2 is not None:
            z = z + torch.randn_like(z) * (s2.view(-1, 1, 1) * mu.detach().var(dim=1, keepdim=True)).sqrt()
        if blur is not None:
            a = blur.view(-1, 1, 1).to(z.dtype); zp = F.pad(z.transpose(1, 2), (1, 1), mode="replicate").transpose(1, 2)
            z = a * zp[:, :-2] + (1 - 2 * a) * zp[:, 1:-1] + a * zp[:, 2:]
        y = s.decode(z); S = wav.shape[-1]                      # transposed-conv stack is a few samples short: pad / trim to the input
        y = F.pad(y, (0, S - y.shape[-1])) if y.shape[-1] < S else y[..., :S]
        return y, mu, logvar, z


class LyricCTC(nn.Module):
    """Deliberately small reader: if it can transcribe the lyrics, the information is explicit in the latent."""
    def __init__(s, latent=64, d=256, up=4, n=2):
        super().__init__(); s.up = up; s.inp = nn.Linear(latent, d * up); s.blocks = nn.ModuleList(BiBlock(d) for _ in range(n)); s.out = nn.Linear(d, N_CTC)
    def forward(s, z):                      # [B,T,latent] -> log-probs [B, T*up, N_CTC]
        B, T, _ = z.shape; h = s.inp(z).view(B, T * s.up, -1)
        for b in s.blocks: h = b(h)
        return F.log_softmax(s.out(h).float(), -1)


class LatentPrior(nn.Module):
    """Causal RWKV predicting the next latent frame; the error (normalised per dimension) is the predictability loss."""
    def __init__(s, latent=64, d=384, n=3):
        super().__init__(); s.inp = nn.Linear(latent, d); s.blocks = nn.ModuleList(CausalBlock(d) for _ in range(n)); s.out = nn.Linear(d, latent)
    def forward(s, z):                      # -> prediction of z[:, 1:]
        h = s.inp(z[:, :-1])
        for b in s.blocks: h = b(h)
        return s.out(h)


class STFTDisc(nn.Module):
    """EnCodec-style multi-scale complex-STFT discriminator (stereo: real/imag of both channels = 4 input planes)."""
    def __init__(s, n_ffts=(2048, 1024, 512), c=32):
        super().__init__(); s.n_ffts = n_ffts; s.nets = nn.ModuleList()
        for _ in n_ffts:
            s.nets.append(nn.ModuleList([
                weight_norm(nn.Conv2d(4, c, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(c, c, (3, 9), stride=(1, 2), dilation=(1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(c, c, (3, 9), stride=(1, 2), dilation=(2, 1), padding=(2, 4))),
                weight_norm(nn.Conv2d(c, c, (3, 9), stride=(1, 2), dilation=(4, 1), padding=(4, 4))),
                weight_norm(nn.Conv2d(c, c, (3, 3), padding=(1, 1))),
                weight_norm(nn.Conv2d(c, 1, (3, 3), padding=(1, 1)))]))

    def forward(s, wav):                    # wav [B,2,S] -> list of (logits, [features]) per scale
        out = []; B = wav.shape[0]
        for n, net in zip(s.n_ffts, s.nets):
            X = torch.stft(wav.float().reshape(B * 2, -1), n, hop_length=n // 4, window=torch.hann_window(n, device=wav.device), return_complex=True)
            Fq, Tn = X.shape[-2], X.shape[-1]
            X = torch.view_as_real(X).reshape(B, 2, Fq, Tn, 2).permute(0, 1, 4, 3, 2).reshape(B, 4, Tn, Fq)   # [B, 4 planes, time, freq]
            feats = []; h = X
            for i, conv in enumerate(net):
                h = conv(h)
                if i < len(net) - 1: h = F.leaky_relu(h, 0.2); feats.append(h)
            out.append((h, feats))
        return out
