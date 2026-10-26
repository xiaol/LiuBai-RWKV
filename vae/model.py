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
    def __init__(s, ch=32, mults=(1, 2, 4, 8, 16, 16), strides=(2, 2, 4, 4, 5, 6), latent=64, d=512, n_rwkv=4, audio_ch=2,
                 sem=0, sem_bound=0.5, sem_layer=12):
        super().__init__(); s.hop = math.prod(strides); s.latent = latent; s.sem, s.sem_bound, s.sem_layer = sem, sem_bound, sem_layer
        if sem:                             # v4: semantic-first posterior for dims [0, sem) (MSVAE-style), base = frozen MERT-v2
            s.sem_in = nn.Sequential(nn.Linear(2048, d), nn.GELU(), nn.Linear(d, d))      # 2 MERT frames (25 Hz) per latent frame
            s.sem_out = nn.Linear(d, 2 * sem); s.sem_gate = nn.Linear(d, sem)
            nn.init.zeros_(s.sem_gate.weight); nn.init.constant_(s.sem_gate.bias, -2.0)  # corrections start almost closed
        s._mert = [None]                    # frozen MERT model, attached at run time (attach_mert), never in the state dict
        m = [1] + list(mults)
        enc = [WNConv1d(audio_ch, ch, 7, padding=3)] + [EncBlock(m[i] * ch, m[i + 1] * ch, strides[i]) for i in range(len(strides))]
        s.enc_conv = nn.Sequential(*enc); s.enc_in = nn.Conv1d(m[-1] * ch, d, 1)
        s.enc_rwkv = nn.ModuleList(BiBlock(d) for _ in range(n_rwkv)); s.enc_out = nn.Linear(d, 2 * latent)
        s.dec_in = nn.Linear(latent, d); s.dec_rwkv = nn.ModuleList(BiBlock(d) for _ in range(n_rwkv)); s.dec_proj = nn.Conv1d(d, m[-1] * ch, 1)
        dec = [DecBlock(m[i + 1] * ch, m[i] * ch, strides[i]) for i in reversed(range(len(strides)))]
        s.dec_conv = nn.Sequential(*dec, Snake(ch), WNConv1d(ch, audio_ch, 7, padding=3, bias=False))

    def attach_mert(s, mert): s._mert[0] = mert; return s

    def _mert_layer(s, x):
        """MERT2Model(x, output_hidden_states=True).hidden_states[sem_layer], running only the layers up to it (identical, half the cost)"""
        M = s._mert[0]; mel = M.feature_extractor(x)
        h = M.subsampling_module(mel.to(dtype=M.subsampling_module[0].convnext_layers[0].depthwise_block[1].weight.dtype))
        pos = M.embed_positions(h)
        for layer in M.layers[:s.sem_layer + 1]: h = layer(h, pos)
        return h

    @torch.no_grad()
    def mert_feats_local(s, wav, T, blk=4, keep=2, win=4.0, ahead=0.32, bs=96):
        """Bounded-lookahead MERT features [B,T,2048]: the frames of blocks [j, j+keep) come from MERT run on the audio
        [end of block j+keep - win, end of block j+keep + ahead] only (instance-normalised over that window), so no latent
        frame sees more than (keep - 1) blocks + `ahead` s past its own block. mert_feats() attends over the whole crop."""
        B, C, S = wav.shape; L = blk * s.hop; nB = -(-T // blk); W = max(keep * L, int(round(win * 24000)) // L * L)
        A = int(round(ahead * 24000)) // L * L; x = F.pad(wav.float().mean(1), (W, A + L * keep))          # left pad: zeros
        st = [(j + keep) * L for j in range(0, nB, keep)]           # window [end - W, end + A] of chunk j, in padded coordinates
        segs = torch.stack([x[:, a:a + W + A] for a in st], 1).reshape(-1, W + A)                         # [B*nw, W+A]
        hs = []
        for i in range(0, segs.shape[0], bs):
            with torch.autocast("cuda", dtype=torch.bfloat16): h = s._mert_layer(segs[i:i + bs]).float()
            hs.append((h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-4))
        h = torch.cat(hs)                                                                                    # [B*nw, ~25 Hz, 1024]
        f = 2 * (W + A) // s.hop; h = F.interpolate(h.transpose(1, 2), size=f, mode="linear", align_corners=False).transpose(1, 2)
        c0 = 2 * (W - keep * L) // s.hop; h = h[:, c0:c0 + 2 * keep * blk]                                  # the window's own blocks
        h = h.reshape(B, -1, 2 * keep * blk, 1024).flatten(1, 2)[:, :2 * T]
        return h.reshape(B, T, 2048)

    def encode_local(s, wav, **kw):
        """encode() with the MERT branch replaced by mert_feats_local (bounded lookahead); the acoustic path is unchanged
        (its influence decays to ~0 within ~0.6 s, measured). Use the same kw for training latents and generation prompts."""
        h = s.enc_in(s.enc_conv(wav)).transpose(1, 2)
        for b in s.enc_rwkv: h = b(h)
        mu, logvar = s.enc_out(h).chunk(2, -1)
        if s.sem:
            k = s.sem; hs = s.sem_in(s.mert_feats_local(wav, mu.shape[1], **kw).to(h.dtype))
            mu_s, lv_s = s.sem_out(hs).chunk(2, -1)
            corr = s.sem_bound * torch.tanh(mu[..., :k]) * torch.sigmoid(s.sem_gate(hs))
            mu = torch.cat([mu_s + corr, mu[..., k:]], -1); logvar = torch.cat([lv_s, logvar[..., k:]], -1)
        return mu, logvar.clamp(-12, 6)

    @torch.no_grad()
    def mert_feats(s, wav, T, chunk=30 * 24000):
        """wav [B,2,S] 24 kHz -> [B,T,2048]: MERT layer sem_layer, instance-normalised per <= 30 s chunk, 2 frames per latent frame"""
        M = s._mert[0]; assert M is not None, "sem VAE needs attach_mert() (vae.model.load_vae does it)"
        x = wav.float().mean(1); out = []
        for a in range(0, x.shape[1], chunk):
            c = x[:, a:a + chunk]
            if c.shape[1] < 2400: c = F.pad(c, (0, 2400 - c.shape[1]))
            with torch.autocast("cuda", dtype=torch.bfloat16): h = s._mert_layer(c).float()
            out.append((h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-4))
        h = torch.cat(out, 1)
        h = F.interpolate(h.transpose(1, 2), size=2 * T, mode="linear", align_corners=False).transpose(1, 2)
        return h.reshape(h.shape[0], T, 2048)

    def encode(s, wav):                     # wav [B,2,S] -> mean, logvar [B,T,latent]
        h = s.enc_in(s.enc_conv(wav)).transpose(1, 2)
        for b in s.enc_rwkv: h = b(h)
        mu, logvar = s.enc_out(h).chunk(2, -1)
        if s.sem:                           # dims [0, sem): MERT base + gated, bounded acoustic correction; rest: acoustic
            k = s.sem; hs = s.sem_in(s.mert_feats(wav, mu.shape[1]).to(h.dtype))
            mu_s, lv_s = s.sem_out(hs).chunk(2, -1)
            corr = s.sem_bound * torch.tanh(mu[..., :k]) * torch.sigmoid(s.sem_gate(hs))
            mu = torch.cat([mu_s + corr, mu[..., k:]], -1); logvar = torch.cat([lv_s, logvar[..., k:]], -1)
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


def load_vae(path, dev="cuda", mert=None):
    """VAE from a vae/train.py checkpoint (any version); sem VAEs get MERT-v2 attached (pass `mert` to share one model)"""
    ck = torch.load(path, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a.get("mults", "1,2,4,8,16,16").split(",")), d=a["d"], n_rwkv=a["n_rwkv"],
            sem=a.get("sem", 0), sem_bound=a.get("sem_bound", 0.5), sem_layer=a.get("align_layer", 12))
    V.load_state_dict(ck["G"]); V = V.to(dev).eval().requires_grad_(False)
    if V.sem:
        if mert is None:
            from .align_targets import MertTarget
            mert = MertTarget(V.sem_layer).mert.to(dev)
        V.attach_mert(mert)
    return V
