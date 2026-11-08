"""MSVAE (ASLP-lab, arXiv 2610.17187, Apache-2.0 release at /models/msvae) behind the small VAE interface the LM tools use:
sr / hop / rate of the encoder side, out_sr of the decoder, encode(wav) -> (mu,), decode(z) -> wav.

encode : 24 kHz stereo [B,2,S] -> posterior mean [B,T,64] at 25 Hz (hop 960). Same windows as the release's encode_audio
         (128 frames, 32 overlap, half the overlap trimmed at interior edges, last window flush with the end), but all
         windows of a song are batched (--bs windows per forward). Audio shorter than one window is zero-padded to it.
decode : [B,T,64] -> 48 kHz stereo [B,2,T*1920], release decode_audio windows, batched the same way.
Runs in venvs/yue2 with PYTHONPATH=venvs/msvae_shim (MuQ + its pure-python deps symlinked from /root/envs/msvae-venv, a
torchvision stub: x_clip imports it, MuQ never calls it). load_msvae() adds the shim itself.
"""
import math, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CODE, CKPT, MUQ = "/models/msvae/code", "/models/msvae/weights/model.pt", "/models/msvae/muq"


def _windows(T, chunk, overlap):
    if T <= chunk: return [0]
    st = list(range(0, T - chunk + 1, chunk - overlap))
    if st[-1] != T - chunk: st.append(T - chunk)
    return st


class MSVAE:
    sr, out_sr, hop, rate, latent = 24000, 48000, 960, 25.0, 64

    def __init__(s, dev="cuda", chunk=128, overlap=32, bs=16, amp=False):
        shim = str(ROOT / "venvs/msvae_shim")
        if shim not in sys.path: sys.path.insert(0, shim)
        if CODE not in sys.path: sys.path.insert(0, CODE)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import yaml
        from src.model import DAC
        from src.infer.infer import load_checkpoint
        cfg = yaml.safe_load(open(Path(CODE) / "src/config/vae/config.yaml"))
        kw = {k[4:]: v for k, v in cfg.items() if k.startswith("DAC.")}
        kw["bigvgan_conf"] = str(Path(CODE) / kw["bigvgan_conf"]); kw["muq_model_name"] = MUQ
        s.M = load_checkpoint(DAC(**kw).float(), CKPT, dev).eval()
        for p in s.M.parameters(): p.requires_grad_(False)
        s.dev, s.chunk, s.overlap, s.bs, s.amp = dev, chunk, overlap, bs, amp
        s.up = s.out_sr // s.sr * s.hop                                       # 1920 output samples per frame

    def _run(s, f, xs):
        out = []
        for i in range(0, len(xs), s.bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.amp): out.append(f(torch.stack(xs[i:i + s.bs])).float())
        return torch.cat(out)

    @torch.no_grad()
    def encode(s, wav):
        """wav [B,2,S] (24 kHz) -> (mu [B,T,64],), T = ceil(S / 960)"""
        B, _, S = wav.shape; T = math.ceil(S / s.hop); Tp = max(T, s.chunk); c, o = s.chunk, s.overlap
        x = F.pad(wav.float(), (0, Tp * s.hop - S)); st = _windows(Tp, c, o); out = []
        for b in range(B):
            z = s._run(lambda a: s.M.encode_latents(a, "mean"), [x[b, :, a * s.hop:(a + c) * s.hop] for a in st])   # [n,c,64]
            mu = torch.zeros(Tp, s.latent, device=x.device)
            for i, a in enumerate(st):
                lo = 0 if i == 0 else o // 2; hi = c if i == len(st) - 1 else c - o // 2
                mu[a + lo:a + hi] = z[i, lo:hi]
            out.append(mu[:T])
        return (torch.stack(out),)

    @torch.no_grad()
    def decode(s, z):
        """z [B,T,64] -> [B,2,T*1920] (48 kHz)"""
        B, T, _ = z.shape; Tp = max(T, s.chunk); c, o = s.chunk, s.overlap
        z = F.pad(z.float(), (0, 0, 0, Tp - T)); st = _windows(Tp, c, o); out = []
        for b in range(B):
            y = s._run(s.M.decode, [z[b, a:a + c] for a in st])                    # [n,2,c*1920]
            w = torch.zeros(2, Tp * s.up, device=z.device)
            for i, a in enumerate(st):
                lo = (0 if i == 0 else o // 2) * s.up; hi = (c if i == len(st) - 1 else c - o // 2) * s.up
                w[:, a * s.up + lo:a * s.up + hi] = y[i, :, lo:hi]
            out.append(w[:, :T * s.up])
        return torch.stack(out)
