#!/usr/bin/env python3
"""Full-song round trip for tokenizer v2 (vae/train_tokenizer2.py: hybrid VAE-mu + MERT-L12 FSQ codes + joint flow renderer).
original -> 24 kHz -> (VAE mu, MERT L12) -> HybridTok codes -> Renderer.sample(code vectors) -> VAE decode -> C_roundtrip_<tag>.flac
30 s windows, 2 s audio cross-fade; scored by tools/yue2_roundtrip_eval.py. Also saves tokens_<tag>.npy (code ids, 12.5 Hz).
usage: eval_tok2_roundtrip.py --vae out/vae/v1/G_s200000.pt --ckpt out/tok2/h1/last.pt --tag tok2h1_30k [--cfg 1.0]
"""
import argparse, sys
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.renderer import Renderer  # noqa: E402
from vae.train_tokenizer2 import HybridTok  # noqa: E402
YUE2 = ROOT / "models/yue2"


def rs(x, a, b):
    g = gcd(a, b); return resample_poly(x, b // g, a // g, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vae", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(ROOT / "out/yue2_roundtrip_gate20")); ap.add_argument("--cfg", type=float, default=1.0); ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--chunk", type=float, default=30.0); ap.add_argument("--overlap", type=float, default=2.0); args = ap.parse_args(); dev = "cuda"
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
    sd = torch.load(args.ckpt, map_location="cpu"); ta = sd["args"]; levels = tuple(int(x) for x in ta["levels"].split(","))
    TK = HybridTok(levels); TK.load_state_dict(sd["TK"]); TK = TK.to(dev).eval()
    R = Renderer(d=ta["rd"], n=ta["rn"], cond_dim=len(levels)); R.load_state_dict(sd["R"]); R = R.to(dev).eval()
    from transformers import AutoModel, AutoFeatureExtractor
    proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
    mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()

    @torch.no_grad()
    def render(seg):
        x = torch.tensor(seg.T[None], device=dev); S = x.shape[-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = V.encode(x)[0].float()
            inp = proc([x.mean(1)[0].float().cpu().numpy()], sampling_rate=24000, return_tensors="pt")
            h = mert(**{k: v.to(dev) for k, v in inp.items()}, output_hidden_states=True).hidden_states[ta["layer"]].float()
            h = F.interpolate(h.transpose(1, 2), size=2 * mu.shape[1], mode="linear", align_corners=False).transpose(1, 2)
            h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-5)
            q, ids = TK.encode(mu, h); z = R.sample(q.float(), T=mu.shape[1], steps=args.steps, cfg=args.cfg)
            y = V.decode(z.float())[0].float()
        return F.pad(y, (0, max(0, S - y.shape[-1])))[..., :S].cpu().numpy().T, ids[0].cpu().numpy()

    for d in sorted(p for p in Path(args.root).iterdir() if (p / "A_original.flac").exists()):
        x, sr = sf.read(d / "A_original.flac", dtype="float32", always_2d=True); x24 = rs(x, sr, 24000); x24 = x24[:len(x24) - len(x24) % V.hop]
        W = int(args.chunk * 24000) // V.hop * V.hop; O = int(args.overlap * 24000) // V.hop * V.hop; H = W - O
        out = np.zeros_like(x24); wsum = np.zeros((len(x24), 1), np.float32); toks = np.zeros(len(x24) // V.hop, np.int32)
        fade = np.ones(W, np.float32); fade[:O] = np.linspace(0, 1, O); fade[W - O:] = np.linspace(1, 0, O)
        for s0 in range(0, max(1, len(x24) - O), H):
            seg = x24[s0:s0 + W]; n = len(seg)
            if n < W: seg = np.pad(seg, ((0, W - n), (0, 0)))
            y, ids = render(seg); f = fade[:n].copy()
            if s0 == 0: f[:O] = 1
            if s0 + W >= len(x24): f[max(0, n - O):] = 1
            out[s0:s0 + n] += y[:n] * f[:, None]; wsum[s0:s0 + n] += f[:, None]
            k0 = s0 // V.hop; k1 = min(len(toks), k0 + n // V.hop); toks[k0:k1] = ids[:k1 - k0]
        sf.write(d / f"C_roundtrip_{args.tag}.flac", np.clip(rs(out / np.maximum(wsum, 1e-6), 24000, 48000), -1, 1), 48000, subtype="PCM_16")
        np.save(d / f"tokens_{args.tag}.npy", toks); print(d.name, f"{len(x24) / 24000:.0f}s unique {len(np.unique(toks))}", flush=True)


if __name__ == "__main__":
    main()
