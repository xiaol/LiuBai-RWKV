#!/usr/bin/env python3
"""Where does the token round trip lose the audio? On N gate songs, for each chunk length (8 s = training crop, 30 s = eval):
  1. latent stats: per-channel std of mu, and the tokenizer's per-channel reconstruction NMSE (rec vs mu)
  2. FSQ health: per-dimension level histogram of the pre-round values (saturation at the edges, unused levels)
  3. audio: mel-L1 dB of decode(mu), decode(rec), decode(mu + gaussian noise with the same per-channel error) vs the original
     -> separates "tokenizer loses information" from "VAE decoder is fragile to small latent errors"
  4. stereo: L/R correlation of each decode
usage: diag_tokenizer.py --vae out/vae/v1/G_s200000.pt --tok out/tok/b1_ctc1/last.pt [--n 6]
"""
import argparse, sys
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vae", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--root", default=str(ROOT / "out/yue2_roundtrip_gate20")); args = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a.get("mults", "1,2,4,8,16,16").split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
    tk = torch.load(args.tok, map_location="cpu"); T = SemTokenizer(levels=tuple(int(x) for x in tk["args"]["levels"].split(","))); T.load_state_dict(tk["T"]); T = T.to(dev).eval()
    mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80, power=1.0).to(dev)
    mdb = lambda y, x: ((torch.log(mel(y.mean(0)) + 1e-5) - torch.log(mel(x.mean(0)) + 1e-5)).abs().mean() * 20 / np.log(10)).item()
    lr = lambda y: float(torch.corrcoef(y)[0, 1])
    songs = sorted(p for p in Path(args.root).iterdir() if (p / "A_original.flac").exists())[:args.n]
    for chunk in (8.0, 30.0):
        W = int(chunk * 24000) // V.hop * V.hop; R = {k: [] for k in ("nmse", "mel_mu", "mel_rec", "mel_noise", "lr_orig", "lr_mu", "lr_rec", "lr_noise")}
        ch_std, ch_err, pre_all = [], [], []
        for d in songs:
            x, sr = sf.read(d / "A_original.flac", dtype="float32", always_2d=True); g = gcd(sr, 24000)
            x = resample_poly(x, 24000 // g, sr // g, axis=0).astype(np.float32)
            for s0 in range(0, len(x) - W, W * 2)[:4]:                                  # 4 chunks per song, spread out
                xc = torch.tensor(x[s0:s0 + W].T, device=dev)
                with torch.inference_mode():
                    mu, _ = V.encode(xc[None]); mu = mu.float()
                    h = T.inp(mu)
                    for b in T.enc: h = b(h)
                    pre = torch.tanh(T.to_q(h)); q, ids = T.fsq(T.to_q(h)); rec = T.decode(q).float()
                    err = (rec - mu); sd = mu[0].std(0); e = err[0].pow(2).mean(0).sqrt()
                    noisy = mu + torch.randn_like(mu) * e
                    ys = {k: torch.nn.functional.pad(V.decode(z)[0].float(), (0, W))[..., :W] for k, z in (("mu", mu), ("rec", rec), ("noise", noisy))}
                R["nmse"].append((err.pow(2).mean() / mu.var()).item()); ch_std.append(sd.cpu()); ch_err.append(e.cpu()); pre_all.append(pre[0].cpu())
                R["lr_orig"].append(lr(xc))
                for k, y in ys.items(): R[f"mel_{k}"].append(mdb(y, xc)); R[f"lr_{k}"].append(lr(y))
        sd = torch.stack(ch_std).mean(0); er = torch.stack(ch_err).mean(0); rel = (er / sd.clamp_min(1e-6)).numpy(); o = np.argsort(-sd.numpy())
        print(f"\n=== chunk {chunk:.0f} s, {len(R['nmse'])} chunks ===")
        for k, v in R.items(): print(f"  {k:10s} {np.mean(v):.3f}")
        print("  mu channel std (top 8 / bottom 8):", np.round(sd.numpy()[o[:8]], 3), np.round(sd.numpy()[o[-8:]], 4))
        print("  rel. error per channel (same order):", np.round(rel[o[:8]], 2), np.round(rel[o[-8:]], 2))
        print(f"  channels with std<0.05: {(sd < 0.05).sum().item()}  | rel err >0.5: {(rel > 0.5).sum()} of 64")
        P = torch.cat(pre_all)                                                             # [N, 5] in (-1, 1)
        for j, L in enumerate(T.fsq.L.long().tolist()):
            half = (L - 1) / 2; off = 0.5 if L % 2 == 0 else 0.0; lev = torch.round(P[:, j] * half - off) + off + half
            hist = torch.bincount(lev.long().clamp(0, L - 1), minlength=L).float(); hist /= hist.sum()
            print(f"  FSQ dim {j} (L={L}) |tanh|>0.98: {(P[:, j].abs() > 0.98).float().mean():.3f} levels used: {np.round(hist.numpy(), 3)}")


if __name__ == "__main__":
    main()
