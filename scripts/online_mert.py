#!/usr/bin/env python3
"""Online MERT features from YuE2-Vae latents (PLAN §8.6): latent [T,64] -> VAE decode -> 24 kHz mono -> MERT-v2-FullSong
layers -> 25 Hz -> per-track instance-normalised features on the GPU. Same recipe as tools/yue2_prep_real.py /
tools/yue2_extract_mert.py (30 s chunks, linear resample to 25 Hz), so a head trained on stored features sees the
same distribution up to the VAE round trip. Used by scripts/train_joint_online.py; run directly to measure the shift
between stored real-audio features and online features on data/yue2_real tracks."""
import sys, time
from pathlib import Path
import numpy as np, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
YUE2 = ROOT / "models/yue2"
LAYERS = [12, 16, 20, 23]


class OnlineMERT:
    def __init__(s, dev="cuda", layers=LAYERS, vae="YuE2-Vae"):
        from transformers import AutoModel, AutoFeatureExtractor
        from yue2.modeling_vae import YuE2VAE
        s.dev = dev; s.layers = layers
        s.proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
        s.mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()
        s.vae = YuE2VAE.from_pretrained(str(YUE2 / vae), decoder_only=True, device=dev, local_files_only=True)

    @torch.inference_mode()
    def mono24_from_latent(s, z):            # z float32/16 [T,64] -> float32 numpy 24 kHz mono
        zt = torch.as_tensor(np.asarray(z, dtype=np.float32)).T[None].contiguous().to(s.dev)
        audio = s.vae.decode_tiled(zt, core_frames=750, halo_frames=16, output_device=s.dev)
        mono = audio[0].float().clamp(-1, 1).mean(0).cpu().numpy()
        return resample_poly(mono, 1, 2).astype(np.float32)

    @torch.inference_mode()
    def feats_from_mono24(s, m24, T25=None):  # -> fp16 [K, T25, 1024] on GPU (raw, not normalised)
        CH = 24000 * 30; chunks = [m24[a:a + CH] for a in range(0, len(m24), CH)]; chunks = [c for c in chunks if len(c) >= 24000]
        full = [c for c in chunks if len(c) == CH]; tail = [c for c in chunks if len(c) < CH]; feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for group in ([full] if full else []) + [[c] for c in tail]:
                inp = {k: v.to(s.dev) for k, v in s.proc(group, sampling_rate=24000, return_tensors="pt").items()}
                hs = s.mert(**inp, output_hidden_states=True).hidden_states
                feats.append(torch.stack([hs[l].reshape(-1, 1024) for l in s.layers]))
        Hm = torch.cat(feats, 1).float(); T25 = int(round(len(m24) / 24000 * 25)) if T25 is None else T25
        return torch.nn.functional.interpolate(Hm.transpose(1, 2), size=T25, mode="linear", align_corners=False).transpose(1, 2).half()

    def feats_from_latent(s, z):
        return s.feats_from_mono24(s.mono24_from_latent(z), T25=len(z))


class MemTrack:
    """GPU-resident normalised features with the Track.window() interface used by the joint trainer."""
    def __init__(s, x, y=None, n=None):     # x fp16 [K,T,1024] (GPU)
        xf = x.float(); s.mu = xf.mean(1, keepdim=True); s.sd = xf.std(1, keepdim=True) + 1e-5; s.x = x
        s.n = x.shape[1] if n is None else min(n, x.shape[1]); s.y = None if y is None else y[:s.n]
    def window(s, a, win):                   # -> float32 [K, min(win, n-a), 1024] on GPU
        return (s.x[:, a:a + win].float() - s.mu) / s.sd
    def full(s): return s.window(0, s.n)


if __name__ == "__main__":                   # shift test: stored real features vs online-from-latent features
    dirs = sorted(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/yue2_real/shard-00001").iterdir())[:int(sys.argv[2]) if len(sys.argv) > 2 else 8]
    om = OnlineMERT(); t0 = time.time()
    for d in dirs:
        z = np.load(d / "latent_true.npy"); X = torch.tensor(np.load(d / "mert_L_12_16_20_23.npy"), device="cuda")
        t1 = time.time(); Y = om.feats_from_latent(z); dt = time.time() - t1
        n = min(X.shape[1], Y.shape[1]); a = MemTrack(X[:, :n]).full(); b = MemTrack(Y[:, :n]).full()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(1)          # per layer, over frames
        rel = ((a - b).norm(dim=-1) / a.norm(dim=-1)).mean(1)
        print(f"{d.name[:8]} T {n} online {dt:.1f}s | cos per layer {[round(float(c), 3) for c in cos]} | rel err {[round(float(r), 3) for r in rel]}", flush=True)
    print(f"done {time.time() - t0:.0f}s")
