#!/usr/bin/env python3
"""Bounded-lookahead v4 encoding (VAE.encode_local) vs the bidirectional VAE.encode on held-out 24 s crops.

  exact   : VAE._mert_layer (layers up to sem_layer only) == MERT2Model(...).hidden_states[sem_layer]
  leak    : first 12 s fixed, audio after 12 s replaced -> mean |diff| / std of the latent frames >= 1 s before the cut
            (encode leaks; encode_local must be 0 beyond its bound)
  quality : latent CTC CER (the VAE's own LyricCTC reader) and decoded mel distance to the original, causal vs full,
            plus the per-dim relative difference between the two latents
configs keep:win:ahead (blocks, s, s); leak is measured on the frames before the cut minus the config's lookahead bound
usage: check_causal.py --vae out/vae/v4/G_s60000.pt --cfgs 1:4:0.32,2:4:0.32,3:4:0
"""
import argparse, math, sys, time
from pathlib import Path
import torch, torchaudio

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import load_vae, LyricCTC  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vae", default=str(ROOT / "out/vae/v4/G_s60000.pt"))
    ap.add_argument("--cfgs", default="1:4:0.32,2:4:0.32,1:4:0.64,3:4:0,4:4:0.64"); ap.add_argument("--n", type=int, default=24); ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
    V = load_vae(args.vae, dev); ck = torch.load(args.vae, map_location="cpu")
    ctc = LyricCTC().to(dev).eval(); ctc.load_state_dict(ck["ctc"])
    it = iter(Crops(str(ROOT / "data/real_corpus/manifest/slim_val.jsonl"), holdout=0, seed=5, seg=24.0))
    crops = [next(it) for _ in range(args.n)]; ac = torch.autocast("cuda", dtype=torch.bfloat16)

    with torch.no_grad(), ac:
        x = collate(crops[:2])[0].to(dev).float().mean(1)
        a = V._mert[0](input_values=x, output_hidden_states=True).hidden_states[V.sem_layer].float(); b = V._mert_layer(x).float()
    print(f"exact: max |diff| {float((a - b).abs().max()):.2e} (scale {float(a.abs().mean()):.2f})", flush=True)

    def cer(mu, tg, lens, ok, cf):
        lp = ctc(mu.float()); offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]); e = n = 0
        for j in ok.nonzero().flatten().tolist():
            p = lp[j, :cf[j]].argmax(-1).tolist(); hyp = [c for k, c in enumerate(p) if c != 0 and (k == 0 or c != p[k - 1])]
            ref = tg[offs[j]:offs[j + 1]].tolist(); e += edit(hyp, ref); n += len(ref)
        return e, n

    mels = [torchaudio.transforms.MelSpectrogram(24000, n_fft=n, hop_length=n // 4, n_mels=m, power=1.0).to(dev) for n, m in ((512, 40), (1024, 80), (2048, 128))]

    def mel(mu, x):                         # vae/train.py mel_db: mean |log-mel difference| in dB over 3 resolutions
        y = V.decode(mu).float(); S = x.shape[-1]; y = torch.nn.functional.pad(y, (0, max(0, S - y.shape[-1])))[..., :S]
        y, x = y.reshape(-1, S).float(), x.reshape(-1, S).float()
        return float(sum((torch.log(m(y) + 1e-5) - torch.log(m(x) + 1e-5)).abs().mean() for m in mels) / 3) * 20 / math.log(10)

    res = {}
    cfgs = {c: dict(keep=int(c.split(":")[0]), win=float(c.split(":")[1]), ahead=float(c.split(":")[2])) for c in args.cfgs.split(",")}
    for name in ["full"] + list(cfgs):
        E = N = 0; md = []; rd = []; t = 0.0
        for i in range(0, args.n, 4):
            wav, tg, lens, ok, cf = collate(crops[i:i + 4]); x = wav.to(dev)
            with torch.no_grad(), ac:
                full = V.encode(x)[0].float(); torch.cuda.synchronize(); t0 = time.time()
                mu = full if name == "full" else V.encode_local(x, **cfgs[name])[0].float()
                torch.cuda.synchronize(); t += time.time() - t0; mu = mu[:, :full.shape[1]]; f = full[:, :mu.shape[1]]
                e, n = cer(mu, tg, lens, ok, cf); E += e; N += n; md.append(mel(mu, x))
                rd.append(float(((mu - f).abs() / f.reshape(-1, 64).std(0)).mean()))
        res[name] = dict(ctc_cer=E / max(N, 1), mel_db=sum(md) / len(md), rel_diff=sum(rd) / len(rd), sec_per_crop=t / args.n)
        print(name, {k: round(v, 4) for k, v in res[name].items()}, flush=True)

    wav = collate(crops[:4])[0].to(dev); other = collate(crops[4:8])[0].to(dev); S = 12 * 24000
    sp = torch.cat([wav[..., :S], other[..., S:]], -1)
    for name in ["full"] + list(cfgs):
        enc = (lambda z: V.encode(z)[0]) if name == "full" else (lambda z, c=cfgs[name]: V.encode_local(z, **c)[0])
        with torch.no_grad(), ac:
            a, b = enc(wav).float(), enc(sp).float()
        bound = 12 if name == "full" else cfgs[name]["keep"] * 4 + math.ceil(cfgs[name]["ahead"] * 12.5) + 8   # + 8 frames: acoustic path
        T = S // V.hop - bound; sd = a.reshape(-1, 64).std(0); d = ((a[:, :T] - b[:, :T]).abs() / sd)
        print(f"leak {name} (frames >= {bound} before the cut): sem {float(d[..., :V.sem].mean()):.4f} acoustic {float(d[..., V.sem:].mean()):.4f}", flush=True)

if __name__ == "__main__":
    main()
