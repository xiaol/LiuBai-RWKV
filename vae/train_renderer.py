#!/usr/bin/env python3
"""Phase C trainer: flow-matching renderer token -> VAE latent (vae/renderer.py), on the fly from 8 s LRC-anchored crops.

--cond fsq      our tokens (vae/tokenizer.py checkpoint --tok), 12.5 Hz ids
--cond mertkm   k-means units of MERT-v2 layer --layer (fitted at start on --fit-batches batches, --k clusters, saved), 25 Hz ids
Eval on held-out shard-0 crops: sample latents from the tokens (24 Euler steps), decode with the VAE, mel-L1 dB and L/R
correlation vs the original crop, next to decode(mu) (the VAE ceiling). Full-song lyric WER comes from vae/eval_render_roundtrip.py.
usage: train_renderer.py --vae out/vae/v1/G_s200000.pt --cond fsq --tok out/tok/b1_ctc1/last.pt --out out/render/r_fsq [--steps 100000]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, torchaudio

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.renderer import Renderer  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.probe_lyrics import kmeans  # noqa: E402
YUE2 = ROOT / "models/yue2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cond", default="fsq", choices=["fsq", "mertkm"])
    ap.add_argument("--tok", default=""); ap.add_argument("--layer", type=int, default=12); ap.add_argument("--k", type=int, default=8192); ap.add_argument("--fit-batches", type=int, default=200)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"))
    ap.add_argument("--steps", type=int, default=100000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=768); ap.add_argument("--n", type=int, default=12); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5000)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1); torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval().requires_grad_(False)
    dl = iter(torch.utils.data.DataLoader(Crops(args.manifest, seed=11), batch_size=args.bs, num_workers=args.workers, collate_fn=collate, prefetch_factor=4, persistent_workers=True))

    if args.cond == "fsq":
        tk = torch.load(args.tok, map_location="cpu"); TK = SemTokenizer(levels=tuple(int(x) for x in tk["args"]["levels"].split(","))); TK.load_state_dict(tk["T"])
        TK = TK.to(dev).eval().requires_grad_(False); vocab, rate = TK.fsq.n_codes, 1
    else:
        from transformers import AutoModel, AutoFeatureExtractor
        proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
        mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()
        vocab, rate = args.k, 2
        @torch.no_grad()
        def mert_feat(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inp = proc([w.cpu().numpy() for w in x.mean(1).float()], sampling_rate=24000, return_tensors="pt")
                h = mert(**{k: v.to(dev) for k, v in inp.items()}, output_hidden_states=True).hidden_states[args.layer].float()
            T25 = int(round(x.shape[-1] / 24000 * 25)); h = F.interpolate(h.transpose(1, 2), size=T25, mode="linear", align_corners=False).transpose(1, 2)
            return (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-5)
        cp = out / "centroids.pt"
        if cp.exists(): C = torch.load(cp).to(dev)
        else:
            X = torch.cat([mert_feat(next(dl)[0].to(dev)).reshape(-1, 1024)[::2].half() for _ in range(args.fit_batches)]).float()
            C = kmeans(X, args.k); torch.save(C.cpu(), cp); del X; print(f"k-means L{args.layer}: {args.k} centroids", flush=True)

    @torch.no_grad()
    def encode(x):                                   # wav [B,2,S] on dev -> mu [B,T,64], cond ids
        with torch.autocast("cuda", dtype=torch.bfloat16): mu = V.encode(x)[0].float()
        if args.cond == "fsq": c = TK.encode(mu)[1]
        else: h = mert_feat(x); c = torch.cat([torch.cdist(hh, C[None]).argmin(-1) for hh in h.split(4)])
        return mu, c

    R = Renderer(d=args.d, n=args.n, vocab=vocab, cond_rate=rate).to(dev)
    opt = torch.optim.AdamW(R.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    ema = {k: v.detach().clone() for k, v in R.state_dict().items()}
    step = 0; ckp = out / "last.pt"
    if ckp.exists():
        sd = torch.load(ckp, map_location=dev); R.load_state_dict(sd["R"]); opt.load_state_dict(sd["opt"]); ema = sd["ema"]; step = sd["step"]; print("resumed", step, flush=True)
    print(f"renderer {sum(p.numel() for p in R.parameters()) / 1e6:.1f}M cond {args.cond} vocab {vocab}", flush=True)

    mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80, power=1.0).to(dev)
    mdb = lambda y, x: ((torch.log(mel(y.mean(1)) + 1e-5) - torch.log(mel(x.mean(1)) + 1e-5)).abs().mean() * 20 / math.log(10)).item()
    lrc = lambda y: float(np.mean([torch.corrcoef(b)[0, 1].item() for b in y]))
    vd = Crops(args.manifest, holdout=0, max_rows=1000, seed=12345); vit = iter(vd); val = [collate([next(vit) for _ in range(8)])[0] for _ in range(4)]
    @torch.no_grad()
    def evaluate():
        live = {k: v.clone() for k, v in R.state_dict().items()}; R.load_state_dict(ema); R.eval(); r = {"mel_render": 0, "mel_vae": 0, "lr_render": 0, "lr_orig": 0}
        for w in val:
            x = w.to(dev); mu, c = encode(x)
            with torch.autocast("cuda", dtype=torch.bfloat16): z = R.sample(c, T=mu.shape[1], steps=24)
            S = x.shape[-1]; dec = lambda q: F.pad(V.decode(q.float())[..., :S], (0, max(0, S - V.decode(q.float()).shape[-1])))
            yr, yv = dec(z), dec(mu)
            r["mel_render"] += mdb(yr, x) / len(val); r["mel_vae"] += mdb(yv, x) / len(val); r["lr_render"] += lrc(yr) / len(val); r["lr_orig"] += lrc(x) / len(val)
        R.load_state_dict(live); R.train(); return r

    log = open(out / "log.txt", "a"); t0 = time.time(); acc = 0
    while step < args.steps:
        step += 1; wav = next(dl)[0].to(dev, non_blocking=True)
        for g in opt.param_groups: g["lr"] = args.lr * min(1, step / 2000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        mu, c = encode(wav)
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = R.loss(mu, c)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for k, v in R.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        acc += loss.item()
        if step % 100 == 0:
            msg = f"step {step} flow {acc / 100:.4f} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"; print(msg, flush=True); log.write(msg + "\n"); log.flush(); acc = 0
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k} {v:.3f}" for k, v in e.items()); print(msg, flush=True); log.write(msg + "\n"); log.flush()
            torch.save(dict(R=R.state_dict(), opt=opt.state_dict(), ema=ema, step=step, args=vars(args)), out / "last.tmp"); os.replace(out / "last.tmp", ckp)
            if step % 25000 == 0: torch.save(dict(R=ema, step=step, args=vars(args)), out / f"R_ema_s{step}.pt")


if __name__ == "__main__":
    main()
