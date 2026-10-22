#!/usr/bin/env python3
"""Train the lyric-conditioned RWKV renderer (vae/lyric_renderer.py) on 8 s LRC-anchored crops.

condition: our FSQ codes (vae/tokenizer.py checkpoint --tok, 12.5 Hz) + the crop's timed lyric lines (romanised letters)
loss     : rectified flow on the VAE latent mu
           + lyr_w * CTC of the FROZEN VAE lyric reader (the LyricCTC trained with --vae) on the clean-latent estimate
             x1_hat = x_t + (1 - t) v_hat, only for t >= --lyr-tmin (at small t the estimate is mostly noise)
dropout  : codes -> null 10 %, lyrics -> null 15 % (independent) for two-way CFG
eval     : held-out crops, 24-step samples with / without lyrics: reader CER on the sampled latent (= do the words come out?),
           mel dB of the decoded audio vs the original, and the same CER on mu (ceiling)
usage: train_lyric_renderer.py --vae out/vae/v1/G_s200000.pt --tok out/tok/b1_ctc1/last.pt --out out/render/lyr1 [--steps 150000]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, torchaudio

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE, LyricCTC  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.lyric_renderer import LyricRenderer  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402
from vae.mms_judge import MMSJudge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"))
    ap.add_argument("--steps", type=int, default=150000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--d", type=int, default=1024); ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--judge", default="reader", choices=["reader", "mms"], help="mms = audio-domain judge: decode x1_hat through the "
                    "frozen VAE and score the audio with MMS_FA CTC (the latent reader is fooled: it reads renders at 0.49 while Whisper hears 0.98)")
    ap.add_argument("--judge-n", type=int, default=3, help="mms: samples per step that are decoded and judged (memory)")
    ap.add_argument("--align", default="line", choices=["line", "word"], help="word: MMS forced alignment of complete lines in the "
                    "crop, every word enters at its sung frame (vae/align_words.py); line: whole line at its LRC start")
    ap.add_argument("--no-norm", action="store_true", help="train the flow on the raw latent (v1 runs). Default: per-channel "
                    "normalised latent (v2's latent has channel std ~7.7; flow from unit noise needs ~unit-scale data)")
    ap.add_argument("--init", default="", help="initialise the renderer from this checkpoint (R / ema weights), fresh optimiser")
    ap.add_argument("--lyr-w", type=float, default=0.1); ap.add_argument("--lyr-tmin", type=float, default=0.5); ap.add_argument("--lyr-start", type=int, default=5000)
    ap.add_argument("--p-tok", type=float, default=0.1); ap.add_argument("--p-lyr", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--eval-every", type=int, default=5000)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1); torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval().requires_grad_(False)
    reader = LyricCTC(); reader.load_state_dict(ck["ctc"]); reader = reader.to(dev).eval().requires_grad_(False)
    J = MMSJudge().to(dev) if args.judge == "mms" else None
    from vae.align_words import WordAligner
    AL = WordAligner(dev) if args.align == "word" else None
    if J is not None: V.train()                         # ResUnit gradient checkpointing (train mode only); no dropout/BN in the VAE
    tk = torch.load(args.tok, map_location="cpu"); TK = SemTokenizer(levels=tuple(int(x) for x in tk["args"]["levels"].split(","))); TK.load_state_dict(tk["T"])
    TK = TK.to(dev).eval().requires_grad_(False)
    R = LyricRenderer(d=args.d, n=args.n, vocab=TK.fsq.n_codes).to(dev)
    opt = torch.optim.AdamW(R.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    ema = {k: v.detach().clone() for k, v in R.state_dict().items()}
    step = 0; ckp = out / "last.pt"; lat_m = torch.zeros(64, device=dev); lat_s = torch.ones(64, device=dev); have_stats = False
    if args.init and not ckp.exists():
        ik = torch.load(args.init, map_location=dev); R.load_state_dict(ik.get("ema", ik["R"])); ema = {k: v.detach().clone() for k, v in R.state_dict().items()}
        if "lat_m" in ik: lat_m, lat_s = ik["lat_m"].to(dev), ik["lat_s"].to(dev); have_stats = True   # same flow space as the init
        print("init from", args.init, "(with its latent stats)" if have_stats else "", flush=True)
    if ckp.exists():
        sd = torch.load(ckp, map_location=dev); R.load_state_dict(sd["R"]); opt.load_state_dict(sd["opt"]); ema = sd["ema"]; step = sd["step"]; print("resumed", step, flush=True)
        if "lat_m" in sd: lat_m, lat_s = sd["lat_m"].to(dev), sd["lat_s"].to(dev)
    elif not args.no_norm and not have_stats:                              # per-channel latent stats from 128 training crops (fixed for the whole run)
        sit = iter(Crops(args.manifest, seed=999)); fs = []
        with torch.no_grad():
            for _ in range(8):
                xb = collate([next(sit) for _ in range(16)])[0].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): fs.append(V.encode(xb)[0].float().reshape(-1, 64))
        f = torch.cat(fs); lat_m, lat_s = f.mean(0), f.std(0).clamp_min(1e-3)
        print(f"latent norm: channel std min/median/max {lat_s.min():.3f} {lat_s.median():.3f} {lat_s.max():.3f}", flush=True)
    nrm = lambda z: (z - lat_m) / lat_s; den = lambda z: z * lat_s + lat_m
    print(f"lyric renderer {sum(p.numel() for p in R.parameters()) / 1e6:.1f}M, codes {TK.fsq.n_codes}", flush=True)

    @torch.no_grad()
    def encode(x):
        with torch.autocast("cuda", dtype=torch.bfloat16): mu = V.encode(x)[0].float()
        return mu, TK.encode(mu)[1]

    def ctc(lp, tg, lens, ok, cf):
        idx = ok.nonzero().flatten()
        if len(idx) == 0: return None
        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]); tgt = torch.cat([tg[offs[i]:offs[i + 1]] for i in idx.tolist()]).to(dev)
        return F.ctc_loss(lp[idx].transpose(0, 1), tgt, torch.clamp(cf[idx], max=lp.shape[1]).to(dev), lens[idx].to(dev), blank=0, zero_infinity=True)

    def cer(lp, tg, lens, ok, cf):
        e = c = 0; offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
        for j in ok.nonzero().flatten().tolist():
            p = lp[j, :cf[j]].argmax(-1).tolist(); hyp = [t for i, t in enumerate(p) if t != 0 and (i == 0 or t != p[i - 1])]
            ref = tg[offs[j]:offs[j + 1]].tolist(); e += edit(hyp, ref); c += len(ref)
        return e, c

    mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80, power=1.0).to(dev)
    mdb = lambda y, x: ((torch.log(mel(y.mean(1)) + 1e-5) - torch.log(mel(x.mean(1)) + 1e-5)).abs().mean() * 20 / math.log(10)).item()
    vd = Crops(args.manifest, holdout=0, max_rows=1000, seed=12345, with_lines=True); vit = iter(vd)
    val = [collate([next(vit) for _ in range(16)]) for _ in range(3)]
    @torch.no_grad()
    def evaluate():
        live = {k: v.clone() for k, v in R.state_dict().items()}; R.load_state_dict(ema); R.eval()
        acc = {k: [0, 0] for k in ("cer_mu", "cer_lyr", "cer_nolyr")}; mels = []
        for wav, tg, lens, ok, cf, lines in val:
            x = wav.to(dev); mu, c = encode(x); S = x.shape[-1]
            if AL is not None: lines = AL(x, lines, mu.shape[1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_l = den(R.sample(c, lines, steps=24).float()); z_n = den(R.sample(c, [[] for _ in lines], steps=24).float())
                for k, z in (("cer_mu", mu), ("cer_lyr", z_l), ("cer_nolyr", z_n)):
                    e, n = cer(reader(z.float()), tg, lens, ok, cf); acc[k][0] += e; acc[k][1] += n
                y = V.decode(z_l.float()).float(); y = F.pad(y, (0, max(0, S - y.shape[-1])))[..., :S]; mels.append(mdb(y, x))
        R.load_state_dict(live); R.train()
        r = {k: e / max(n, 1) for k, (e, n) in acc.items()}; r["mel_render"] = float(np.mean(mels)); return r

    dl = iter(torch.utils.data.DataLoader(Crops(args.manifest, seed=31 + step, with_lines=True), batch_size=args.bs, num_workers=args.workers,
                                          collate_fn=collate, prefetch_factor=4, persistent_workers=True))
    log = open(out / "log.txt", "a"); t0 = time.time(); accl = {}
    while step < args.steps:
        step += 1; wav, tg, lens, ok, cf, lines = next(dl); x = wav.to(dev, non_blocking=True)
        for g in opt.param_groups: g["lr"] = args.lr * min(1, step / 3000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        mu, c = encode(x); B = mu.shape[0]
        if AL is not None: lines = AL(x, lines, mu.shape[1])                        # word-level timing from the crop's own audio
        mu = nrm(mu)                                                                # flow space = normalised latent
        t = torch.sigmoid(torch.randn(B, device=dev)); noise = torch.randn_like(mu); xt = (1 - t[:, None, None]) * noise + t[:, None, None] * mu
        lay = R.build(c, lines, drop_tok=torch.rand(B, device=dev) < args.p_tok, drop_lyr=(torch.rand(B) < args.p_lyr).tolist())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = R(xt, t, lay).float(); lf = F.mse_loss(v, mu - noise); loss = lf; ll = torch.zeros((), device=dev)
            if args.lyr_w > 0 and step > args.lyr_start:
                sel = (t >= args.lyr_tmin) & ok.to(dev)
                if sel.any():
                    x1 = den(xt + (1 - t[:, None, None]) * v)                             # clean-latent estimate, VAE units
                    if J is None: l = ctc(reader(x1), tg, lens, sel.cpu(), cf)
                    else:
                        idx = sel.nonzero().flatten()[:args.judge_n]; S = x.shape[-1]
                        y = V.decode(x1[idx].float()); y = F.pad(y, (0, max(0, S - y.shape[-1])))[..., :S]
                        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]); il = idx.tolist()
                        tg_s = torch.cat([tg[offs[i]:offs[i + 1]] for i in il]); lens_s = lens[il]
                        l = J.ctc(J(y), tg_s, lens_s, torch.ones(len(il), dtype=torch.bool), cf[il].float() / 50)
                    if l is not None: ll = l; loss = loss + args.lyr_w * l
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for k, p in R.state_dict().items():
                if p.dtype.is_floating_point: ema[k].mul_(0.999).add_(p.detach(), alpha=0.001)
        for k, val_ in dict(flow=lf, lyr=ll).items(): accl[k] = accl.get(k, 0) + float(val_)
        if step % 100 == 0:
            if AL is not None:                                                     # words aligned / fallbacks per window (x100: printed /100)
                accl["al_words"] = AL.stats["aligned_words"] * 100 / max(AL.stats["windows"], 1); accl["al_fallback"] = AL.stats["fallback"] * 100 / max(AL.stats["windows"], 1)
            msg = f"step {step} " + " ".join(f"{k} {v_ / 100:.4f}" for k, v_ in accl.items()) + f" {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush(); accl = {}
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k} {v_:.4f}" for k, v_ in e.items()); print(msg, flush=True); log.write(msg + "\n"); log.flush()
            torch.save(dict(R=R.state_dict(), opt=opt.state_dict(), ema=ema, step=step, args=vars(args), lat_m=lat_m, lat_s=lat_s), out / "last.tmp"); os.replace(out / "last.tmp", ckp)
            if step % 25000 == 0: torch.save(dict(R=ema, step=step, args=vars(args), lat_m=lat_m, lat_s=lat_s), out / f"R_ema_s{step}.pt")


if __name__ == "__main__":
    main()
