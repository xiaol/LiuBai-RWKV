#!/usr/bin/env python3
"""LiuBai VAE pilot trainer (docs/VAE_PLAN.md, Phase A).

Every run trains the codec (multi-scale mel + multi-resolution STFT + KL, STFT-discriminator hinge GAN + feature matching after
--gan-start) AND two readers on the latent mean:
  * lyric CTC reader   (vae/model.LyricCTC)   - probe of how readable the lyrics are from the latent
  * latent prior       (vae/model.LatentPrior) - probe of how predictable the latent is for a causal RWKV
In the baseline both readers see mu.detach() (pure probes, no effect on the codec). --ctc-w > 0 lets the CTC gradient reach the
encoder; --prior-w > 0 adds the prior's normalised next-frame error as an encoder loss (prior weights frozen for that term, so
the encoder adapts to the prior rather than the prior chasing the encoder). Same probes in every variant -> comparable numbers.
usage: train.py --out out/vae/pilot_A [--ctc-w 0] [--prior-w 0] [--steps 40000] [--bs 8]
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, torchaudio

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE, LyricCTC, LatentPrior, STFTDisc, LETTERS  # noqa: E402
from vae.data import Crops, collate, SR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"))
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--seg", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--gan-start", type=int, default=3000)
    ap.add_argument("--mel-w", type=float, default=15.0); ap.add_argument("--stft-w", type=float, default=1.0); ap.add_argument("--kl-w", type=float, default=1e-4)
    ap.add_argument("--adv-w", type=float, default=1.0); ap.add_argument("--fm-w", type=float, default=2.0)
    ap.add_argument("--ctc-w", type=float, default=0.0, help="> 0: lyric CTC gradient reaches the encoder (variant B/D)")
    ap.add_argument("--prior-w", type=float, default=0.0, help="> 0: latent predictability loss on the encoder (variant C/D)")
    ap.add_argument("--prior-start", type=int, default=5000, help="encoder predictability term only after the codec has formed")
    ap.add_argument("--ch", type=int, default=32); ap.add_argument("--d", type=int, default=512); ap.add_argument("--n-rwkv", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    json.dump(vars(args), open(out / "config.json", "w"), indent=1)

    G = VAE(ch=args.ch, d=args.d, n_rwkv=args.n_rwkv).to(dev); D = STFTDisc().to(dev)
    ctc = LyricCTC().to(dev); prior = LatentPrior().to(dev)
    optG = torch.optim.AdamW(list(G.parameters()) + list(ctc.parameters()) + list(prior.parameters()), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    optD = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    n_par = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"VAE {n_par(G):.1f}M (enc conv {n_par(G.enc_conv):.1f}M, dec conv {n_par(G.dec_conv):.1f}M) D {n_par(D):.1f}M ctc {n_par(ctc):.1f}M prior {n_par(prior):.1f}M", flush=True)
    step = 0; ck = out / "last.pt"
    if ck.exists():
        sd = torch.load(ck, map_location=dev); G.load_state_dict(sd["G"]); D.load_state_dict(sd["D"]); ctc.load_state_dict(sd["ctc"]); prior.load_state_dict(sd["prior"])
        optG.load_state_dict(sd["optG"]); optD.load_state_dict(sd["optD"]); step = sd["step"]; print("resumed at step", step, flush=True)

    mels = [torchaudio.transforms.MelSpectrogram(SR, n_fft=n, hop_length=n // 4, n_mels=m, power=1.0).to(dev) for n, m in ((512, 40), (1024, 80), (2048, 128))]
    wins = {n: torch.hann_window(n, device=dev) for n in (2048, 1024, 512, 256)}
    def recon_losses(y, x):                     # [B,2,S] float32
        y, x = y.reshape(-1, y.shape[-1]).float(), x.reshape(-1, x.shape[-1]).float()
        mel = sum((torch.log(m(y) + 1e-5) - torch.log(m(x) + 1e-5)).abs().mean() for m in mels) / len(mels)
        st = 0
        for n, w in wins.items():
            A = torch.stft(y, n, n // 4, window=w, return_complex=True).abs(); B = torch.stft(x, n, n // 4, window=w, return_complex=True).abs()
            st = st + (A - B).norm() / (B.norm() + 1e-6) + (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().mean()
        return mel, st / len(wins)
    def kl(mu, lv): return 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).mean()
    def ctc_loss(mu, tg, lens, ok, cf, grad):
        lp = ctc(mu if grad else mu.detach())                                                # [B, T*4, V]
        idx = ok.nonzero().flatten()
        if len(idx) == 0: return torch.zeros((), device=dev), lp
        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
        tgt = torch.cat([tg[offs[i]:offs[i + 1]] for i in idx.tolist()]).to(dev)
        il = torch.clamp(cf[idx], max=lp.shape[1]).to(dev)
        l = F.ctc_loss(lp[idx].transpose(0, 1), tgt, il, lens[idx].to(dev), blank=0, reduction="mean", zero_infinity=True)
        return l, lp
    def prior_terms(mu, z, enc_grad):
        tgt = mu[:, 1:]; var = tgt.detach().var(dim=(0, 1), keepdim=True) + 1e-4
        lp = ((prior(mu.detach()) - tgt.detach()) ** 2 / var).mean()                         # trains the prior on mu (probe in every run)
        if not enc_grad: return lp, torch.zeros((), device=dev)
        # encoder term on the SAMPLED latent: posterior noise is unpredictable, so the encoder can only lower it by keeping the
        # signal strong relative to its noise. Applied to mu it collapsed (KL -> 0.002, information hidden under the noise).
        for p in prior.parameters(): p.requires_grad_(False)
        zt = z[:, 1:]; le = ((prior(z) - zt) ** 2 / (zt.detach().var(dim=(0, 1), keepdim=True) + 1e-4)).mean()
        for p in prior.parameters(): p.requires_grad_(True)
        return lp, le

    # fixed validation crops from the held-out shard-0 rows
    vd = Crops(args.manifest, seg=args.seg, holdout=0, max_rows=1000, seed=12345); vd.rows = vd.rows[:1000]
    vit = iter(vd); val = [next(vit) for _ in range(48)]
    @torch.no_grad()
    def evaluate():
        G.eval(); ctc.eval(); prior.eval(); mel_db = st_ = pr = 0; errs = chars = 0
        for i in range(0, len(val), 8):
            wav, tg, lens, ok, cf = collate(val[i:i + 8]); x = wav.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): y, mu, lv, z = G(x, sample=False)
            m, s_ = recon_losses(y, x); mel_db += m.item() * 20 / math.log(10) / 6; st_ += s_.item() / 6
            tgt = mu[:, 1:].float(); pr += (((prior(mu.float()) - tgt) ** 2) / (tgt.var(dim=(0, 1), keepdim=True) + 1e-4)).mean().item() / 6
            lp = ctc(mu.float()); offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
            for j in ok.nonzero().flatten().tolist():
                p = lp[j, :cf[j]].argmax(-1).tolist(); hyp = [c for k, c in enumerate(p) if c != 0 and (k == 0 or c != p[k - 1])]
                ref = tg[offs[j]:offs[j + 1]].tolist(); errs += edit(hyp, ref); chars += len(ref)
        G.train(); ctc.train(); prior.train()
        return dict(mel_db=mel_db, stft=st_, prior_nmse=pr, ctc_cer=errs / max(chars, 1))
    if step == 0:
        e = evaluate(); print("EVAL step 0 " + " ".join(f"{k} {v:.4f}" for k, v in e.items()), flush=True)

    dl = torch.utils.data.DataLoader(Crops(args.manifest, seg=args.seg, seed=args.seed + step), batch_size=args.bs, num_workers=args.workers,
                                     collate_fn=collate, prefetch_factor=4, persistent_workers=True)
    log = open(out / "log.txt", "a"); t0 = time.time(); acc = {}
    for wav, tg, lens, ok, cf in dl:
        step += 1
        if step > args.steps: break
        x = wav.to(dev, non_blocking=True)
        lr = args.lr * min(1, step / 1000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for o in (optG, optD):
            for g in o.param_groups: g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y, mu, lv, z = G(x)
        mu = mu.float(); y = y.float()
        mel, st = recon_losses(y, x); lkl = kl(mu, lv.float())
        lc, _ = ctc_loss(mu, tg, lens, ok, cf, grad=args.ctc_w > 0)
        lpr, lpe = prior_terms(mu, z.float(), enc_grad=args.prior_w > 0 and step > args.prior_start)
        loss = args.mel_w * mel + args.stft_w * st + args.kl_w * lkl + (args.ctc_w if args.ctc_w > 0 else 1.0) * lc + lpr + args.prior_w * lpe
        gan = step > args.gan_start
        if gan:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dr = D(x); df = D(y)
            adv = sum(F.relu(1 - l.float()).mean() for l, _ in df) / len(df)
            fm = sum(sum((a.float() - b.float().detach()).abs().mean() / (b.float().detach().abs().mean() + 1e-5) for a, b in zip(ff, fr)) / len(ff)
                     for (_, ff), (_, fr) in zip(df, dr)) / len(df)
            loss = loss + args.adv_w * adv + args.fm_w * fm
        optG.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for g in optG.param_groups for p in g["params"]], 1e3)
        optG.step()
        if gan:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dr = D(x); df = D(y.detach())
            ld = sum(F.relu(1 - a.float()).mean() + F.relu(1 + b.float()).mean() for (a, _), (b, _) in zip(dr, df)) / len(dr)
            optD.zero_grad(set_to_none=True); ld.backward(); torch.nn.utils.clip_grad_norm_(D.parameters(), 1e3); optD.step()
        for k, v in dict(mel=mel, stft=st, kl=lkl, ctc=lc, prior=lpr, prior_enc=lpe, gn=gn).items(): acc[k] = acc.get(k, 0) + float(v)
        if gan: acc["adv"] = acc.get("adv", 0) + float(adv); acc["fm"] = acc.get("fm", 0) + float(fm); acc["d"] = acc.get("d", 0) + float(ld)
        if step % 50 == 0:
            msg = f"step {step} " + " ".join(f"{k} {v / 50:.4f}" for k, v in acc.items()) + f" lr {lr:.2e} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush(); acc = {}
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k} {v:.4f}" for k, v in e.items()); print(msg, flush=True); log.write(msg + "\n"); log.flush()
            sd = dict(G=G.state_dict(), D=D.state_dict(), ctc=ctc.state_dict(), prior=prior.state_dict(), optG=optG.state_dict(), optD=optD.state_dict(), step=step, args=vars(args))
            torch.save(sd, out / "last.tmp"); os.replace(out / "last.tmp", ck)
            if step % 10000 == 0: torch.save(dict(G=G.state_dict(), ctc=ctc.state_dict(), prior=prior.state_dict(), step=step, args=vars(args)), out / f"G_s{step}.pt")
    print("done", step, flush=True)


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


if __name__ == "__main__":
    main()
