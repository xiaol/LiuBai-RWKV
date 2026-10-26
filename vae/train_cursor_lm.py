#!/usr/bin/env python3
"""Train the gate-1 block-flow RWKV LM (vae/cursor_lm.py) on 24 s LRC-anchored crops of the real corpus, latent = frozen VAE v2.

loss : per-block rectified flow on the normalised latent (K noise/t draws per block, frames sung by cut lines masked)
       + adv_w * CE of the cursor advance (cursor mode)
aug  : context blocks get Gaussian noise of relative power U(0, --ctx-noise) per crop (exposure bias; VAE v2 is trained to
       decode such latents), lyrics dropped to null with p --p-lyr (CFG / no-lyrics control)
latent: VAE.encode_local (MERT branch on <= --enc-win s windows with --enc-ahead s lookahead, one block at a time): the
       context latents carry no information from more than --enc-ahead s past their own block (VAE.encode leaks the whole
       crop through MERT, so the context already told the model what is sung next and the lyrics were redundant)
loss weighting: flow error on the MERT-aligned dims [0, sem) x --sem-w (weights renormalised to mean 1)
eval : teacher-forced flow loss on held-out crops with own / swapped (another crop's letters, own timing) / null lyrics at a
       fixed t grid and fixed noise -> binding gaps; advance accuracy. Generation + Whisper gate: vae/eval_cursor_lm.py
usage: train_cursor_lm.py --mode cursor --out out/clm/cursor1
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.align_letters import LetterAligner  # noqa: E402
from vae.cursor_lm import CursorLM, cursor_arrays, windows, PAD, NULL  # noqa: E402


def lyric_inputs(M, al, NB, drop, roll=0):
    """al: LetterAligner output per sample -> (lyr tensor, a_cond [B,NB], adv target [B,NB]); roll: take letters from sample
    (b + roll) % B with own timing (swap control); drop[b]: null lyrics"""
    B = len(al); a_t = np.zeros((B, NB), np.int64)
    if M.mode == "cursor":
        lyr = np.zeros((B, NB, M.wb + M.wf), np.int64); a_c = np.zeros((B, NB), np.int64)
        for b in range(B):
            c, a = cursor_arrays(al[b][1], NB, M.blk); a_t[b] = np.minimum(a, M.max_adv - 1)
            lyr[b] = NULL if drop[b] else windows(al[(b + roll) % B][0], c, M.wb, M.wf)
            a_c[b] = M.max_adv if drop[b] else a_t[b]
        return torch.from_numpy(lyr), torch.from_numpy(a_c), torch.from_numpy(a_t)
    lyr = np.full((B, M.max_prefix), PAD, np.int64)
    for b in range(B):
        ids = [] if drop[b] else al[(b + roll) % B][0][:M.max_prefix]
        if ids: lyr[b, -len(ids):] = ids
    return torch.from_numpy(lyr), None, torch.from_numpy(a_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cursor", choices=["cursor", "prefix"]); ap.add_argument("--out", required=True)
    ap.add_argument("--vae", default=str(ROOT / "out/vae/v2/G_s60000.pt"))
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/pilot_clm_train.jsonl"))
    ap.add_argument("--val-manifest", default=str(ROOT / "data/real_corpus/manifest/slim_val.jsonl"), help="held-out test-shard rows (slim_manifest.py)")
    ap.add_argument("--seg", type=float, default=24.0); ap.add_argument("--blk", type=int, default=4)
    ap.add_argument("--d", type=int, default=1024); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--head-d", type=int, default=1024); ap.add_argument("--head-n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--bs", type=int, default=12); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--adv-w", type=float, default=0.2)
    ap.add_argument("--ctx-noise", type=float, default=0.3); ap.add_argument("--p-lyr", type=float, default=0.1)
    ap.add_argument("--p-instr", type=float, default=0.1, help="share of training rows without timed lyrics (instrumental)")
    ap.add_argument("--enc", default="local", choices=["local", "full"]); ap.add_argument("--enc-win", type=float, default=3.0)
    ap.add_argument("--enc-ahead", type=float, default=0.64); ap.add_argument("--enc-keep", type=int, default=2)
    ap.add_argument("--sem-w", type=float, default=1.0, help="flow-loss weight of the semantic dims [0, VAE.sem) relative to the rest")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--eval-every", type=int, default=2500)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1); torch.backends.cuda.matmul.allow_tf32 = True
    from vae.model import load_vae; V = load_vae(args.vae, dev)                  # attaches MERT for sem (v4) VAEs
    LA = LetterAligner(dev); ekw = dict(blk=args.blk, keep=args.enc_keep, win=args.enc_win, ahead=args.enc_ahead)
    encode = (lambda x: V.encode_local(x, **ekw)[0]) if args.enc == "local" and V.sem else (lambda x: V.encode(x)[0])
    K_sem = V.sem or 0; dw = torch.ones(64, device=dev); dw[:K_sem] = args.sem_w; dw = dw / dw.mean()   # flow-loss dim weights
    M = CursorLM(blk=args.blk, d=args.d, n=args.n, mode=args.mode, head_d=args.head_d, head_n=args.head_n).to(dev)
    opt = torch.optim.AdamW(M.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    ema = {k: v.detach().clone() for k, v in M.state_dict().items()}; random.seed(0)   # Crops mixes global random into its seeds
    step = 0; ckp = out / "last.pt"
    if ckp.exists():
        sd = torch.load(ckp, map_location=dev); M.load_state_dict(sd["M"]); opt.load_state_dict(sd["opt"]); ema = sd["ema"]; step = sd["step"]
        lat_m, lat_s = sd["lat_m"].to(dev), sd["lat_s"].to(dev); print("resumed", step, flush=True)
    else:                                                       # per-channel latent stats from 128 training crops (fixed for the run)
        sit = iter(Crops(args.manifest, holdout=0, seed=999, seg=args.seg)); fs = []
        with torch.no_grad():
            for _ in range(8):
                xb = collate([next(sit) for _ in range(16)])[0].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): fs.append(encode(xb).float().reshape(-1, 64))
        f = torch.cat(fs); lat_m, lat_s = f.mean(0), f.std(0).clamp_min(1e-3)
    nrm = lambda z: (z - lat_m) / lat_s
    print(f"cursor LM ({args.mode}) {sum(p.numel() for p in M.parameters()) / 1e6:.1f}M", flush=True)

    @torch.no_grad()
    def prep(wav):
        x = wav.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): mu = encode(x).float()
        NB = mu.shape[1] // args.blk; mu = mu[:, :NB * args.blk]
        return x, nrm(mu).reshape(mu.shape[0], NB, -1), NB

    def flow_loss(h, z, fm, a_c, gen=None, tgrid=None, dims=None):
        """masked flow MSE, dims weighted by dw; dims=slice -> unweighted error on those dims only (eval)"""
        B, NB, D = z.shape; K = args.K
        c = M.cond(h, a_c.to(dev) if a_c is not None else None)[:, :, None].expand(-1, -1, K, -1)
        zr = z[:, :, None].expand(B, NB, K, D)
        t = torch.sigmoid(torch.randn(B, NB, K, device=dev, generator=gen)) if tgrid is None else tgrid.view(1, 1, K).expand(B, NB, K)
        n = torch.randn(zr.shape, device=dev, generator=gen); xt = (1 - t[..., None]) * n + t[..., None] * zr
        e2 = ((M.head(xt, t, c).float() - (zr - n)) ** 2).view(B, NB, K, args.blk, -1)
        err = e2[..., dims].mean(-1) if dims is not None else (e2 * dw).mean(-1)
        w = fm[:, :, None].expand(B, NB, K, args.blk)
        return (err * w).sum() / w.sum().clamp_min(1)

    def adv_loss(h, a_t, valid):
        if M.mode != "cursor" or not valid.any(): return torch.zeros((), device=dev), 0.0
        lg = M.adv(h).float()[valid]; tg = a_t.to(dev)[valid]
        return F.cross_entropy(lg, tg), (lg.argmax(-1) == tg).float().mean().item()

    random.seed(1); vd = Crops(args.val_manifest, holdout=0, seed=12345, seg=args.seg, with_lines=True); vd.rows = [r for r in vd.rows if len(r["lines"]) >= 4]; vit = iter(vd)
    val = [collate([next(vit) for _ in range(args.bs)]) for _ in range(3)]

    @torch.no_grad()
    def evaluate():
        live = {k: v.clone() for k, v in M.state_dict().items()}; M.load_state_dict(ema); M.eval()
        tgrid = torch.tensor([0.3, 0.5, 0.7, 0.9], device=dev)[:args.K] if args.K <= 4 else None
        r = {k: [] for k in ("fl_own", "fl_swap", "fl_null", "sem_own", "sem_swap", "sem_null", "adv_acc", "adv_ce")}
        for wav, *_, lines in val:
            x, z, NB = prep(wav); al = LA(x, lines, NB * args.blk); B = z.shape[0]
            fm = torch.from_numpy(np.stack([m[:NB * args.blk] for _, _, m in al])).to(dev).view(B, NB, args.blk).float()
            for name, roll, drop in (("own", 0, False), ("swap", 1, False), ("null", 0, True)):
                lyr, a_c, a_t = lyric_inputs(M, al, NB, [drop] * B, roll)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = M.hidden(z, lyr.to(dev))
                    g = torch.Generator(device=dev).manual_seed(7)
                    r[f"fl_{name}"].append(flow_loss(h, z, fm, a_c, gen=g, tgrid=tgrid, dims=slice(None)).item())
                    g = torch.Generator(device=dev).manual_seed(7)                      # same noise: lyric-dim-only error
                    r[f"sem_{name}"].append(flow_loss(h, z, fm, a_c, gen=g, tgrid=tgrid, dims=slice(0, K_sem or 64)).item())
                    if name == "own":
                        l, acc = adv_loss(h, a_t, fm.min(-1).values > 0); r["adv_ce"].append(float(l)); r["adv_acc"].append(acc)
        M.load_state_dict(live); M.train()
        r = {k: float(np.mean(v)) for k, v in r.items()}
        r["gap_swap"] = r["fl_swap"] - r["fl_own"]; r["gap_null"] = r["fl_null"] - r["fl_own"]
        r["sgap_swap"] = r["sem_swap"] - r["sem_own"]; r["sgap_null"] = r["sem_null"] - r["sem_own"]; return r

    ds = Crops(args.manifest, holdout=0, seed=31 + step, seg=args.seg, with_lines=True); rng = np.random.default_rng(0)
    voc = [r for r in ds.rows if len(r["lines"]) >= 4]; ins = [r for r in ds.rows if not r["lines"]]; ds.rows = voc
    if args.p_instr > 0:
        k = min(len(ins), int(len(voc) * args.p_instr / (1 - args.p_instr))); ds.rows = voc + [ins[i] for i in rng.choice(len(ins), k, replace=False)]
    print(f"train rows: {len(voc)} with >= 4 timed lines + {len(ds.rows) - len(voc)} instrumental", flush=True)
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=args.bs,
                                          num_workers=args.workers, collate_fn=collate, prefetch_factor=4, persistent_workers=True))
    log = open(out / "log.txt", "a"); t0 = time.time(); accl = {}
    while step < args.steps:
        step += 1; wav, *_, lines = next(dl)
        for g in opt.param_groups: g["lr"] = args.lr * min(1, step / 2000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        x, z, NB = prep(wav); B = z.shape[0]; al = LA(x, lines, NB * args.blk)
        fm = torch.from_numpy(np.stack([m[:NB * args.blk] for _, _, m in al])).to(dev).view(B, NB, args.blk).float()
        drop = (torch.rand(B) < args.p_lyr).tolist(); lyr, a_c, a_t = lyric_inputs(M, al, NB, drop)
        sig = (torch.rand(B, 1, 1, device=dev) * args.ctx_noise).sqrt(); z_ctx = z + sig * torch.randn_like(z)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = M.hidden(z_ctx, lyr.to(dev))
            lf = flow_loss(h, z, fm, a_c)
            valid = (fm.min(-1).values > 0) & ~torch.tensor(drop, device=dev)[:, None]
            la, acc = adv_loss(h, a_t, valid)
            loss = lf + args.adv_w * la
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
        dec = min(0.999, (1 + step) / (10 + step))                  # EMA warm-up: no trace of the random init after a few k steps
        with torch.no_grad():
            for k, p in M.state_dict().items():
                if p.dtype.is_floating_point: ema[k].mul_(dec).add_(p.detach(), alpha=1 - dec)
        for k, v_ in dict(flow=lf, adv=la, adv_acc=acc, gn=gn, letters=float(np.mean([len(q[0]) for q in al]))).items(): accl[k] = accl.get(k, 0) + float(v_)
        if step % 100 == 0:
            accl["fallback"] = LA.stats["fallback"] * 100 / max(LA.stats["windows"], 1)
            msg = f"step {step} " + " ".join(f"{k} {v_ / 100:.4f}" for k, v_ in accl.items()) + f" {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush(); accl = {}
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k} {v_:.4f}" for k, v_ in e.items()); print(msg, flush=True); log.write(msg + "\n"); log.flush()
            sv = dict(M=M.state_dict(), opt=opt.state_dict(), ema=ema, step=step, args=vars(args), lat_m=lat_m, lat_s=lat_s)
            torch.save(sv, out / "last.tmp"); os.replace(out / "last.tmp", ckp)
            if step % 20000 == 0: torch.save(dict(M=ema, step=step, args=vars(args), lat_m=lat_m, lat_s=lat_s), out / f"M_ema_s{step}.pt")


if __name__ == "__main__":
    main()
