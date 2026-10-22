#!/usr/bin/env python3
"""Phase B trainer: lyric-aware FSQ tokens on the frozen LiuBai VAE (vae/tokenizer.py).

Crops come from the same loader as the VAE (8 s, LRC-anchored, romanised targets). The VAE encoder runs frozen on the fly
(no latent cache needed for the pilot); the tokenizer is trained on
    loss = rec_w * MSE(decoded latent, mu) / var(mu)  +  ctc_w * CTC(lyric reader on the quantised codes)
Evaluation on held-out shard-0 crops: latent NMSE, greedy CTC CER from the CODES (the number that says whether the RWKV LM's
tokens carry the words), and code usage (perplexity of the code histogram).
usage: train_tokenizer.py --vae out/vae/v1/G_s50000.pt --out out/tok/b1 [--ctc-w 1.0] [--steps 60000]
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"))
    ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rec-w", type=float, default=1.0); ap.add_argument("--ctc-w", type=float, default=1.0)
    ap.add_argument("--n-q", type=int, default=1, help="FSQ codes per latent frame (product quantisation)")
    ap.add_argument("--levels", default="8,8,8,6,5"); ap.add_argument("--workers", type=int, default=10); ap.add_argument("--eval-every", type=int, default=2000)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1); torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a.get("mults", "1,2,4,8,16,16").split(",")), d=a["d"], n_rwkv=a["n_rwkv"])
    V.load_state_dict(ck["G"]); V = V.to(dev).eval().requires_grad_(False)
    T = SemTokenizer(levels=tuple(int(x) for x in args.levels.split(",")), n_q=args.n_q).to(dev)
    opt = torch.optim.AdamW(T.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    print(f"tokenizer {sum(p.numel() for p in T.parameters()) / 1e6:.1f}M, {T.fsq.n_codes} codes, VAE {args.vae} step {ck['step']}", flush=True)

    @torch.no_grad()
    def latent(x):
        with torch.autocast("cuda", dtype=torch.bfloat16): mu, _ = V.encode(x)
        return mu.float()

    def ctc_loss(lp, tg, lens, ok, cf):
        idx = ok.nonzero().flatten()
        if len(idx) == 0: return torch.zeros((), device=dev)
        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
        tgt = torch.cat([tg[offs[i]:offs[i + 1]] for i in idx.tolist()]).to(dev)
        return F.ctc_loss(lp[idx].transpose(0, 1), tgt, torch.clamp(cf[idx], max=lp.shape[1]).to(dev), lens[idx].to(dev), blank=0, zero_infinity=True)

    vd = Crops(args.manifest, holdout=0, max_rows=1000, seed=12345); vit = iter(vd); val = [next(vit) for _ in range(96)]
    @torch.no_grad()
    def evaluate():
        T.eval(); nm = 0; errs = chars = 0; hist = torch.zeros(T.fsq.n_codes, device=dev)
        for i in range(0, len(val), 16):
            wav, tg, lens, ok, cf = collate(val[i:i + 16]); mu = latent(wav.to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16): rec, lp, ids = T(mu)
            nm += (((rec.float() - mu) ** 2).mean() / mu.var()).item() / 6; hist += torch.bincount(ids.flatten(), minlength=T.fsq.n_codes)
            offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
            for j in ok.nonzero().flatten().tolist():
                p = lp[j, :cf[j]].argmax(-1).tolist(); hyp = [c for k, c in enumerate(p) if c != 0 and (k == 0 or c != p[k - 1])]
                ref = tg[offs[j]:offs[j + 1]].tolist(); errs += edit(hyp, ref); chars += len(ref)
        pr = hist / hist.sum(); ppl = math.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum().item())
        T.train(); return dict(latent_nmse=nm, code_ctc_cer=errs / max(chars, 1), code_ppl=ppl)

    dl = torch.utils.data.DataLoader(Crops(args.manifest, seed=1), batch_size=args.bs, num_workers=args.workers, collate_fn=collate, prefetch_factor=4, persistent_workers=True)
    log = open(out / "log.txt", "a"); t0 = time.time(); step = 0; acc = {}
    e = evaluate(); print("EVAL step 0 " + " ".join(f"{k} {v:.4f}" for k, v in e.items()), flush=True)
    for wav, tg, lens, ok, cf in dl:
        step += 1
        if step > args.steps: break
        for g in opt.param_groups: g["lr"] = args.lr * min(1, step / 1000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        mu = latent(wav.to(dev, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, ids = T.encode(mu); rec = T.decode(q)
            lp = T.lyric_logp(q if args.ctc_w > 0 else q.detach())       # ctc_w 0: reader still trains, as a probe only
        lr_ = ((rec.float() - mu) ** 2).mean() / mu.var(); lc = ctc_loss(lp, tg, lens, ok, cf)
        loss = args.rec_w * lr_ + (args.ctc_w if args.ctc_w > 0 else 1.0) * lc
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0); opt.step()
        for k, v in dict(rec=lr_, ctc=lc).items(): acc[k] = acc.get(k, 0) + float(v)
        if step % 50 == 0:
            msg = f"step {step} " + " ".join(f"{k} {v / 50:.4f}" for k, v in acc.items()) + f" {time.time() - t0:.0f}s"; print(msg, flush=True); log.write(msg + "\n"); log.flush(); acc = {}
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k} {v:.4f}" for k, v in e.items()); print(msg, flush=True); log.write(msg + "\n"); log.flush()
            torch.save(dict(T=T.state_dict(), step=step, args=vars(args)), out / "last.pt")
    print("done", step)


if __name__ == "__main__":
    main()
