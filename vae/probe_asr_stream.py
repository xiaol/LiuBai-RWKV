#!/usr/bin/env python3
"""Is the Qwen3-ASR stream more lyric-readable than our latent? (vae/asr_stream.py, test 2026-11-01)

1. fits a PCA (1024 -> --pca-dim) of the bounded-lookahead Qwen3-ASR encoder states on training crops -> <out>/pca<k>.pt
   (the projection the LM's extra stream uses)
2. trains identical lyric-CTC readers (vae/probe_lyrics.Reader, as vae/probe_latents.py) in parallel on
     vae          the VAE latent the LM trains on (encode_local, bounded lookahead), 64 dims
     vae[0:32]    its MERT-aligned half
     qasr         Qwen3-ASR ln_post states, same lookahead bound, 1024 dims
     qasr_pca     the PCA projection
     vae+pca      what the LM would predict (64 + k)
     qasr_lin     --lin k: a bias-free Linear 1024 -> k learned jointly with its reader by the CTC loss (lyric-optimal k dims);
                  saved as <out>/lin<k>.pt in the PCA file format (mean, P) for vae/asr_stream.py
--only a,b restricts the readers (names as above, with the dims suffix).
Held-out crops (slim_val_clean), greedy letter CER, lower = the words are more readable.
usage: probe_asr_stream.py --vae out/vae/v4/G_s60000.pt --out out/probe/asr_stream_v4
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import load_vae  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402
from vae.probe_lyrics import Reader  # noqa: E402
from vae.asr_stream import AsrStream  # noqa: E402


class _Center(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.register_buffer("m", m.clone())
    def forward(s, x): return x - s.m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vae", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/pilot_clm_train_clean.jsonl"))
    ap.add_argument("--val-manifest", default=str(ROOT / "data/real_corpus/manifest/slim_val_clean.jsonl"))
    ap.add_argument("--pca-dim", type=int, default=32); ap.add_argument("--pca-batches", type=int, default=40)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=12); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lin", type=int, default=0); ap.add_argument("--only", default="")
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--enc-win", type=float, default=3.0); ap.add_argument("--enc-ahead", type=float, default=0.64); ap.add_argument("--enc-keep", type=int, default=2)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    V = load_vae(args.vae, dev); Q = AsrStream(dev=dev); k = args.pca_dim
    ekw = dict(blk=4, keep=args.enc_keep, win=args.enc_win, ahead=args.enc_ahead)

    @torch.no_grad()
    def base(wav):
        x = wav.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): mu = V.encode_local(x, **ekw)[0].float()
        q = Q.local(x, V.sr, mu.shape[1], blk=4, keep=args.enc_keep, ahead=args.enc_ahead, raw=True)
        return mu, q

    import random; random.seed(0)
    pf = out / f"pca{k}.pt"
    if not pf.exists():                                                   # PCA on training crops (not the probe's eval crops)
        it = iter(Crops(args.manifest, holdout=0, seed=777, sr=V.sr)); X = []
        for _ in range(args.pca_batches): X.append(base(collate([next(it) for _ in range(args.bs)])[0])[1].reshape(-1, 1024))
        X = torch.cat(X); m = X.mean(0); C = torch.cov((X - m).T); ev, U = torch.linalg.eigh(C); ev, U = ev.flip(0), U.flip(1)
        torch.save(dict(mean=m.cpu(), P=U[:, :k].cpu(), ev=ev.cpu(), frames=len(X)), pf)
        print(f"PCA on {len(X)} frames: top-{k} explains {ev[:k].sum() / ev.sum():.3f} of the variance", flush=True)
    p = torch.load(pf); Pm, PP = p["mean"].to(dev), p["P"].to(dev)

    def feats(wav):
        mu, q = base(wav); qp = (q - Pm) @ PP
        return {"vae": mu, "vae[0:32]": mu[..., :32], "qasr": q, f"qasr_pca{k}": qp, f"vae+pca{k}": torch.cat([mu, qp], -1)}

    dims = {"vae": 64, "vae[0:32]": 32, "qasr": 1024, f"qasr_pca{k}": k, f"vae+pca{k}": 64 + k}
    readers = {n: Reader(d, 12).to(dev) for n, d in dims.items()}
    if args.lin:
        lin = torch.nn.Linear(1024, args.lin, bias=False); ln = f"qasr_lin{args.lin}"; dims[ln] = 1024
        readers[ln] = torch.nn.Sequential(); readers[ln].add_module("0", _Center(Pm)); readers[ln].add_module("1", lin); readers[ln].add_module("2", Reader(args.lin, 12))
        readers[ln] = readers[ln].to(dev); feats0 = feats; feats = lambda w: (lambda r: {**r, ln: r["qasr"]})(feats0(w))
    if args.only: keep = set(args.only.split(",")); dims = {n: d for n, d in dims.items() if n in keep}; readers = {n: readers[n] for n in dims}
    names = list(dims)
    opts = {n: torch.optim.AdamW(readers[n].parameters(), lr=args.lr, weight_decay=0.01) for n in names}

    def ctc(lp, tg, lens, ok, cf):
        idx = ok.nonzero().flatten()
        if len(idx) == 0: return None
        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]); tgt = torch.cat([tg[offs[i]:offs[i + 1]] for i in idx.tolist()]).to(dev)
        return F.ctc_loss(lp[idx].transpose(0, 1), tgt, torch.clamp(cf[idx], max=lp.shape[1]).to(dev), lens[idx].to(dev), blank=0, zero_infinity=True)

    random.seed(0)
    vd = Crops(args.val_manifest, holdout=0, seed=12345, sr=V.sr); vd.rows = [r for r in vd.rows if len(r["lines"]) >= 4]; vit = iter(vd)
    val = [collate([next(vit) for _ in range(16)]) for _ in range(8)]; vin = [(feats(b[0]), b) for b in val]

    @torch.no_grad()
    def evaluate():
        res = {}
        for n in names:
            r = readers[n].eval(); e = c = 0
            for x, (_, tg, lens, ok, cf) in vin:
                lp = r(x[n]); offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
                for j in ok.nonzero().flatten().tolist():
                    pr = lp[j, :cf[j]].argmax(-1).tolist(); hyp = [t for i, t in enumerate(pr) if t != 0 and (i == 0 or t != pr[i - 1])]
                    ref = tg[offs[j]:offs[j + 1]].tolist(); e += edit(hyp, ref); c += len(ref)
            res[n] = e / max(c, 1); r.train()
        return res

    ds = Crops(args.manifest, holdout=0, seed=3, sr=V.sr); ds.rows = [r for r in ds.rows if len(r["lines"]) >= 4]
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, collate_fn=collate, prefetch_factor=4, persistent_workers=True))
    log = open(out / "log.txt", "a"); t0 = time.time()
    for step in range(1, args.steps + 1):
        wav, tg, lens, ok, cf = next(dl); x = feats(wav)
        lr = args.lr * min(1, step / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for n in names:
            for pg in opts[n].param_groups: pg["lr"] = lr
            l = ctc(readers[n](x[n]), tg, lens, ok, cf)
            if l is None: continue
            opts[n].zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(readers[n].parameters(), 1.0); opts[n].step()
        if step % 100 == 0: print(f"step {step} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k_} {v:.4f}" for k_, v in res.items()) + f" {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if args.lin: torch.save(dict(mean=Pm.cpu(), P=lin.weight.detach().T.contiguous().cpu(), step=step, cer=res.get(ln)), out / f"lin{args.lin}.pt")


if __name__ == "__main__":
    main()
