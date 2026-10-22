#!/usr/bin/env python3
"""Which representation carries the sung words? Same lyric-CTC reader, same 8 s LRC-anchored crops, trained in parallel on:

  vae_mu      our VAE latent (12.5 Hz x 64)
  our_codes   our FSQ tokens (12.5 Hz, 15,360 codes; reader sees the code embedding it learns)
  mert_L{k}   MERT-v2-FullSong layer k, continuous (25 Hz x 1024), k in --layers
  mertkm_L{k} k-means units of MERT layer k (--k clusters fitted on the first --fit-batches batches; HuBERT/MusicLM-style tokens)

Readers: input proj -> upsample to 50 Hz -> 2 BiRWKV blocks (d 256) -> CTC over romanised letters (identical capacity).
Reported on held-out shard-0 crops: greedy CER (lower = the words are more readable from that representation).
usage: probe_lyrics.py --vae out/vae/v1/G_s200000.pt --tok out/tok/b1_ctc1/last.pt --out out/probe/p1 [--steps 8000]
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from vae.model import VAE, N_CTC  # noqa: E402
from vae.rwkv import BiBlock  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402
YUE2 = ROOT / "models/yue2"


class Reader(nn.Module):
    def __init__(s, din, rate, d=256, n=2, vocab=0):
        super().__init__(); s.up = 50 // rate; s.emb = nn.Embedding(vocab, din) if vocab else None
        s.inp = nn.Linear(din, d * s.up); s.blocks = nn.ModuleList(BiBlock(d) for _ in range(n)); s.out = nn.Linear(d, N_CTC)
    def forward(s, x):
        if s.emb is not None: x = s.emb(x)
        B, T, _ = x.shape; h = s.inp(x.float()).view(B, T * s.up, -1)
        for b in s.blocks: h = b(h)
        return F.log_softmax(s.out(h).float(), -1)


def kmeans(X, k, iters=25):
    C = X[torch.randperm(len(X), device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cat([torch.cdist(X[i:i + 65536], C).argmin(1) for i in range(0, len(X), 65536)])
        S = torch.zeros_like(C).index_add_(0, a, X); n = torch.bincount(a, minlength=k).float()
        dead = n == 0; C = torch.where(dead[:, None], X[torch.randint(len(X), (k,), device=X.device)], S / n.clamp_min(1)[:, None])
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"))
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--layers", default="12,20,23"); ap.add_argument("--km-layers", default="12,20"); ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--fit-batches", type=int, default=150); ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0)
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
    tk = torch.load(args.tok, map_location="cpu"); TK = SemTokenizer(levels=tuple(int(x) for x in tk["args"]["levels"].split(","))); TK.load_state_dict(tk["T"]); TK = TK.to(dev).eval()
    from transformers import AutoModel, AutoFeatureExtractor
    proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
    mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()
    L = [int(x) for x in args.layers.split(",")]; KL = [int(x) for x in args.km_layers.split(",")]; allL = sorted(set(L + KL))

    @torch.no_grad()
    def feats(wav):                                  # wav [B,2,S] 24 kHz -> dict name -> tensor
        x = wav.to(dev); r = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, _ = V.encode(x); mu = mu.float(); r["vae_mu"] = mu; qv, r["our_codes"] = TK.encode(mu); r["our_q"] = qv.float()
            m = x.mean(1).float(); inp = proc([w.cpu().numpy() for w in m], sampling_rate=24000, return_tensors="pt")
            hs = mert(**{k: v.to(dev) for k, v in inp.items()}, output_hidden_states=True).hidden_states
        T25 = int(round(x.shape[-1] / 24000 * 25))
        for l in allL:
            h = F.interpolate(hs[l].float().transpose(1, 2), size=T25, mode="linear", align_corners=False).transpose(1, 2)
            h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-5)          # per-crop instance norm (as the heads use)
            r[f"mert_L{l}"] = h
        return r

    dl = iter(torch.utils.data.DataLoader(Crops(args.manifest, seed=3), batch_size=args.bs, num_workers=args.workers, collate_fn=collate, prefetch_factor=4, persistent_workers=True))
    t0 = time.time(); fit = {l: [] for l in KL}
    for _ in range(args.fit_batches):
        f = feats(next(dl)[0])
        for l in KL: fit[l].append(f[f"mert_L{l}"].reshape(-1, 1024)[::2].half())
    cents = {}
    for l in KL:
        X = torch.cat(fit[l]).float(); cents[l] = kmeans(X, args.k); del X
        print(f"k-means L{l}: {sum(len(x) for x in fit[l])} frames -> {args.k} centroids ({time.time() - t0:.0f}s)", flush=True)
    del fit; torch.cuda.empty_cache()
    @torch.no_grad()
    def units(h, l): return torch.cat([torch.cdist(c, cents[l][None]).argmin(-1) for c in h.split(4)])

    names = ["vae_mu", "our_codes", "our_q"] + [f"mert_L{l}" for l in L] + [f"mertkm_L{l}" for l in KL] + [f"mertkmvec_L{l}" for l in KL]
    readers = {"vae_mu": Reader(64, 12), "our_codes": Reader(128, 12, vocab=TK.fsq.n_codes), "our_q": Reader(5, 12)}
    for l in L: readers[f"mert_L{l}"] = Reader(1024, 25)
    for l in KL: readers[f"mertkm_L{l}"] = Reader(128, 25, vocab=args.k); readers[f"mertkmvec_L{l}"] = Reader(1024, 25)
    for r in readers.values(): r.to(dev)
    opts = {n: torch.optim.AdamW(readers[n].parameters(), lr=args.lr, weight_decay=0.01) for n in names}
    print({n: f"{sum(p.numel() for p in readers[n].parameters()) / 1e6:.1f}M" for n in names}, flush=True)

    def inputs(f):
        d = {n: f[n] for n in ("vae_mu", "our_codes", "our_q")}
        for l in L: d[f"mert_L{l}"] = f[f"mert_L{l}"]
        for l in KL: u = units(f[f"mert_L{l}"], l); d[f"mertkm_L{l}"] = u; d[f"mertkmvec_L{l}"] = cents[l][u]   # same units as centroid vectors
        return d

    def ctc(lp, tg, lens, ok, cf):
        idx = ok.nonzero().flatten()
        if len(idx) == 0: return None
        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]); tgt = torch.cat([tg[offs[i]:offs[i + 1]] for i in idx.tolist()]).to(dev)
        return F.ctc_loss(lp[idx].transpose(0, 1), tgt, torch.clamp(cf[idx], max=lp.shape[1]).to(dev), lens[idx].to(dev), blank=0, zero_infinity=True)

    vd = Crops(args.manifest, holdout=0, max_rows=1000, seed=12345); vit = iter(vd); val = [collate([next(vit) for _ in range(16)]) for _ in range(8)]
    vin = [(inputs(feats(b[0])), b) for b in val]
    @torch.no_grad()
    def evaluate():
        res = {}
        for n in names:
            r = readers[n].eval(); e = c = 0
            for x, (_, tg, lens, ok, cf) in vin:
                lp = r(x[n]); offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
                for j in ok.nonzero().flatten().tolist():
                    p = lp[j, :cf[j]].argmax(-1).tolist(); hyp = [t for i, t in enumerate(p) if t != 0 and (i == 0 or t != p[i - 1])]
                    ref = tg[offs[j]:offs[j + 1]].tolist(); e += edit(hyp, ref); c += len(ref)
            res[n] = e / max(c, 1); r.train()
        return res

    log = open(out / "log.txt", "a"); json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    for step in range(1, args.steps + 1):
        wav, tg, lens, ok, cf = next(dl); x = inputs(feats(wav))
        for g in (o for o in opts.values() for o in [o]):
            for pg in g.param_groups: pg["lr"] = args.lr * min(1, step / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for n in names:
            l = ctc(readers[n](x[n]), tg, lens, ok, cf)
            if l is None: continue
            opts[n].zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(readers[n].parameters(), 1.0); opts[n].step()
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k} {v:.4f}" for k, v in res.items()) + f" {time.time() - t0:.0f}s"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()


if __name__ == "__main__":
    main()
