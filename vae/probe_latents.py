#!/usr/bin/env python3
"""Does a VAE latent carry the sung words? Identical lyric-CTC readers (vae/probe_lyrics.Reader: 2 BiRWKV blocks, d 256)
trained in parallel on several VAE checkpoints' latent means (and slices of them), with MERT-v2 L12 as the reference ceiling.

  <name>          full 64-dim mu of checkpoint <name>
  <name>[a:b]     channel slice (e.g. v3's MERT-aligned half 0:32 and chroma half 32:64)
  mert_L12        MERT-v2-FullSong layer 12, instance-normalised (best representation measured so far: CER 0.399)
Held-out crops from the test shard (slim_val.jsonl); greedy letter CER, lower = the words are more readable.
usage: probe_latents.py --vae v2=out/vae/v2/G_s60000.pt --vae v3=out/vae/v3/G_s20000.pt --slices v3:0:32,v3:32:64 --out out/probe/v3_20k
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import load_vae  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402
from vae.probe_lyrics import Reader  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vae", action="append", required=True, help="name=checkpoint")
    ap.add_argument("--slices", default="", help="name:a:b,... extra readers on channel slices"); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/pilot_clm_train.jsonl"))
    ap.add_argument("--val-manifest", default=str(ROOT / "data/real_corpus/manifest/slim_val.jsonl"))
    ap.add_argument("--mert", type=int, default=12, help="MERT layer for the reference reader (0 = none)")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=12); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    vaes = {}
    for spec in args.vae:
        n, p = spec.split("=", 1); vaes[n] = load_vae(p, dev)
    sl = [(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in args.slices.split(",") if s]
    mert = None
    if args.mert:
        from vae.align_targets import MertTarget
        mert = MertTarget(args.mert).to(dev)
    names = list(vaes) + [f"{n}[{a}:{b}]" for n, a, b in sl] + ([f"mert_L{args.mert}"] if mert else [])

    @torch.no_grad()
    def feats(wav):
        x = wav.to(dev); r = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):                            # micro-batches of 2: runs beside a 35 GB trainer
            for n, V in vaes.items(): r[n] = torch.cat([V.encode(c)[0].float() for c in x.split(2)])
        for n, a, b in sl: r[f"{n}[{a}:{b}]"] = r[n][..., a:b]
        if mert: T = r[next(iter(vaes))].shape[1]; r[f"mert_L{args.mert}"] = torch.cat([mert(c, T) for c in x.split(2)])   # 2 frames per latent frame
        torch.cuda.empty_cache()
        return r

    readers = {n: Reader(64, 12).to(dev) for n in vaes}
    for n, a, b in sl: readers[f"{n}[{a}:{b}]"] = Reader(b - a, 12).to(dev)
    if mert: readers[f"mert_L{args.mert}"] = Reader(1024, 25).to(dev)
    opts = {n: torch.optim.AdamW(readers[n].parameters(), lr=args.lr, weight_decay=0.01) for n in names}

    def ctc(lp, tg, lens, ok, cf):
        idx = ok.nonzero().flatten()
        if len(idx) == 0: return None
        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]); tgt = torch.cat([tg[offs[i]:offs[i + 1]] for i in idx.tolist()]).to(dev)
        return F.ctc_loss(lp[idx].transpose(0, 1), tgt, torch.clamp(cf[idx], max=lp.shape[1]).to(dev), lens[idx].to(dev), blank=0, zero_infinity=True)

    import random; random.seed(0)
    vd = Crops(args.val_manifest, holdout=0, seed=12345); vd.rows = [r for r in vd.rows if len(r["lines"]) >= 4]; vit = iter(vd)
    val = [collate([next(vit) for _ in range(16)]) for _ in range(8)]; vin = [(feats(b[0]), b) for b in val]

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

    ds = Crops(args.manifest, holdout=0, seed=3); ds.rows = [r for r in ds.rows if len(r["lines"]) >= 4]
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
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k} {v:.4f}" for k, v in res.items()) + f" {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()


if __name__ == "__main__":
    main()
