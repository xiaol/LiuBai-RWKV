#!/usr/bin/env python3
"""Train the singing ASR encoder (vae/sing_asr.py) with CTC on LRC-anchored crops of the NetEase corpus.

target : romanised letters of the LRC lines sung completely inside the crop (vae/data.Crops), CTC over the frames up to the
         end of the last complete line (ctc_frames, 50 Hz = the encoder's output rate); crops without a complete line are skipped
aug    : SpecAugment, random gain
eval   : greedy letter CER on held-out crops of the test shard (slim_val.jsonl, rows with >= 4 timed lines)
DDP    : start ranks yourself (vae/run_sing_asr.sh; torchrun's rendezvous needs DNS this host lacks); --bs is per GPU
usage  : train_sing_asr.py --out out/asr/s1
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.sing_asr import SingASR, spec_augment  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/asr_train.jsonl"))
    ap.add_argument("--val-manifest", default=str(ROOT / "data/real_corpus/manifest/slim_val.jsonl"))
    ap.add_argument("--seg", type=float, default=16.0); ap.add_argument("--d", type=int, default=768); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--eval-every", type=int, default=5000)
    args = ap.parse_args()
    ddp = "LOCAL_RANK" in os.environ; rank, world = 0, 1
    if ddp:
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True); torch.backends.cuda.matmul.allow_tf32 = True
    if rank == 0: json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    M = SingASR(d=args.d, n=args.n).to(dev)
    opt = torch.optim.AdamW(M.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    step = 0; ckp = out / "last.pt"
    if ckp.exists():
        sd = torch.load(ckp, map_location=dev); M.load_state_dict(sd["M"]); opt.load_state_dict(sd["opt"]); step = sd["step"]
        if rank == 0: print("resumed", step, flush=True)
    ema = {k: v.detach().clone() for k, v in M.state_dict().items()} if not ckp.exists() else torch.load(ckp, map_location=dev)["ema"]
    if rank == 0: print(f"sing ASR {sum(p.numel() for p in M.parameters()) / 1e6:.1f}M, world {world}", flush=True)
    MD = torch.nn.parallel.DistributedDataParallel(M) if ddp else M

    def ctc_loss(lp, tg, lens, ok, cf):
        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]); il = torch.clamp(cf, max=lp.shape[1])
        l = F.ctc_loss(lp.transpose(0, 1), tg.to(dev), il.to(dev), lens.to(dev), blank=0, reduction="none", zero_infinity=True)
        w = ok.to(dev).float(); return (l / lens.to(dev).clamp_min(1) * w).sum() / w.sum().clamp_min(1), w.sum()

    vd = Crops(args.val_manifest, holdout=0, seed=777, seg=args.seg); vd.rows = [r for r in vd.rows if len(r["lines"]) >= 4]; vit = iter(vd)
    val = [] if rank else [collate([next(vit) for _ in range(16)]) for _ in range(8)]

    @torch.no_grad()
    def evaluate():
        live = {k: v.clone() for k, v in M.state_dict().items()}; M.load_state_dict(ema); M.eval(); e = c = 0
        for wav, tg, lens, ok, cf in val:
            with torch.autocast("cuda", dtype=torch.bfloat16): lp = M(wav.to(dev))
            offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
            for j in ok.nonzero().flatten().tolist():
                p = lp[j, :cf[j]].argmax(-1).tolist(); hyp = [t for i, t in enumerate(p) if t != 0 and (i == 0 or t != p[i - 1])]
                ref = tg[offs[j]:offs[j + 1]].tolist(); e += edit(hyp, ref); c += len(ref)
        M.load_state_dict(live); M.train(); return e / max(c, 1)

    ds = Crops(args.manifest, holdout=0, seed=1000 * rank + 31 + step, seg=args.seg)
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, collate_fn=collate, prefetch_factor=4, persistent_workers=True))
    log = open(out / "log.txt", "a") if rank == 0 else None; t0 = time.time(); acc = {}
    while step < args.steps:
        step += 1; wav, tg, lens, ok, cf = next(dl); wav = wav.to(dev, non_blocking=True)
        for g in opt.param_groups: g["lr"] = args.lr * min(1, step / 3000) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        wav = wav * torch.exp(torch.randn(wav.shape[0], 1, 1, device=dev) * 0.4)           # random gain (log-mel shift)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = MD(spec_augment(M.logmel(wav)), is_mel=True)
            loss, n_ok = ctc_loss(lp, tg, lens, ok, cf)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(M.parameters(), 5.0); opt.step()
        with torch.no_grad():
            for k, p in M.state_dict().items():
                if p.dtype.is_floating_point: ema[k].mul_(0.9995).add_(p.detach(), alpha=0.0005)
        for k, v in dict(ctc=loss, ok=n_ok, gn=gn).items(): acc[k] = acc.get(k, 0) + float(v)
        if rank == 0 and step % 100 == 0:
            msg = f"step {step} " + " ".join(f"{k} {v / 100:.4f}" for k, v in acc.items()) + f" lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % 100 == 0: acc = {}
        if rank == 0 and (step % args.eval_every == 0 or step == args.steps):
            c = evaluate(); msg = f"EVAL step {step} val_cer {c:.4f}"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
            torch.save(dict(M=M.state_dict(), opt=opt.state_dict(), ema=ema, step=step, args=vars(args)), out / "last.tmp"); os.replace(out / "last.tmp", ckp)
            if step % 25000 == 0: torch.save(dict(M=ema, step=step, args=vars(args)), out / f"asr_ema_s{step}.pt")
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
