#!/usr/bin/env python3
"""Train LiuBai-H (vae/gated_rwkv7.py: g1k-3b + gated read-only NAR rows + lyric cursor) on precomputed VAE v5 latents
(tools/precompute_latents.py shards), FSDP2 over the local GPUs. v1 = lyrics only (cursor letters), no captions / text prefix.

data : 24 s crops cut from whole-song latents on the 8-frame grid (the bounded-lookahead MERT windows of encode_local sit on
       it), 80 % starting just before a random line; letters of the lines sung completely inside the crop with their
       whole-song MMS starts, frames of cut lines masked (= vae/align_letters.py on the crop); songs with >= 4 full lines, plus
       instrumental songs (no full line) drawn with p --p-instr (= train_cursor_lm.py; their natural share in the shards is ~2 %)
loss : as vae/train_cursor_lm.py --head gated: rectified flow on K noisy copies of every block (read-only F rows), MERT dims
       x --sem-w, context noise U(0, --ctx-noise), lyrics dropped with p --p-lyr (CFG), + adv_w x CE of the cursor advance
optim: AdamW, fp32 master params sharded (FSDP2, bf16 compute, fp32 grad reduce), pretrained weights at --lr, new modules at
       --lr-new; emb / head frozen (no text rows in v1); EMA of all params
eval : teacher-forced own / swap / null flow loss on held-out songs (fixed crops, t grid, noise) -> gap_swap / gap_null
ckpt : <out>/dcp (torch.distributed.checkpoint: model + opt + ema + step, resumable), <out>/M_ema_s{step}.pt (bf16 full EMA,
       vae/eval_cursor_lm.py format: M, args, lat_m, lat_s)
latent rate: read from <data>/meta.json ("rate": 12.5 for v5, 25 for MSVAE shards of tools/precompute_msvae.py, use --blk 8 there
       so a block stays 0.32 s); --vae is then recorded as "msvae" for vae/eval_cursor_lm.py
NAR deltas: --lora r > 0 = LoRA, --lora 0 = full-rank delta matrices (vae/gated_rwkv7.FullDelta) trained at --lr-delta
usage: scripts/run_liubai_h.sh (starts one process per GPU with env:// init; torchrun cannot resolve this host)
"""
import argparse, glob, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.gated_rwkv7 import LiuBaiH  # noqa: E402
from vae.cursor_lm import cursor_arrays, windows, NULL  # noqa: E402
from vae.align_words import SPACE  # noqa: E402

G3B = ROOT / "models/rwkv-g1k-3b-temp"
TINY = dict(n_layer=4, C=256, vocab=256, dims=(32, 32, 32, 64), lora=16, dc=64, da=128)


def crop_letters(ids, st, tab, c0, F_):
    """song letters + line table (precompute_latents.song_lines) -> LetterAligner output for frames [c0, c0 + F_)"""
    full = [t for t in tab if t[0] >= c0 and t[2] <= c0 + F_]
    mask = np.ones(F_, bool)
    for t in tab:                                                     # cut lines: their frames carry no letters
        if t[2] > c0 and t[0] < c0 + F_ and not (t[0] >= c0 and t[2] <= c0 + F_):
            mask[max(0, t[0] - c0):max(0, min(F_, t[2] - c0))] = False
    out_i, out_s = [], []
    for t in full:
        if out_i: out_i.append(SPACE); out_s.append(np.nan)            # separator, timed by the next letter
        out_i += ids[t[3]:t[4]].tolist(); out_s += (st[t[3]:t[4]] - c0).tolist()
    s = np.asarray(out_s, np.float64); nxt = np.nan
    for i in range(len(s) - 1, -1, -1):
        if np.isnan(s[i]): s[i] = nxt
        else: nxt = s[i]
    s = np.where(np.isnan(s), 0.0, s)
    return out_i, (np.maximum.accumulate(np.clip(s, 0, F_ - 1e-3)) if len(s) else s), mask


def load_shard(p):
    d = np.load(p); return {k: d[k] for k in d.files}


def song(d, j):
    a, b = d["lat_off"][j], d["lat_off"][j + 1]; i0, i1 = d["id_off"][j], d["id_off"][j + 1]; l0, l1 = d["line_off"][j], d["line_off"][j + 1]
    return d["lat"][a:b], d["ids"][i0:i1], d["st"][i0:i1], d["lines"][l0:l1]


def make_crop(rng, d, j, F_, p_line=0.8, grid=8):
    lat, ids, st, tab = song(d, j); T = len(lat)
    if T < F_: return None
    hi = (T - F_) // grid * grid
    if len(tab) and rng.random() < p_line: c0 = int(tab[rng.integers(len(tab)), 0]) - int(rng.integers(1, 16))
    else: c0 = int(rng.integers(0, hi + 1))
    c0 = min(max(0, c0 // grid * grid), hi)
    return lat[c0:c0 + F_].astype(np.float32), crop_letters(ids, st, tab, c0, F_)


class Shards(torch.utils.data.IterableDataset):
    """random shard -> --per-visit random songs of it (instrumental with p p_instr) -> one crop each; shard list re-scanned
    (precompute may still run)"""
    def __init__(s, root, F_, seed, per_visit=64, min_lines=4, p_instr=0.1):
        s.root, s.F, s.seed, s.per_visit, s.min_lines, s.p_instr = root, F_, seed, per_visit, min_lines, p_instr
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); rk = dist.get_rank() if dist.is_initialized() else 0
        rng = np.random.default_rng([s.seed, rk, wi.id if wi else 0]); n = 0
        while True:
            if n % 50 == 0: files = sorted(glob.glob(os.path.join(s.root, "r*_*[0-9].npz")))
            n += 1; d = load_shard(files[rng.integers(len(files))]); nl = np.diff(d["line_off"])
            voc, ins = np.flatnonzero(nl >= s.min_lines), np.flatnonzero(nl == 0)
            for _ in range(s.per_visit):
                pool = ins if len(ins) and rng.random() < s.p_instr else voc
                if not len(pool): continue
                c = make_crop(rng, d, int(pool[rng.integers(len(pool))]), s.F)
                if c is not None: yield c


def collate(b): return torch.from_numpy(np.stack([x[0] for x in b])), [x[1] for x in b]


def lyric_inputs(M, al, NB, drop, roll=0):
    """= vae/train_cursor_lm.lyric_inputs, cursor mode"""
    B = len(al); a_t = np.zeros((B, NB), np.int64); lyr = np.zeros((B, NB, M.wb + M.wf), np.int64); a_c = np.zeros((B, NB), np.int64)
    for b in range(B):
        c, a = cursor_arrays(al[b][1], NB, M.blk); a_t[b] = np.minimum(a, M.max_adv - 1)
        lyr[b] = NULL if drop[b] else windows(al[(b + roll) % B][0], c, M.wb, M.wf)
        a_c[b] = M.max_adv if drop[b] else a_t[b]
    return torch.from_numpy(lyr), torch.from_numpy(a_c), torch.from_numpy(a_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/models/rwkv-music/data/lat_v5/pilot_clm_train_clean")
    ap.add_argument("--val-data", default="/models/rwkv-music/data/lat_v5/slim_val_clean")
    ap.add_argument("--vae", default=str(ROOT / "out/vae/v5/G_s80000.pt"), help="recorded for eval (latents are precomputed with it)")
    ap.add_argument("--init", default="g1k-3b", help="g1k-3b | none (random init, e.g. with --tiny)")
    ap.add_argument("--tiny", action="store_true", help="4 x 256 random model (pipeline test)")
    ap.add_argument("--seg", type=float, default=24.0); ap.add_argument("--blk", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--bs", type=int, default=4, help="per GPU")
    ap.add_argument("--lr", type=float, default=6e-5); ap.add_argument("--lr-new", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--adv-w", type=float, default=0.2)
    ap.add_argument("--ctx-noise", type=float, default=0.3); ap.add_argument("--p-lyr", type=float, default=0.1)
    ap.add_argument("--p-instr", type=float, default=0.1, help="share of instrumental crops (songs without a full timed line)")
    ap.add_argument("--sem-w", type=float, default=3.0); ap.add_argument("--sem", type=int, default=32)
    ap.add_argument("--sem-off", type=int, default=0, help="weighted dims = [sem_off, sem_off + sem): 0 for v5 (MERT half), 32 for MSVAE (linguistic half)")
    ap.add_argument("--lora", type=int, default=64, help="0 = full-rank NAR deltas");
    ap.add_argument("--lr-delta", type=float, default=0.0, help="lr of full-rank NAR deltas (default: --lr)");
    ap.add_argument("--rewarm", type=int, default=500, help="on resume: groups whose lr went up ramp to it over this many steps"); ap.add_argument("--lattn", action="store_true"); ap.add_argument("--iso", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=2500); ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--export-every", type=int, default=10000); ap.add_argument("--n-val", type=int, default=32)
    args = ap.parse_args()
    # fields vae/eval_cursor_lm.py reads from ckpt["args"]
    args.arch, args.mode, args.head, args.enc, args.enc_keep, args.enc_win, args.enc_ahead = "liubai_h", "cursor", "gated", "local", 2, 3.0, 0.64

    import datetime   # rank-0-only file work (dcp rmtree, export torch.save on NFS) sits before barriers: 10 min default is tight
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60)); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr_); dev = torch.device("cuda", lr_); torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out); main_ = rank == 0
    if main_: out.mkdir(parents=True, exist_ok=True); json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    log = open(out / "log.txt", "a") if main_ else None
    def say(m):
        if main_: print(m, flush=True); log.write(m + "\n"); log.flush()
    meta = json.load(open(Path(args.data) / "meta.json")); rate = float(meta.get("rate", 12.5))
    if meta.get("vae") == "msvae": args.vae, args.enc = "msvae", "msvae"
    args.rate = rate; F_ = int(round(args.seg * rate)) // 8 * 8; NB = F_ // args.blk; assert F_ % args.blk == 0

    # latent stats: same 6 shards on every rank (deterministic)
    files = sorted(glob.glob(os.path.join(args.data, "r*_*[0-9].npz"))); assert files, args.data
    lat = np.concatenate([load_shard(p)["lat"][::7].astype(np.float32) for p in files[:6]])
    lat_m = torch.from_numpy(lat.mean(0)).to(dev); lat_s = torch.from_numpy(lat.std(0)).clamp_min(1e-3).to(dev); del lat
    dw = torch.ones(64, device=dev); S0 = slice(args.sem_off, args.sem_off + args.sem); dw[S0] = args.sem_w; dw = dw / dw.mean()

    torch.manual_seed(args.seed)
    tiny = dict(TINY, lora=args.lora if args.lora in (0, 16) else 16) if args.tiny else None
    M = LiuBaiH(**(tiny or dict(lora=args.lora)), lattn=args.lattn, iso=args.iso, blk=args.blk)
    if args.init == "g1k-3b" and not args.tiny: M.load_pretrained(str(G3B / os.listdir(G3B)[0]))
    pre = set(torch.load(str(G3B / os.listdir(G3B)[0]), map_location="cpu", mmap=True, weights_only=True).keys()) if not args.tiny else set()
    for n_, p in M.named_parameters():
        if n_.startswith(("emb.", "head.")): p.requires_grad_(False)        # v1: no text rows
        if n_ in ("blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"): p.requires_grad_(False)   # unused (layer 0 sets v_first);
        # no grad -> no AdamW state -> the dcp checkpoint lacks them and resume fails
    M.set_ckpt(True); M = M.to(dev)
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for b in M.blocks: fully_shard(b, mp_policy=mp)
    fully_shard(M, mp_policy=mp)
    named = [(n_, p) for n_, p in M.named_parameters() if p.requires_grad]
    from vae.gated_rwkv7 import FullDelta
    full = {n_ + ".weight" for n_, m in M.named_modules() if isinstance(m, FullDelta)}       # FSDP keeps the module names
    groups = [dict(params=[p for n_, p in named if n_ in pre], lr=args.lr),
              dict(params=[p for n_, p in named if n_ not in pre and n_ not in full], lr=args.lr_new),
              dict(params=[p for n_, p in named if n_ in full], lr=args.lr_delta or args.lr)]
    for g in groups: g["base_lr"] = g["lr"]
    conf_lr = [g["lr"] for g in groups if g["params"]]                 # the optimizer state of a resume would restore the old ones
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.99), weight_decay=0.01)
    ema = {n_: p.detach().clone() for n_, p in M.named_parameters()}
    n_all = sum(p.numel() for p in M.parameters()); n_tr = sum(p.numel() for _, p in named); n_new = sum(p.numel() for n_, p in named if n_ not in pre)
    say(f"LiuBai-H {n_all / 1e9:.3f}B params, trainable {n_tr / 1e9:.3f}B (new {n_new / 1e6:.0f}M), world {world}, bs {args.bs}/GPU, "
        f"full-rank deltas {sum(p.numel() for n_, p in named if n_ in full) / 1e9:.3f}B, rate {rate} Hz, crop {F_} frames = {NB} blocks x K {args.K}, train shards {len(files)}")

    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, StateDictOptions
    import torch.distributed.checkpoint as dcp
    sched = lambda st_: min(1, st_ / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(st_, args.steps) / args.steps)))
    step = 0; ck = out / "dcp"
    if (ck / ".metadata").exists():
        msd, osd = get_state_dict(M, opt); st = dict(model=msd, opt=osd, ema=ema, step=torch.tensor(0))
        dcp.load(st, checkpoint_id=str(ck)); set_state_dict(M, opt, model_state_dict=st["model"], optim_state_dict=st["opt"])
        step = int(st["step"]); say(f"resumed at step {step}")
        for og, b in zip(opt.param_groups, conf_lr):     # ramp from the last lr actually used to the new schedule (raised base
            og["rewarm_from"] = min(1.0, og["lr"] / (b * sched(step)))   # lr, or --steps extended past the old cosine end)
            say(f"lr group: last {og['lr']:.1e}, scheduled now {b * sched(step):.1e}, ramp over {args.rewarm} steps")
            if og["base_lr"] != b: say(f"lr group of {sum(p.numel() for p in og['params']) / 1e9:.3f}B: base lr {og['base_lr']:.1e} -> {b:.1e}")
            og["base_lr"] = b
    step0 = step

    def save():
        msd, osd = get_state_dict(M, opt); tmp = out / f"dcp.tmp{step}"
        dcp.save(dict(model=msd, opt=osd, ema=ema, step=torch.tensor(step)), checkpoint_id=str(tmp)); dist.barrier()
        if main_:
            import shutil
            if ck.exists(): shutil.rmtree(ck)
            os.replace(tmp, ck)
        dist.barrier()

    @torch.no_grad()
    def swap_ema():                                                    # params <-> ema (in place, sharded)
        for m in M.modules():                                          # a no-grad forward leaves the root unsharded (no backward
            if isinstance(m, FSDPModule): m.reshard()                  # to reshard it): params would be plain tensors
        for n_, p in M.named_parameters():
            t = p.detach().clone(); p.detach().copy_(ema[n_]); ema[n_].copy_(t)

    @torch.no_grad()
    def export():
        # gather the EMA shard by shard straight into bf16 on the GPU (all ranks, back-to-back collectives, ~6 GB on rank 0),
        # then copy to CPU once. The old path (get_model_state_dict full + cpu_offload) did an fp32 CPU copy between
        # all-gathers on rank 0; under host memory pressure that stalled it > 10 min and NCCL aborted the job (prod1 at 10k).
        sd = {}
        for n_, t in ema.items():
            full = (t.full_tensor() if hasattr(t, "full_tensor") else t).to(torch.bfloat16)
            if main_: sd[n_] = full
            else: del full
        if main_:
            sd = {k: v.cpu() for k, v in sd.items()}; torch.cuda.empty_cache()
            torch.save(dict(M=sd, step=step,
                            args=dict(vars(args), tiny_cfg=tiny), lat_m=lat_m.cpu(), lat_s=lat_s.cpu()), out / f"M_ema_s{step}.pt")
        dist.barrier()

    def flow_err(z_in, lyr, a_c, z, gen=None, tgrid=None):
        B = z.shape[0]; K = args.K; D = z.shape[-1]; zr = z[:, :, None].expand(B, NB, K, D)
        t = torch.sigmoid(torch.randn(B, NB, K, device=dev, generator=gen)) if tgrid is None else tgrid.view(1, 1, K).expand(B, NB, K)
        n = torch.randn(zr.shape, device=dev, generator=gen); xt = (1 - t[..., None]) * n + t[..., None] * zr
        h, v, _, lg = M.velocity(z_in, lyr.to(dev), a_c.to(dev), xt.reshape(B, NB * K, D), t.reshape(B, NB * K), range(NB), reps=K)
        return lg, ((v.view(B, NB, K, D).float() - (zr - n)) ** 2).view(B, NB, K, args.blk, -1)

    def masked(err, fm):
        w = fm[:, :, None].expand_as(err); return (err * w).sum() / w.sum().clamp_min(1)

    def prep(lat, al):
        z = ((lat.to(dev) - lat_m) / lat_s).reshape(lat.shape[0], NB, -1)
        fm = torch.from_numpy(np.stack([m for _, _, m in al])).to(dev).view(-1, NB, args.blk).float()
        return z, fm

    # held-out crops: identical on every rank, each rank evaluates its slice
    vfiles = sorted(glob.glob(os.path.join(args.val_data, "r*_*[0-9].npz"))); val = []
    if vfiles:
        vr = np.random.default_rng(12345); vd = [load_shard(p) for p in vfiles]
        cand = [(i, j) for i, d in enumerate(vd) for j in range(len(d["row"])) if np.diff(d["line_off"])[j] >= 4]
        for k in vr.permutation(len(cand))[:args.n_val]:
            c = make_crop(vr, vd[cand[k][0]], cand[k][1], F_)
            if c is not None: val.append(c)
        val = val[:len(val) // world * world][rank::world]; del vd                # equal share per rank (FSDP: same forwards)
    say(f"val crops: {len(val) * world} ({len(vfiles)} val shards)")

    @torch.no_grad()
    def evaluate():
        swap_ema(); M.eval(); tgrid = torch.tensor([0.3, 0.5, 0.7, 0.9], device=dev)[:args.K] if args.K <= 4 else None
        acc = torch.zeros(7, device=dev)
        for i in range(0, len(val), args.bs):
            lat, al = collate(val[i:i + args.bs]); z, fm = prep(lat, al); B = z.shape[0]
            for k, (roll, drop) in enumerate(((0, False), (1, False), (0, True))):
                lyr, a_c, a_t = lyric_inputs(M, al, NB, [drop] * B, roll)
                lg, e2 = flow_err(z, lyr, a_c, z, gen=torch.Generator(device=dev).manual_seed(7), tgrid=tgrid)
                acc[k] += masked(e2.mean(-1), fm) * B; acc[3 + k] += masked(e2[..., S0].mean(-1), fm) * B
            acc[6] += B
        dist.all_reduce(acc); M.train(); swap_ema(); a = (acc[:6] / acc[6]).tolist()
        return dict(fl_own=a[0], gap_swap=a[1] - a[0], gap_null=a[2] - a[0], sem_own=a[3], sgap_swap=a[4] - a[3], sgap_null=a[5] - a[3])

    dl = iter(torch.utils.data.DataLoader(Shards(args.data, F_, args.seed * 1000 + step, p_instr=args.p_instr), batch_size=args.bs, num_workers=args.workers,
                                          collate_fn=collate, prefetch_factor=4, persistent_workers=True))
    rng = np.random.default_rng([args.seed, rank, step]); t0 = time.time(); accl, n_acc = {}, 0; M.train()
    while step < args.steps:
        step += 1; lat, al = next(dl); z, fm = prep(lat, al); B = z.shape[0]
        f = sched(step)
        for g in opt.param_groups:
            r0 = g.get("rewarm_from", 1.0); r = r0 + (1 - r0) * min(1, (step - step0) / max(1, args.rewarm))
            g["lr"] = g["base_lr"] * f * r
        drop = (rng.random(B) < args.p_lyr).tolist(); lyr, a_c, a_t = lyric_inputs(M, al, NB, drop)
        sig = (torch.rand(B, 1, 1, device=dev) * args.ctx_noise).sqrt(); z_ctx = z + sig * torch.randn_like(z)
        lg, e2 = flow_err(z_ctx, lyr, a_c, z); lf = masked((e2 * dw).mean(-1), fm)
        valid = (fm.min(-1).values > 0) & ~torch.tensor(drop, device=dev)[:, None]
        if valid.any(): la = F.cross_entropy(lg.float()[valid], a_t.to(dev)[valid]); aac = (lg[valid].argmax(-1) == a_t.to(dev)[valid]).float().mean()
        else: la = lg.float().sum() * 0; aac = torch.zeros((), device=dev)
        loss = lf + args.adv_w * la
        opt.zero_grad(set_to_none=True); loss.backward()
        if step == 1: assert all(p.grad is not None for _, p in named), [n_ for n_, p in named if p.grad is None][:8]
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0); gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
        opt.step()
        dec = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for n_, p in M.named_parameters(): ema[n_].lerp_(p.detach(), 1 - dec)
        for k, v_ in dict(flow=lf.detach(), adv=la.detach(), adv_acc=aac, gn=gn).items(): accl[k] = accl.get(k, 0) + v_
        n_acc += 1
        if step % 50 == 0:
            red = torch.stack([v_ for v_ in accl.values()]).float() / n_acc; dist.all_reduce(red); red /= world
            say(f"step {step} " + " ".join(f"{k} {v_:.4f}" for k, v_ in zip(accl, red.tolist())) +
                f" lr {'/'.join(f"{g['lr']:.1e}" for g in opt.param_groups)} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"); accl, n_acc = {}, 0
        if val and (step % args.eval_every == 0 or step == args.steps):
            say(f"EVAL step {step} " + " ".join(f"{k} {v_:.4f}" for k, v_ in evaluate().items()))
        if step % args.save_every == 0 or step == args.steps: save()
        if step % args.export_every == 0 or step == args.steps: export()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
