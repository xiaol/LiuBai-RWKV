#!/usr/bin/env python3
"""LiuBai-H whole-song training (user 2026-11-06: keep the 3B, prompt = metadata + lyrics, whole songs, MSVAE).

model : vae/gated_rwkv7.LiuBaiH, full-rank NAR deltas (--lora 0), blk 8 (0.32 s at 25 Hz), text prefix = vae/song_prompt.py
        (RWKV world tokens: title / artist / album / language / live / length / lyrics with or without timestamps).
        Init from an exported LiuBai-H checkpoint (--init-ckpt, default the MSVAE full-rank 20k EMA) incl. its lat_m / lat_s.
data  : tools/precompute_msvae_full.py shards (<data>/r*_*.npz + .meta.jsonl, md5 order, split field), train split only,
        songs with >= --min-lines timed lines. One song per GPU per step: the whole song, or a random --max-sec window of
        longer ones (prompt = the lines starting inside it, times relative to it, Length = window length). Lines cut by a
        window edge carry no letters (crop_letters mask). Letter timing: MMS from the shards, replaced by NetEase YRC word timing (tools/yrc.py) on matched lines of the ~144k songs that have it (--yrc).
loss  : rectified flow on --nar random blocks of the song x --K noisy copies (read-only F rows; the A stream runs over the
        whole song), dims [sem_off, sem_off + sem) x --sem-w; + adv_w x CE of the cursor advance on all blocks.
        CFG: with p --p-null the prompt is null_prompt() and the cursor NULL; otherwise prompt fields drop with --p-field,
        timestamps with --p-ts, the lyric text with --p-lyr-text (the cursor keeps the letters).
eval  : held-out dev songs (first --val-sec s), teacher-forced flow loss with own / swapped (another song's prompt +
        letters) / null conditioning on a fixed block subset and t grid -> gap_swap / gap_null.
ckpt  : <out>/dcp (resumable) and <out>/M_ema_s{step}.pt (bf16 EMA, args, lat_m, lat_s) for tools/gen_song.py.
usage : scripts/run_liubai_song.sh OUT [args]
"""
import argparse, datetime, glob, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.gated_rwkv7 import LiuBaiH  # noqa: E402
from vae.train_liubai_h import crop_letters, load_shard, song, lyric_inputs, TINY  # noqa: E402
from vae.cursor_lm import cursor_arrays, windows, NULL  # noqa: E402
from vae import song_prompt as SP  # noqa: E402

RATE, GRID = 25.0, 8
_YRC = {}                                                                # path of the yrc index -> {id: yrc file}


def yrc_map(path):
    if path and path not in _YRC:
        sys.path.insert(0, str(ROOT / "tools")); import yrc; _YRC[path] = yrc.load_index(path)
    return _YRC.get(path, {})


def shard_files(root):
    """committed shards: the .meta.jsonl is written after the .npz"""
    return [p[:-len(".meta.jsonl")] for p in sorted(glob.glob(os.path.join(root, "r*_*[0-9].meta.jsonl")))]


def prompt_ids(meta, c0, F_, rng, a, null):
    if null: return SP.encode(SP.null_prompt(rng=rng, lang=meta.get("lang")))
    t0, t1 = c0 / RATE, (c0 + F_) / RATE
    lines = [(l["t0"] - t0, l["text"]) for l in meta.get("lines") or [] if t0 <= l["t0"] < t1]
    ids = SP.encode(SP.build(meta, lines, F_ / RATE, rng, a.p_field, a.p_ts, a.p_lyr_text))
    if len(ids) > a.max_tok: tail = SP.encode("\n\n" + SP.THINK); ids = ids[:a.max_tok - len(tail)] + tail      # very long lyrics: cut, keep the turn end
    return ids


def make_song(rng, d, meta, j, a, train=True):
    lat, ids, st, tab = song(d, j); T = len(lat) // GRID * GRID; Fm = int(a.max_sec * RATE) // GRID * GRID
    yp = yrc_map(a.yrc).get(meta["id"]) if a.yrc else None
    if yp and len(tab):                                                  # YRC word timing replaces MMS on the lines it matches
        import yrc
        try: st, _ = yrc.overlay_starts(ids, st, tab, yrc.parse(yp), RATE)
        except Exception: pass
    if T < 4 * GRID: return None
    F_ = min(T, Fm); c0 = int(rng.integers(0, (T - F_) // GRID + 1)) * GRID if train else 0
    null = train and rng.random() < a.p_null
    pr = prompt_ids(meta, c0, F_, random.Random(int(rng.integers(1 << 30))) if train else None, a, null and train)
    return lat[c0:c0 + F_].astype(np.float32), crop_letters(ids, st, tab, c0, F_), np.asarray(pr, np.int64), null


class Songs(torch.utils.data.IterableDataset):
    def __init__(s, root, seed, a, per_visit=16):
        s.root, s.seed, s.a, s.per_visit = root, seed, a, per_visit
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); rk = dist.get_rank() if dist.is_initialized() else 0
        rng = np.random.default_rng([s.seed, rk, wi.id if wi else 0]); n = 0
        while True:
            if n % 50 == 0: files = shard_files(s.root)
            n += 1; f = files[rng.integers(len(files))]; d = load_shard(f + ".npz")
            meta = [json.loads(l) for l in open(f + ".meta.jsonl", encoding="utf-8")]; nl = np.diff(d["line_off"])
            pool = [j for j, m in enumerate(meta) if m.get("split") == "train" and nl[j] >= s.a.min_lines]
            for j in rng.permutation(pool)[:s.per_visit]:
                x = make_song(rng, d, meta[j], int(j), s.a)
                if x is not None: yield x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/models/rwkv-music/data/lat_msvae_full/lyric")
    ap.add_argument("--init-ckpt", default=str(ROOT / "out/liubai_h/msvae_full1/M_ema_s20000.pt"))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--max-sec", type=float, default=300.0); ap.add_argument("--min-lines", type=int, default=4)
    ap.add_argument("--max-tok", type=int, default=1536)
    ap.add_argument("--yrc", default=str(ROOT / "data/real_corpus/yrc/yrc_index.jsonl"), help="YRC word-timing index ('' = MMS only)")
    ap.add_argument("--nar", type=int, default=192, help="blocks per song that get F rows"); ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200000); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=6e-5); ap.add_argument("--lr-new", type=float, default=3e-4); ap.add_argument("--lr-delta", type=float, default=1.5e-4)
    ap.add_argument("--adv-w", type=float, default=0.2); ap.add_argument("--ctx-noise", type=float, default=0.3)
    ap.add_argument("--p-null", type=float, default=0.1); ap.add_argument("--p-field", type=float, default=0.3)
    ap.add_argument("--p-ts", type=float, default=0.5); ap.add_argument("--p-lyr-text", type=float, default=0.2)
    ap.add_argument("--sem-w", type=float, default=3.0); ap.add_argument("--sem", type=int, default=32); ap.add_argument("--sem-off", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=2500); ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--export-every", type=int, default=10000); ap.add_argument("--n-val", type=int, default=32); ap.add_argument("--val-sec", type=float, default=120.0)
    args = ap.parse_args()
    args.arch, args.mode, args.head, args.vae, args.enc, args.rate, args.blk, args.lora = "liubai_h", "cursor", "gated", "msvae", "msvae", RATE, 8, 0

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60)); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr_); dev = torch.device("cuda", lr_); torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out); main_ = rank == 0
    if main_: out.mkdir(parents=True, exist_ok=True); json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    log = open(out / "log.txt", "a") if main_ else None
    def say(m):
        if main_: print(m, flush=True); log.write(m + "\n"); log.flush()

    torch.manual_seed(args.seed)
    tiny = dict(TINY, lora=0) if args.tiny else None
    with torch.device(dev): M = LiuBaiH(**(tiny or dict(lora=0)), blk=8)                    # built on the GPU: no 20 GB host copy per rank
    if args.tiny:
        files = shard_files(args.data); lat = np.concatenate([load_shard(f + ".npz")["lat"][::7].astype(np.float32) for f in files[:4]])
        lat_m, lat_s = torch.from_numpy(lat.mean(0)), torch.from_numpy(lat.std(0)).clamp_min(1e-3)
    else:
        sd = torch.load(args.init_ckpt, map_location="cpu", mmap=True, weights_only=False)
        M.load_state_dict(sd["M"]); lat_m, lat_s = sd["lat_m"].float(), sd["lat_s"].float(); del sd
    lat_m, lat_s = lat_m.to(dev), lat_s.to(dev)
    dw = torch.ones(64, device=dev); S0 = slice(args.sem_off, args.sem_off + args.sem); dw[S0] = args.sem_w; dw = dw / dw.mean()
    for n_, p in M.named_parameters():
        if n_.startswith(("emb.", "head.")) or n_ in ("blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"): p.requires_grad_(False)
    M.set_ckpt(True)
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for b in M.blocks: fully_shard(b, mp_policy=mp)
    fully_shard(M, mp_policy=mp)
    from vae.gated_rwkv7 import FullDelta
    full = {n_ + ".weight" for n_, m in M.named_modules() if isinstance(m, FullDelta)}
    G3B = ROOT / "models/rwkv-g1k-3b-temp"
    pre = set(torch.load(str(G3B / os.listdir(G3B)[0]), map_location="cpu", mmap=True, weights_only=True).keys()) if not args.tiny else set()
    named = [(n_, p) for n_, p in M.named_parameters() if p.requires_grad]
    groups = [dict(params=[p for n_, p in named if n_ in pre], lr=args.lr),
              dict(params=[p for n_, p in named if n_ not in pre and n_ not in full], lr=args.lr_new),
              dict(params=[p for n_, p in named if n_ in full], lr=args.lr_delta)]
    groups = [g for g in groups if g["params"]]
    for g in groups: g["base_lr"] = g["lr"]
    conf_lr = [g["lr"] for g in groups]                                  # a resume restores the saved ones; these win
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.01)
    ema = {n_: p.detach().clone() for n_, p in M.named_parameters()}
    say(f"LiuBai-H song: {sum(p.numel() for p in M.parameters()) / 1e9:.3f}B params, world {world}, 1 song / GPU, max {args.max_sec:.0f} s, "
        f"nar {args.nar} blocks x K {args.K}, shards {len(shard_files(args.data))}, init {args.init_ckpt if not args.tiny else 'tiny'}")

    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    import torch.distributed.checkpoint as dcp
    step = 0; ck = out / "dcp"
    if (ck / ".metadata").exists():
        msd, osd = get_state_dict(M, opt); st = dict(model=msd, opt=osd, ema=ema, step=torch.tensor(0))
        dcp.load(st, checkpoint_id=str(ck)); set_state_dict(M, opt, model_state_dict=st["model"], optim_state_dict=st["opt"])
        for g, b in zip(opt.param_groups, conf_lr): g["base_lr"] = b
        step = int(st["step"]); say(f"resumed at step {step}")

    def save():
        msd, osd = get_state_dict(M, opt); tmp = out / f"dcp.tmp{step}"
        dcp.save(dict(model=msd, opt=osd, ema=ema, step=torch.tensor(step)), checkpoint_id=str(tmp)); dist.barrier()
        if main_:
            import shutil
            if ck.exists(): shutil.rmtree(ck)
            os.replace(tmp, ck)
        dist.barrier()
        torch._C._host_emptyCache()           # dcp stages each rank's model + opt + ema shard (~24 GB) in cached pinned host memory;
                                              # kept, 4 ranks + the precompute ran the host out of RAM (OOM kill at step 3700)

    @torch.no_grad()
    def swap_ema():
        for m in M.modules():
            if isinstance(m, FSDPModule): m.reshard()
        for n_, p in M.named_parameters():
            t = p.detach().clone(); p.detach().copy_(ema[n_]); ema[n_].copy_(t)

    @torch.no_grad()
    def export():
        sd = {}
        for n_, t in ema.items():                 # copy each gathered tensor to host at once: the full 5.1B bf16 model (~10 GB)
            full_ = (t.full_tensor() if hasattr(t, "full_tensor") else t).to(torch.bfloat16)   # on GPU 0 next to training +
            if main_: sd[n_] = full_.cpu()                                                     # precompute OOMed at step 10k
            del full_
        if main_:
            torch.cuda.empty_cache()
            torch.save(dict(M=sd, step=step, args=dict(vars(args), tiny_cfg=tiny), lat_m=lat_m.cpu(), lat_s=lat_s.cpu()), out / f"M_ema_s{step}.pt")
        dist.barrier()

    def inputs(x, lyr_src=None):
        """one song -> z [1,NB,D], fm [1,NB,blk], lyr / a_c / a_t, txt [1,P]; lyr_src = (letters, prompt) of another song (swap)"""
        lat, al, pr, null = x; lat, pr = np.asarray(lat), np.asarray(pr)      # the DataLoader hands back tensors
        al = (np.asarray(al[0]).tolist(), np.asarray(al[1]), np.asarray(al[2]).astype(bool)); NB = len(lat) // 8
        z = ((torch.from_numpy(lat).to(dev) - lat_m) / lat_s).reshape(1, NB, -1)
        fm = torch.from_numpy(al[2]).to(dev).view(1, NB, 8).float()
        if lyr_src is None: lyr, a_c, a_t = lyric_inputs(M, [al], NB, [null])
        else:
            c, a = cursor_arrays(al[1], NB, 8); lyr = torch.from_numpy(windows(lyr_src[0], c, M.wb, M.wf))[None]
            a_t = torch.from_numpy(np.minimum(a, M.max_adv - 1))[None]; a_c = a_t.clone(); pr = lyr_src[1]
        txt = torch.from_numpy(pr)[None].to(dev)
        return z, fm, lyr, a_c, a_t, (txt % M.emb.num_embeddings if args.tiny else txt), NB    # tiny vocab 256

    def flow_err(z_in, z, lyr, a_c, txt, nar, K, gen=None, tg=None):
        G = len(nar) * K; D = z.shape[-1]; zr = z[:, nar].repeat_interleave(K, 1)
        t = torch.sigmoid(torch.randn(1, G, device=dev, generator=gen)) if tg is None else tg.repeat(len(nar))[None]
        n = torch.randn(zr.shape, device=dev, generator=gen); xt = (1 - t[..., None]) * n + t[..., None] * zr
        h, v, _, lg = M.velocity(z_in, lyr.to(dev), a_c.to(dev), xt, t, nar, reps=K, txt=txt)
        return lg, ((v.float() - (zr - n)) ** 2).view(1, G, 8, -1)

    vr = np.random.default_rng(12345); val = []                             # dev songs, identical on every rank
    for f in shard_files(args.data):
        if len(val) >= args.n_val: break
        d = load_shard(f + ".npz"); meta = [json.loads(l) for l in open(f + ".meta.jsonl", encoding="utf-8")]; nl = np.diff(d["line_off"])
        for j, m in enumerate(meta):
            if m.get("split") == "dev" and nl[j] >= args.min_lines and len(val) < args.n_val:
                a2 = argparse.Namespace(**{**vars(args), "max_sec": args.val_sec})
                x = make_song(vr, d, m, j, a2, train=False)
                if x is not None and len(x[0]) >= 8 * 64: val.append(x)
    val = val[:len(val) // world * world]; vmine = list(range(rank, len(val), world))
    say(f"val songs: {len(val)}")

    @torch.no_grad()
    def evaluate():
        swap_ema(); M.eval(); tg = torch.tensor([0.3, 0.5, 0.7, 0.9], device=dev); acc = torch.zeros(7, device=dev)
        for i in vmine:
            other = val[(i + 1) % len(val)]
            for k, src in enumerate((None, (other[1][0], other[2]), "null")):
                if src == "null":
                    z, fm, lyr, a_c, a_t, _, NB = inputs(val[i]); lyr = torch.full_like(lyr, NULL); a_c = torch.full_like(a_c, M.max_adv)
                    txt = torch.tensor([SP.encode(SP.null_prompt())], device=dev) % M.emb.num_embeddings
                else: z, fm, lyr, a_c, a_t, txt, NB = inputs(val[i], src)
                nar = list(range(0, NB, 4))
                _, e2 = flow_err(z, z, lyr, a_c, txt, nar, 4, gen=torch.Generator(device=dev).manual_seed(7), tg=tg)
                w = fm[:, nar].repeat_interleave(4, 1)[..., None].expand_as(e2)
                acc[k] += (e2 * w).sum() / w.sum().clamp_min(1); acc[3 + k] += (e2[..., S0] * w[..., S0]).sum() / w[..., S0].sum().clamp_min(1)
            acc[6] += 1
        dist.all_reduce(acc); M.train(); swap_ema(); a = (acc[:6] / acc[6].clamp_min(1)).tolist()
        return dict(fl_own=a[0], gap_swap=a[1] - a[0], gap_null=a[2] - a[0], sem_own=a[3], sgap_swap=a[4] - a[3], sgap_null=a[5] - a[3])

    dl = iter(torch.utils.data.DataLoader(Songs(args.data, args.seed * 1000 + step, args), batch_size=None, num_workers=args.workers,
                                          prefetch_factor=4, persistent_workers=True))
    rng = np.random.default_rng([args.seed, rank, step]); t0 = time.time(); accl, n_acc = {}, 0; M.train()
    sched = lambda s_: min(1, s_ / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s_, args.steps) / args.steps)))
    while step < args.steps:
        step += 1; x = next(dl); z, fm, lyr, a_c, a_t, txt, NB = inputs(x)
        for g in opt.param_groups: g["lr"] = g["base_lr"] * sched(step)
        nar = sorted(rng.choice(NB, min(NB, args.nar), replace=False).tolist())
        sig = (torch.rand(()) * args.ctx_noise).sqrt().item(); z_ctx = z + sig * torch.randn_like(z)
        lg, e2 = flow_err(z_ctx, z, lyr, a_c, txt, nar, args.K)
        w = fm[:, nar].repeat_interleave(args.K, 1)[..., None]
        lf = ((e2 * dw).mean(-1, keepdim=True) * w).sum() / w.sum().clamp_min(1)
        valid = (fm.min(-1).values > 0) & (not x[3])
        if valid.any(): la = F.cross_entropy(lg.float()[valid], a_t.to(dev)[valid]); aac = (lg[valid].argmax(-1) == a_t.to(dev)[valid]).float().mean()
        else: la = lg.float().sum() * 0; aac = torch.zeros((), device=dev)
        loss = lf + args.adv_w * la
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0); gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
        opt.step()
        dec = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for n_, p in M.named_parameters(): ema[n_].lerp_(p.detach(), 1 - dec)
        for k, v_ in dict(flow=lf.detach(), adv=la.detach(), adv_acc=aac, gn=gn, sec=torch.tensor(NB * 8 / RATE, device=dev),
                          tok=torch.tensor(float(txt.shape[1]), device=dev)).items(): accl[k] = accl.get(k, 0) + v_
        n_acc += 1
        if step % 50 == 0:
            lrs = "/".join("%.1e" % g["lr"] for g in opt.param_groups)
            red = torch.stack([v_ for v_ in accl.values()]).float() / n_acc; dist.all_reduce(red); red /= world
            say(f"step {step} " + " ".join(f"{k} {v_:.4f}" for k, v_ in zip(accl, red.tolist())) +
                f" lr {lrs} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G host {int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 2**20:.1f}G")
            accl, n_acc = {}, 0
        if val and (step % args.eval_every == 0 or step == args.steps):
            say(f"EVAL step {step} " + " ".join(f"{k} {v_:.4f}" for k, v_ in evaluate().items()))
        if step % args.save_every == 0 or step == args.steps: save()
        if step % args.export_every == 0 or step == args.steps: export()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
