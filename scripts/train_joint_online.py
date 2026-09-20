#!/usr/bin/env python3
"""Joint R1 head + R2 NAR-LoRA training with YuE2's flow-matching teacher, with ONLINE MERT features (PLAN §8.6).

Same objective as train_joint_teacher.py (real window -> head -> straight-through codec embeddings -> frozen YuE2 AR+NAR
(+LoRA) -> flow MSE vs the song's true VAE latents; minted soft-CE anchor; minted flow windows with true tokens), but the
real-audio pool is no longer limited to tracks with 38 MB of stored MERT features:
  * stored real dirs   data/yue2_real/<shard>/<id>/{mert_L_12_16_20_23.npy, latent_true.npy, prefix.npy}   (as before)
  * G1 shards          data/yue2_corpus/suno94k/shard-NNNNN.{latents.npz,jsonl}[,audio/<id>.opus]
                       features are computed on the fly by a producer thread: opus audio if present (preferred: MERT on
                       real audio), else VAE.decode(latent) (≈ 11 % exact-token agreement with real-audio features, see §9)
Held-out eval stays on stored-feature real tracks (8, md5 order) + minted val, so numbers are comparable with out/joint/v2.
usage: train_joint_online.py --out out/joint/v3 --head-init out/joint/v2/head_best.pt --lora-init out/joint/v2/lora_best.pt
         --latent-shards 'data/yue2_corpus/suno94k/shard-*.latents.npz' [--wins-per-song 2] [--rescan-every 500] ...
"""
import argparse, glob, hashlib, json, math, random, sys, threading, queue, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
YUE2 = ROOT / "models/yue2"
sys.path.insert(0, str(ROOT / "scripts"))
from train_inverse_tokenizer import InverseTokenizer, Track  # noqa: E402
from online_mert import OnlineMERT, MemTrack  # noqa: E402
VOCAB = 32768
FEAT = "mert_L_12_16_20_23.npy"


class LoRALinear(nn.Module):
    def __init__(s, base, r):
        super().__init__(); s.base = base
        s.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        s.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(s, x): return s.base(x) + ((x.float() @ s.A.T) @ s.B.T).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--head-init", required=True)
    ap.add_argument("--train-head", type=int, default=1); ap.add_argument("--train-lora", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lora-init", default="none")
    ap.add_argument("--win", type=int, default=256); ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--real-dirs", nargs="*", default=[str(ROOT / "data/yue2_real/shard-00000"), str(ROOT / "data/yue2_real/shard-00001")])
    ap.add_argument("--latent-shards", default=str(ROOT / "data/yue2_corpus/suno94k/shard-*.latents.npz"), help="glob of G1 latent shards ('' = none)")
    ap.add_argument("--exclude-shards", default="2", help="G1 shard numbers never used for training (test material)")
    ap.add_argument("--min-sec", type=float, default=30); ap.add_argument("--max-sec", type=float, default=300)
    ap.add_argument("--require-opus", type=int, default=1, help="1 = only use G1 songs with stored opus audio (MERT on real audio); 0 = also VAE-decode songs without audio")
    ap.add_argument("--online-p", type=float, default=-1, help="prob. that a real step uses an online (G1) song; -1 = proportional to pool sizes")
    ap.add_argument("--wins-per-song", type=int, default=2, help="training windows (= optimizer steps) per online song")
    ap.add_argument("--rescan-every", type=int, default=500, help="re-glob the G1 shards every N steps (G1 is still running)")
    ap.add_argument("--minted-dirs", nargs="+", default=[str(ROOT / "data/yue2_minted/tracks"), str(ROOT / "data/yue2_minted_own")])
    ap.add_argument("--max-minted", type=int, default=0)
    ap.add_argument("--lr-head", type=float, default=3e-5); ap.add_argument("--lr-lora", type=float, default=3e-5); ap.add_argument("--lr-io", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ce-weight", type=float, default=1.0); ap.add_argument("--ce-bs", type=int, default=4)
    ap.add_argument("--minted-flow-p", type=float, default=0.25); ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--hold", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = True
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.protocol import CODEC_OFFSET, MUSIC_END, SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from yue2.nar import attention as nar_attention

    # ---- YuE2 (frozen) + LoRA on the NAR branch ----
    snap = str(YUE2 / "YuE2-3B")
    model = YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(dev)
    model.requires_grad_(False); bb = model.model; tok = YuE2TextTokenizer(snap + "/qwen.tiktoken")
    Ecodec = bb.embed_tokens.weight[CODEC_OFFSET:CODEC_OFFSET + VOCAB]
    lora_params = []
    for layer in bb.layers:
        for mod, names in ((layer.nar_self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.nar_mlp, ("gate_proj", "up_proj", "down_proj"))):
            for n in names: l = LoRALinear(getattr(mod, n), args.rank); setattr(mod, n, l); lora_params += [l.A, l.B]
    model.vae2llm.float(); model.llm2vae.float(); io_params = list(model.vae2llm.parameters()) + list(model.llm2vae.parameters())
    def load_lora(path):
        ck = torch.load(path, map_location=dev)
        with torch.no_grad():
            for p, v in zip(lora_params, ck["lora"]): p.copy_(v.to(dev))
            model.vae2llm.load_state_dict({k: v.float() for k, v in ck["io"]["vae2llm"].items()}); model.llm2vae.load_state_dict({k: v.float() for k, v in ck["io"]["llm2vae"].items()})
    def save_lora(path): torch.save({"lora": [p.detach().cpu() for p in lora_params], "io": {"vae2llm": model.vae2llm.state_dict(), "llm2vae": model.llm2vae.state_dict()}, "rank": args.rank}, path)
    if args.lora_init != "none": load_lora(args.lora_init); print("loaded NAR LoRA", args.lora_init, flush=True)
    for p in lora_params + io_params: p.requires_grad_(bool(args.train_lora))

    # ---- head ----
    hck = torch.load(args.head_init, map_location=dev); head = InverseTokenizer(**hck["cfg"]).to(dev); head.load_state_dict(hck["model"])
    layers = hck["layers"]; assert layers == [12, 16, 20, 23], layers
    head.requires_grad_(bool(args.train_head)); head.train(bool(args.train_head))
    groups = []
    if args.train_head: groups.append({"params": list(head.parameters()), "lr": args.lr_head, "weight_decay": 0.05})
    if args.train_lora: groups += [{"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.0}, {"params": io_params, "lr": args.lr_io, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95)); base_lrs = [g["lr"] for g in opt.param_groups]
    NB = torch.tensor(np.load(ROOT / "data/yue2_minted/codec_nbr_idx.npy").astype(np.int64), device=dev)
    NW = torch.softmax(torch.tensor(np.load(ROOT / "data/yue2_minted/codec_nbr_cos.npy"), device=dev) / 0.05, dim=1)

    # ---- stored-feature data (as train_joint_teacher.py) ----
    def load_real(d):
        z = np.load(d / "latent_true.npy").astype(np.float32); tr = Track(d / FEAT, None, len(z))
        return dict(name=d.name, tr=tr, lat=z[:tr.n], prefix=[int(v) for v in np.load(d / "prefix.npy")])
    real_dirs = sorted(d for rd in args.real_dirs for d in Path(rd).iterdir() if (d / "prefix.npy").exists() and (d / FEAT).exists())
    real_dirs.sort(key=lambda d: hashlib.md5(d.name.encode()).hexdigest()); hold_dirs, real_dirs = real_dirs[:args.hold], real_dirs[args.hold:]
    real = [load_real(d) for d in real_dirs if len(np.load(d / "latent_true.npy", mmap_mode="r")) >= args.win]; hold = [load_real(d) for d in hold_dirs]
    def load_minted(d):
        y = np.load(d / "semantic.npy").astype(np.int64); tr = Track(d / FEAT, y, len(y))
        r = json.load(open(d / "request.json")) if (d / "request.json").exists() else {}
        it = dict(name=d.name, tr=tr, tok=tr.y, style=r.get("style", ""), lyrics=r.get("lyrics", ""))
        if (d / "latent.npy").exists(): it["lat_path"] = d / "latent.npy"
        return it
    minted_dirs = sorted(d for md in args.minted_dirs for d in Path(md).iterdir() if (d / FEAT).exists() and (d / "semantic.npy").exists())
    if args.max_minted: minted_dirs = minted_dirs[:args.max_minted]
    mheld = lambda d: int(hashlib.md5(d.name.encode()).hexdigest(), 16) % 20 == 0
    mtrain = [load_minted(d) for d in minted_dirs if not mheld(d)]; mval = [load_minted(d) for d in minted_dirs if mheld(d)]
    mflow = [m for m in mtrain if "lat_path" in m]

    # ---- online (G1) pool: song metadata only; features computed by the producer thread ----
    excl = {int(x) for x in args.exclude_shards.split(",") if x}
    online = {}                                    # id -> dict(npz, caption, lyrics, n, opus)
    def rescan():
        n_new = 0
        for npz in sorted(glob.glob(args.latent_shards)) if args.latent_shards else []:
            tag = Path(npz).name.split(".")[0]; shard = int(tag.split("-")[1]); jl = Path(npz).with_name(f"{tag}.jsonl")
            if shard in excl or not jl.exists(): continue
            adir = Path(npz).with_name(f"{tag}.audio")
            for line in open(jl):
                r = json.loads(line)
                if r.get("error") or r["id"] in online or "n_frames" not in r: continue
                sec = r["n_frames"] / 25; lyr = r.get("lyrics", "") or ""
                if not (args.min_sec <= sec <= args.max_sec) or "[instrumental]" in lyr.lower() or len(lyr) < 50: continue
                op = adir / f"{r['id']}.opus"
                if args.require_opus and not op.exists(): continue
                online[r["id"]] = dict(npz=npz, caption=r.get("caption", ""), lyrics=lyr, n=int(r["n_frames"]), opus=op if op.exists() else None); n_new += 1
        return n_new
    rescan(); print(f"real stored {len(real)} held {len(hold)} | online {len(online)} ({sum(1 for v in online.values() if v['opus'])} with opus) | minted train {len(mtrain)} val {len(mval)} | win {args.win} rank {args.rank}", flush=True)
    om = OnlineMERT(dev=dev)
    npz_cache = {}
    def prefix_of(caption, lyrics, sid):
        style = " ".join(caption.split())[:1500] or "pop"; lyrics = lyrics.strip() or "[instrumental]"
        return token_prefixes(SongRequest(style=style, lyrics=lyrics, cot="off", seed=1, id=sid), tok)
    def build_online(sid):
        it = online[sid]
        if it["npz"] not in npz_cache: npz_cache[it["npz"]] = np.load(it["npz"])
        z = npz_cache[it["npz"]][sid].astype(np.float32)
        if it["opus"] is not None:
            a, sr = sf.read(it["opus"], dtype="float32")
            if a.ndim > 1: a = a.mean(1)
            if sr != 24000: g = gcd(sr, 24000); a = resample_poly(a, 24000 // g, sr // g).astype(np.float32)
            X = om.feats_from_mono24(a, T25=len(z))
        else:
            X = om.feats_from_latent(z)
        n = min(len(z), X.shape[1]); tr = MemTrack(X[:, :n], None, n)
        return dict(name=sid, tr=tr, lat=z[:n], prefix=prefix_of(it["caption"], it["lyrics"], sid), online=True)
    q = queue.Queue(maxsize=3); stop = threading.Event()
    def producer():
        s = torch.cuda.Stream()
        while not stop.is_set():
            if not online: time.sleep(5); continue
            sid = random.choice(list(online.keys()))
            try:
                with torch.cuda.stream(s): it = build_online(sid)
                s.synchronize(); q.put(it)
            except Exception as e:
                print(f"  online FAIL {sid}: {str(e)[:120]}", flush=True); online.pop(sid, None)
    threading.Thread(target=producer, daemon=True).start()

    # ---- YuE2 forward pieces ----
    def ar_layer(layer, x, cos_, sin_):
        q_, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos_, sin_); h = nar_attention(q_[0], k[0], v[0], causal=True)
        x = x + layer.self_attn.o_proj(h.flatten(1)[None]); return x + layer.mlp(layer.post_attention_layernorm(x)), k[0], v[0]
    def nar_layer(layer, h, ak, av, ncos, nsin):
        q_, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(h), ncos, nsin); a = nar_attention(q_[0], torch.cat((ak, k[0])), torch.cat((av, v[0])))
        h = h + layer.nar_self_attn.o_proj(a.flatten(1)[None]); return h + layer.nar_mlp(layer.nar_pre_mlp_layernorm(h))
    def flow_loss(prefix, codec_emb, x1, t, noise, grad_ar, grad_nar):
        pre = bb.embed_tokens(torch.tensor([prefix], device=dev))[0]; end = bb.embed_tokens(torch.tensor([MUSIC_END], device=dev))
        x = torch.cat((pre, codec_emb.to(pre.dtype), end), 0)[None]; Lq = x.shape[1]; cos_, sin_ = bb.rotary_emb(torch.arange(Lq, device=dev)[None]); cache = []
        if grad_ar:
            for layer in bb.layers: x, k, v = checkpoint(ar_layer, layer, x, cos_, sin_, use_reentrant=False); cache.append((k, v))
        else:
            with torch.no_grad():
                for layer in bb.layers: x, k, v = ar_layer(layer, x, cos_, sin_); cache.append((k, v))
        T = codec_emb.shape[0]; xt = t * noise + (1 - t) * x1; target = noise - x1; N = T + 2
        ncos, nsin = bb.rotary_emb(torch.arange(Lq, Lq + N, device=dev)[None])
        pe = model.latent_pos_embed(torch.arange(N, device=dev).clamp(max=model.config.max_latent_frames - 1))[None]
        sh = model._shift_t_value(float(np.clip(np.log(t / (1 - t)), -20, 20)), dev, torch.bfloat16)
        h = model.vae2llm(F.pad(xt, (0, 0, 1, 1))[None].float()).to(torch.bfloat16) + model.time_embedder(sh.expand(N))[None] + pe
        for layer, (ak, av) in zip(bb.layers, cache):
            h = checkpoint(nar_layer, layer, h, ak, av, ncos, nsin, use_reentrant=False) if (grad_ar or grad_nar) else nar_layer(layer, h, ak, av, ncos, nsin)
        return F.mse_loss(model.llm2vae(bb.norm(h)[0, 1:-1].float()), target)
    def st_embed(logits):
        p = torch.softmax(logits.float(), -1); idx = p.argmax(-1); hard = F.one_hot(idx, VOCAB).float()
        return ((hard + (p - p.detach())).to(Ecodec.dtype)) @ Ecodec, idx
    def sample_t(): return float(np.clip(torch.sigmoid(torch.randn(())).item(), 0.02, 0.98))
    def T_(m): return m if torch.is_tensor(m) else torch.tensor(m, device=dev)
    def window(item, key, s=None):
        n = min(len(item[key]), item["tr"].n); s = random.randint(0, max(0, n - args.win)) if s is None else s
        return s, item["tr"].window(s, args.win), item[key][s:s + args.win]
    def soft_ce(lg, y):
        logp = F.log_softmax(lg.float(), -1); return (0.7 * (-logp.gather(1, y[:, None])[:, 0]) + 0.25 * (-(logp.gather(1, NB[y]) * NW[y]).sum(1)) + 0.05 * (-logp.mean(1))).mean()

    @torch.no_grad()
    def evaluate():
        head.eval(); g = torch.Generator(device="cpu").manual_seed(123); tot = 0; cnt = 0; reps = []
        for it in hold:
            n = len(it["lat"])
            for frac in (0.3, 0.6):
                s = int(n * frac) if n * frac + args.win <= n else max(0, n - args.win); _, m, z = window(it, "lat", s)
                if len(z) < args.win: continue
                with torch.autocast("cuda", dtype=torch.bfloat16): idx = head(T_(m)[None])[0].float().argmax(-1)
                reps.append(float((idx[1:] == idx[:-1]).float().mean())); noise = torch.randn(args.win, 64, generator=g).to(dev)
                for t in (0.2, 0.5, 0.8): tot += flow_loss(it["prefix"], Ecodec[idx], torch.tensor(z, device=dev), t, noise, False, False).item(); cnt += 1
        t1 = tot_ = 0
        for it in mval[:40]:
            _, m, y = window(it, "tok")
            with torch.autocast("cuda", dtype=torch.bfloat16): idx = head(T_(m)[None])[0].float().argmax(-1).cpu().numpy()
            t1 += (idx[:len(y)] == y).sum(); tot_ += len(y)
        head.train(bool(args.train_head)); return tot / max(cnt, 1), t1 / max(tot_, 1), float(np.mean(reps)) if reps else 0.0

    def save_all(tag):
        torch.save({"model": head.state_dict(), "cfg": head.cfg, "layers": layers, "step": st, "tag": tag}, out / f"head_{tag}.pt")
        if args.train_lora: save_lora(out / f"lora_{tag}.pt")

    def real_step(it, t, noise):
        s, m, z = window(it, "lat")
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = head(T_(m)[None])[0]
        emb, _ = st_embed(lg)
        ln = flow_loss(it["prefix"], emb, torch.tensor(z, device=dev), t, noise, bool(args.train_head), bool(args.train_lora))
        lc = torch.zeros((), device=dev)
        if args.train_head and args.ce_weight > 0:
            xs, ys = [], []
            for _ in range(args.ce_bs):
                mi = random.choice(mtrain); _, mm, yy = window(mi, "tok")
                if mm.shape[1] < args.win: mm = np.pad(mm, ((0, 0), (0, args.win - mm.shape[1]), (0, 0))); yy = np.pad(yy, (0, args.win - len(yy)), constant_values=-100)
                xs.append(mm); ys.append(yy)
            xb = torch.tensor(np.stack(xs), device=dev); yb = torch.tensor(np.stack(ys), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): lgb = head(xb)
            msk = yb != -100; lc = args.ce_weight * soft_ce(lgb[msk], yb[msk])
        return ln, lc

    st = 0; e0, a0, r0 = evaluate(); print(f"EVAL step 0 real_nar {e0:.4f} minted_top1 {a0:.4f} real_repeat {r0:.3f}", flush=True)
    log = open(out / "log.txt", "a"); log.write(f"EVAL step 0 real_nar {e0:.4f} minted_top1 {a0:.4f} real_repeat {r0:.3f}\n"); best = e0; t0 = time.time()
    json.dump(vars(args) | dict(real=len(real), hold=len(hold), online=len(online), minted=len(mtrain)), open(out / "config.json", "w"), indent=1)
    pending = []; n_online = n_stored = 0; wait_s = 0.0
    for st in range(1, args.steps + 1):
        mult = min(1.0, st / args.warmup) * (0.5 * (1 + math.cos(math.pi * st / args.steps)) * 0.9 + 0.1)
        for g_, b in zip(opt.param_groups, base_lrs): g_["lr"] = b * mult
        t = sample_t(); noise = torch.randn(args.win, 64, device=dev); src = "minted"
        if args.train_lora and random.random() < args.minted_flow_p:
            it = random.choice(mflow); s, m, y = window(it, "tok")
            z = np.load(it["lat_path"]).astype(np.float32)[s:s + args.win]; n = min(len(z), len(y))
            ln = flow_loss(prefix_of(it["style"], it["lyrics"], it["name"]), Ecodec[torch.tensor(y[:n], device=dev)], torch.tensor(z[:n], device=dev), t, noise[:n], False, True); lc = torch.zeros((), device=dev)
        else:
            p_on = args.online_p if args.online_p >= 0 else len(online) / max(1, len(online) + len(real))
            if pending: it = pending.pop(); src = "online"
            elif online and random.random() < p_on:
                tw = time.time(); it = q.get(); wait_s += time.time() - tw; src = "online"
                pending.extend([it] * (args.wins_per_song - 1))
            else: it = random.choice(real); src = "stored"
            ln, lc = real_step(it, t, noise); n_online += src == "online"; n_stored += src == "stored"
        loss = ln + lc; opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g_ in opt.param_groups for p in g_["params"]], 1.0); opt.step()
        if st <= 3 or st % 25 == 0:
            print(f"step {st} {src} nar {ln.item():.4f} ce {lc.item():.3f} t {t:.2f} {time.time() - t0:.0f}s (wait {wait_s:.0f}s) online/stored {n_online}/{n_stored} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if st % args.rescan_every == 0:
            k = rescan()
            if k: print(f"  rescan: +{k} online songs -> {len(online)}", flush=True)
        if st % args.eval_every == 0 or st == args.steps:
            e, a, rp = evaluate(); msg = f"EVAL step {st} real_nar {e:.4f} minted_top1 {a:.4f} real_repeat {rp:.3f} {time.time() - t0:.0f}s"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if e < best: best = e; save_all("best")
            save_all("last")
    stop.set(); print(f"RESULT best real_nar {best:.4f} (start {e0:.4f})", flush=True)


if __name__ == "__main__":
    main()
