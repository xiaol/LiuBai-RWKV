#!/usr/bin/env python3
"""Joint training of the R1 inverse tokenizer and an R2 NAR LoRA with YuE2's flow-matching loss as teacher (PLAN §8.3).

Ported from Mothersuperior's joint.py to our data layout and our InverseTokenizer (4 MERT layers).
Per step:
  real window   (data/yue2_real/*/<id>): MERT feats -> head -> straight-through token embeddings -> frozen(+LoRA) YuE2
                 AR/NAR pass -> flow-matching MSE against the song's TRUE VAE latents. Gradients reach the head through the
                 codec embeddings (grad through the AR stack) and the NAR LoRA/vae2llm/llm2vae if enabled.
  minted window (data/yue2_minted*/<id>): soft-CE of the head on true tokens (anchor), and with prob --minted-flow-p a flow
                 window with the TRUE tokens (keeps the NAR LoRA honest on in-distribution tokens).
Eval every --eval-every: held-out real flow loss (fixed windows / t / noise), minted top-1, repeat rate of predicted tokens.
Run inside venvs/yue2 on one GPU. Memory ~ 8 GiB weights + activations (window 256 -> ~4-6 GiB with checkpointing).

usage: train_joint_teacher.py --out out/joint/v1 --head-init out/inverse_tok/v2/best.pt --train-head 1 --train-lora 1
         [--rank 32] [--lora-init none] [--win 256] [--steps 3000] [--real-dirs data/yue2_real/shard-00000]
         [--minted-dirs data/yue2_minted/tracks data/yue2_minted_own] [--hold N]
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
YUE2 = ROOT / "models/yue2"
sys.path.insert(0, str(ROOT / "scripts"))
from train_inverse_tokenizer import InverseTokenizer, instnorm, predict as head_predict  # noqa: E402
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
    ap.add_argument("--win", type=int, default=256); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--real-dirs", nargs="+", default=[str(ROOT / "data/yue2_real/shard-00000")])
    ap.add_argument("--minted-dirs", nargs="+", default=[str(ROOT / "data/yue2_minted/tracks"), str(ROOT / "data/yue2_minted_own")])
    ap.add_argument("--max-real", type=int, default=0); ap.add_argument("--max-minted", type=int, default=0)
    ap.add_argument("--lr-head", type=float, default=5e-5); ap.add_argument("--lr-lora", type=float, default=5e-5); ap.add_argument("--lr-io", type=float, default=2e-5)
    ap.add_argument("--ce-weight", type=float, default=1.0); ap.add_argument("--ce-bs", type=int, default=4)
    ap.add_argument("--minted-flow-p", type=float, default=0.25); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--hold", type=int, default=8, help="number of held-out real tracks (md5 order)"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = True
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.protocol import CODEC_OFFSET, MUSIC_END, SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from yue2.nar import attention as nar_attention

    # ---- YuE2 (frozen) + optional LoRA on the NAR branch ----
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

    # ---- data ----
    def load_real(d):
        m = np.load(d / FEAT); z = np.load(d / "latent_true.npy").astype(np.float32); n = min(m.shape[1], len(z))
        return dict(name=d.name, mert=instnorm(m[:, :n]).astype(np.float16), lat=z[:n], prefix=[int(v) for v in np.load(d / "prefix.npy")])
    real_dirs = sorted(d for rd in args.real_dirs for d in Path(rd).iterdir() if (d / "prefix.npy").exists() and (d / FEAT).exists())
    real_dirs.sort(key=lambda d: hashlib.md5(d.name.encode()).hexdigest()); hold_dirs, real_dirs = real_dirs[:args.hold], real_dirs[args.hold:]
    if args.max_real: real_dirs = real_dirs[:args.max_real]
    real = [load_real(d) for d in real_dirs if len(np.load(d / "latent_true.npy")) >= args.win]; hold = [load_real(d) for d in hold_dirs]
    def load_minted(d):
        m = np.load(d / FEAT); y = np.load(d / "semantic.npy").astype(np.int64); n = min(m.shape[1], len(y))
        r = json.load(open(d / "request.json")) if (d / "request.json").exists() else {}
        it = dict(name=d.name, mert=instnorm(m[:, :n]).astype(np.float16), tok=y[:n], style=r.get("style", ""), lyrics=r.get("lyrics", ""))
        if (d / "latent.npy").exists(): it["lat_path"] = d / "latent.npy"
        return it
    minted_dirs = sorted(d for md in args.minted_dirs for d in Path(md).iterdir() if (d / FEAT).exists() and (d / "semantic.npy").exists())
    if args.max_minted: minted_dirs = minted_dirs[:args.max_minted]
    mheld = lambda d: int(hashlib.md5(d.name.encode()).hexdigest(), 16) % 20 == 0
    mtrain = [load_minted(d) for d in minted_dirs if not mheld(d)]; mval = [load_minted(d) for d in minted_dirs if mheld(d)]
    print(f"real train {len(real)} held {len(hold)} | minted train {len(mtrain)} val {len(mval)} | win {args.win} head {args.train_head} lora {args.train_lora} rank {args.rank}", flush=True)
    prefix_cache = {}
    def minted_prefix(it):
        if it["name"] not in prefix_cache:
            style = " ".join(it["style"].split())[:1500] or "pop"; lyrics = it["lyrics"].strip() or "[instrumental]"
            prefix_cache[it["name"]] = token_prefixes(SongRequest(style=style, lyrics=lyrics, cot="off", seed=1, id=it["name"]), tok)
        return prefix_cache[it["name"]]

    # ---- YuE2 forward pieces (from joint.py) ----
    def ar_layer(layer, x, cos_, sin_):
        q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos_, sin_); h = nar_attention(q[0], k[0], v[0], causal=True)
        x = x + layer.self_attn.o_proj(h.flatten(1)[None]); return x + layer.mlp(layer.post_attention_layernorm(x)), k[0], v[0]
    def nar_layer(layer, h, ak, av, ncos, nsin):
        q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(h), ncos, nsin); a = nar_attention(q[0], torch.cat((ak, k[0])), torch.cat((av, v[0])))
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
    def window(item, key, s=None):
        n = len(item[key]); s = random.randint(0, max(0, n - args.win)) if s is None else s; return s, item["mert"][:, s:s + args.win], item[key][s:s + args.win]
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
                with torch.autocast("cuda", dtype=torch.bfloat16): idx = head(torch.tensor(instnorm(m.astype(np.float32))[None], device=dev))[0].float().argmax(-1)
                reps.append(float((idx[1:] == idx[:-1]).float().mean())); noise = torch.randn(args.win, 64, generator=g).to(dev)
                for t in (0.2, 0.5, 0.8): tot += flow_loss(it["prefix"], Ecodec[idx], torch.tensor(z, device=dev), t, noise, False, False).item(); cnt += 1
        t1 = tot_ = 0
        for it in mval[:40]:
            _, m, y = window(it, "tok")
            with torch.autocast("cuda", dtype=torch.bfloat16): idx = head(torch.tensor(m.astype(np.float32)[None], device=dev))[0].float().argmax(-1).cpu().numpy()
            t1 += (idx[:len(y)] == y).sum(); tot_ += len(y)
        head.train(bool(args.train_head)); return tot / max(cnt, 1), t1 / max(tot_, 1), float(np.mean(reps)) if reps else 0.0

    def save_all(tag):
        torch.save({"model": head.state_dict(), "cfg": head.cfg, "layers": layers, "step": st, "tag": tag}, out / f"head_{tag}.pt")
        if args.train_lora: save_lora(out / f"lora_{tag}.pt")

    st = 0; e0, a0, r0 = evaluate(); print(f"EVAL step 0 real_nar {e0:.4f} minted_top1 {a0:.4f} real_repeat {r0:.3f}", flush=True)
    log = open(out / "log.txt", "a"); log.write(f"EVAL step 0 real_nar {e0:.4f} minted_top1 {a0:.4f} real_repeat {r0:.3f}\n"); best = e0; t0 = time.time()
    json.dump(vars(args) | dict(real=len(real), hold=len(hold), minted=len(mtrain)), open(out / "config.json", "w"), indent=1)
    for st in range(1, args.steps + 1):
        mult = min(1.0, st / 50) * (0.5 * (1 + math.cos(math.pi * st / args.steps)) * 0.9 + 0.1)
        for g_, b in zip(opt.param_groups, base_lrs): g_["lr"] = b * mult
        t = sample_t(); noise = torch.randn(args.win, 64, device=dev)
        if args.train_lora and random.random() < args.minted_flow_p:               # minted flow window with TRUE tokens
            it = random.choice([m for m in mtrain if "lat_path" in m]); s, m, y = window(it, "tok")
            z = np.load(it["lat_path"]).astype(np.float32)[s:s + args.win]; n = min(len(z), len(y))
            ln = flow_loss(minted_prefix(it), Ecodec[torch.tensor(y[:n], device=dev)], torch.tensor(z[:n], device=dev), t, noise[:n], False, True); lc = torch.zeros((), device=dev)
        else:                                                                       # real window through the head
            it = random.choice(real); s, m, z = window(it, "lat")
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = head(torch.tensor(m.astype(np.float32)[None], device=dev))[0]
            emb, _ = st_embed(lg)
            ln = flow_loss(it["prefix"], emb, torch.tensor(z, device=dev), t, noise, bool(args.train_head), bool(args.train_lora))
            lc = torch.zeros((), device=dev)
            if args.train_head and args.ce_weight > 0:                              # anchor on minted exact tokens
                xs, ys = [], []
                for _ in range(args.ce_bs):
                    mi = random.choice(mtrain); _, mm, yy = window(mi, "tok")
                    if mm.shape[1] < args.win: mm = np.pad(mm, ((0, 0), (0, args.win - mm.shape[1]), (0, 0))); yy = np.pad(yy, (0, args.win - len(yy)), constant_values=-100)
                    xs.append(mm.astype(np.float32)); ys.append(yy)
                xb = torch.tensor(np.stack(xs), device=dev); yb = torch.tensor(np.stack(ys), device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): lgb = head(xb)
                msk = yb != -100; lc = args.ce_weight * soft_ce(lgb[msk], yb[msk])
        loss = ln + lc; opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g_ in opt.param_groups for p in g_["params"]], 1.0); opt.step()
        if st <= 3 or st % 25 == 0:
            print(f"step {st} nar {ln.item():.4f} ce {lc.item():.3f} t {t:.2f} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if st % args.eval_every == 0 or st == args.steps:
            e, a, rp = evaluate(); msg = f"EVAL step {st} real_nar {e:.4f} minted_top1 {a:.4f} real_repeat {rp:.3f} {time.time() - t0:.0f}s"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if e < best: best = e; save_all("best")
            save_all("last")
    print(f"RESULT best real_nar {best:.4f} (start {e0:.4f})", flush=True)


if __name__ == "__main__":
    main()
