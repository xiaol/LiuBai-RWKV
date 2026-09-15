#!/usr/bin/env python3
"""R1 inverse tokenizer (PLAN §8.3): MERT-v2-FullSong multi-layer features @25 Hz -> YuE2 semantic token (32,768-way).

Data: track dirs holding mert_L{layers}.npy (fp16 [K,T,1024], from tools/yue2_extract_mert.py) and semantic.npy (int32 [T]).
Model: learned softmax over the K layers -> Linear -> bidirectional Transformer encoder (pre-LN, GELU, learned positions
over the training window) -> LayerNorm -> Linear(32768). Loss: CE with alpha-weighted soft targets over the token's 8
nearest codec-embedding neighbours (data/yue2_minted/codec_nbr_*.npy, from YuE2's own embedding rows), label smoothing.
Held-out split by md5(track id) % 20 == 0. Eval: top-1 / top-5 / top-1-or-neighbour on held-out windows.
Inference helper `predict(model, feats)` does 50 %-overlap sliding windows (as the v4 head) and is imported by the
round-trip tool via --head-ckpt.

usage: train_inverse_tokenizer.py --dirs data/yue2_minted/tracks data/yue2_minted_own --out out/inverse_tok/v1
          [--layers 12,16,20,23] [--win 1000] [--d 768] [--nlayer 12] [--steps 20000] [--bs 16] [--lr 3e-4] [--init ckpt]
"""
import argparse, glob, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 32768


class InverseTokenizer(nn.Module):
    def __init__(s, k_layers, din=1024, d=768, nlayer=12, nhead=12, win=1000, drop=0.1):
        super().__init__(); s.cfg = dict(k_layers=k_layers, din=din, d=d, nlayer=nlayer, nhead=nhead, win=win)
        s.layer_w = nn.Parameter(torch.zeros(k_layers)); s.inp = nn.Linear(din, d)
        s.pos = nn.Parameter(torch.zeros(1, win, d)); nn.init.normal_(s.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=drop, batch_first=True, norm_first=True, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, nlayer, enable_nested_tensor=False); s.norm = nn.LayerNorm(d); s.head = nn.Linear(d, VOCAB)

    def forward(s, x):                       # x [B, K, T, din] instance-normalised
        w = torch.softmax(s.layer_w, 0); h = (x * w[None, :, None, None]).sum(1)
        h = s.inp(h) + s.pos[:, :h.shape[1]]
        return s.head(s.norm(s.enc(h)))


def instnorm(x):                             # per track, per layer, per channel
    x = x.astype(np.float32); return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)


@torch.no_grad()
def predict(model, feats, dev="cuda"):      # feats fp16 [K,T,1024] -> int64 [T]
    win = model.cfg["win"]; x = instnorm(feats); T = x.shape[1]; out = np.zeros(T, dtype=np.int64)
    starts = list(range(0, max(1, T - win + 1), win // 2))
    if starts[-1] + win < T: starts.append(max(0, T - win))
    for s0 in starts:
        xw = x[:, s0:s0 + win]; n = xw.shape[1]
        if n < win: xw = np.pad(xw, ((0, 0), (0, win - n), (0, 0)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(torch.tensor(xw[None], device=dev))[0, :n].float().argmax(-1).cpu().numpy()
        lo = s0 + (0 if s0 == 0 else win // 4); hi = s0 + n - (0 if s0 + n >= T else win // 4)
        out[lo:hi] = pred[lo - s0:hi - s0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="12,16,20,23"); ap.add_argument("--win", type=int, default=1000)
    ap.add_argument("--d", type=int, default=768); ap.add_argument("--nlayer", type=int, default=12); ap.add_argument("--nhead", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--alpha", type=float, default=0.25, help="neighbour soft-target weight")
    ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--init", default="")
    ap.add_argument("--max-tracks", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-batches (each of --bs windows)")
    ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--time-mask", type=float, default=0.0, help="fraction of frames zeroed in random spans (SpecAugment-style)")
    ap.add_argument("--feat-drop", type=float, default=0.0, help="input channel dropout on the MERT features")
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = True
    layers = [int(x) for x in args.layers.split(",")]; tag = "".join(f"_{l}" for l in layers); K = len(layers)

    tracks = []
    for d in args.dirs:
        for t in sorted(Path(d).iterdir()):
            if (t / f"mert_L{tag}.npy").exists() and (t / "semantic.npy").exists(): tracks.append(t)
    if args.max_tracks: tracks = tracks[:args.max_tracks]
    held = lambda t: int(hashlib.md5(t.name.encode()).hexdigest(), 16) % 20 == 0
    def load(t):
        x = np.load(t / f"mert_L{tag}.npy"); y = np.load(t / "semantic.npy").astype(np.int64); n = min(x.shape[1], len(y))
        if abs(x.shape[1] - len(y)) > 3: print(f"  warn {t.name}: mert {x.shape[1]} vs tokens {len(y)}")
        return instnorm(x[:, :n]).astype(np.float16), y[:n]
    t0 = time.time(); train = [load(t) for t in tracks if not held(t)]; val = [load(t) for t in tracks if held(t)]
    ntr = sum(len(y) for _, y in train); nva = sum(len(y) for _, y in val)
    print(f"tracks train {len(train)} ({ntr / 25 / 3600:.1f} h, {ntr:,} frames) val {len(val)} ({nva:,} frames) loaded in {time.time() - t0:.0f}s", flush=True)
    NB = torch.tensor(np.load(ROOT / "data/yue2_minted/codec_nbr_idx.npy").astype(np.int64), device=dev)
    NW = torch.softmax(torch.tensor(np.load(ROOT / "data/yue2_minted/codec_nbr_cos.npy"), device=dev) / args.tau, dim=1)

    model = InverseTokenizer(K, d=args.d, nlayer=args.nlayer, nhead=args.nhead, win=args.win, drop=args.drop).to(dev)
    if args.init: model.load_state_dict(torch.load(args.init, map_location=dev)["model"]); print("init from", args.init)
    nparam = sum(p.numel() for p in model.parameters()); print(f"params {nparam / 1e6:.1f} M", flush=True)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos" not in n]; nodecay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": nodecay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    sched = lambda st: min(1.0, st / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, st / args.steps)))

    def batch(data, bs, rng=random):
        xs, ys = [], []
        for _ in range(bs):
            x, y = rng.choice(data); s = rng.randint(0, max(0, x.shape[1] - args.win)); xw = x[:, s:s + args.win]; yw = y[s:s + args.win]
            if xw.shape[1] < args.win:
                pad = args.win - xw.shape[1]; xw = np.pad(xw, ((0, 0), (0, pad), (0, 0))); yw = np.pad(yw, (0, pad), constant_values=-100)
            xs.append(xw); ys.append(yw)
        return torch.tensor(np.stack(xs), device=dev), torch.tensor(np.stack(ys), device=dev)

    def loss_fn(lg, y):
        m = y != -100; lg = lg[m].float(); y = y[m]; logp = F.log_softmax(lg, -1)
        hard = -logp.gather(1, y[:, None])[:, 0]; nbr = -(logp.gather(1, NB[y]) * NW[y]).sum(1); uni = -logp.mean(1)
        return ((1 - args.alpha - args.smooth) * hard + args.alpha * nbr + args.smooth * uni).mean()

    fixed = random.Random(123); evb = [batch(val, 8, fixed) for _ in range(8)] if val else []
    @torch.no_grad()
    def evaluate():
        model.eval(); t1 = t5 = nb = tot = 0
        for x, y in evb:
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(x).float()
            m = y != -100; p1 = lg.argmax(-1); top5 = lg.topk(5, -1).indices
            t1 += (p1 == y)[m].sum().item(); t5 += (top5 == y[..., None]).any(-1)[m].sum().item()
            nb += ((p1 == y) | (NB[y.clamp(min=0)] == p1[..., None]).any(-1))[m].sum().item(); tot += m.sum().item()
        model.train(); return (t1 / tot, t5 / tot, nb / tot) if tot else (0, 0, 0)

    log = open(out / "log.txt", "a"); best = 0.0; t0 = time.time(); model.train()
    json.dump(vars(args) | dict(params=nparam, tracks_train=len(train), tracks_val=len(val), frames_train=ntr), open(out / "config.json", "w"), indent=1)
    for st in range(1, args.steps + 1):
        for g in opt.param_groups: g["lr"] = args.lr * sched(st)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            x, y = batch(train, args.bs)
            if args.time_mask > 0:                       # zero ~time_mask of the frames in 25-frame (1 s) spans
                nspan = max(1, int(args.time_mask * args.win / 25)); x = x.clone()
                for b in range(x.shape[0]):
                    for s0 in torch.randint(0, args.win - 25, (nspan,)).tolist(): x[b, :, s0:s0 + 25] = 0
            if args.feat_drop > 0: x = F.dropout(x, args.feat_drop, training=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(x)
            loss = loss_fn(lg, y) / args.accum; loss.backward(); del lg
        loss = loss * args.accum; gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if st <= 5 or st % 50 == 0:
            msg = f"step {st} loss {loss.item():.4f} gn {gn:.2f} lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if st % args.eval_every == 0 or st == args.steps:
            a1, a5, an = evaluate(); msg = f"EVAL step {st} top1 {a1:.4f} top5 {a5:.4f} top1-or-nbr {an:.4f} layer_w {[round(v, 3) for v in torch.softmax(model.layer_w, 0).tolist()]}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            ck = {"model": model.state_dict(), "cfg": model.cfg, "layers": layers, "step": st, "top1": a1}
            torch.save(ck, out / "last.pt")
            if a1 > best: best = a1; torch.save(ck, out / "best.pt")
    print(f"RESULT best top1 {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
