#!/usr/bin/env python3
"""PER-ranked preference fine-tune (Diffusion-DPO, flow-matching form) of a gated block-flow cursor LM (vae/train_cursor_lm.py
--head gated). Target: lyric binding WITHOUT CFG (the E1 gates bind at CFG 3 / 5 but not at CFG 1, docs/VAE_PLAN.md).

pairs : training crops (not the 96 gate crops: those come from slim_val), 4 s causal prompt + 20 s generated with the crop's own
        lyrics, --cands CFG values per crop (e.g. 1,1,1,1,3,3), fast batched sampler (same as vae/eval_cursor_lm.py
        generate_batch), decode with the frozen VAE, Whisper (HeartTranscriptor) letter CER vs the given lyrics.
        -> <out>/pairs_r<rank>.pt: per crop the normalised latent sequence of every candidate, its sampled advances, CER, CFG.
train : winner = lowest-CER candidate, loser = highest, kept if CER gap >= --margin (--use-cfg restricts the candidate pool, e.g.
        1 = on-policy CFG-1 samples only). Per pair the same t / noise for winner and loser (Wallace et al. 2023):
          d_flow = [e_th(w) - e_ref(w)] - [e_th(l) - e_ref(l)]   (e = sem-weighted squared velocity error, mean per element)
          d_adv  = [lp_th(w) - lp_ref(w)] - [lp_th(l) - lp_ref(l)] (summed log-prob of the sampled cursor advances)
          loss   = -logsigmoid(-0.5 * beta * d_flow + beta_adv * d_adv) + sft_w * e_th(w)
        reference = frozen copy of the starting weights. Exports <out>/M_s<step>.pt in the format eval_cursor_lm.py loads.
usage: dpo_cursor_lm.py pairs --ckpt out/clm/e1_gated_s0_wkv/M_ema_s20000.pt --out out/dpo/p1 --n 1200 --rank 0 --world 2
       dpo_cursor_lm.py train --ckpt out/clm/e1_gated_s0_wkv/M_ema_s20000.pt --pairs out/dpo/p1 --out out/dpo/a_onpol --use-cfg 1
"""
import argparse, copy, json, math, random, sys, time
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import LETTERS  # noqa: E402
from vae.data import Crops, romanise, encode_text  # noqa: E402
from vae.train import edit  # noqa: E402
from vae.cursor_lm import windows, NULL  # noqa: E402
from vae.gated_nar import GatedCursorLM  # noqa: E402

dev = "cuda"; RATE = 12.5


def letters(ids): return "".join(LETTERS[i - 1] for i in ids)


def load_lm(ckpt):
    sd = torch.load(ckpt, map_location="cpu"); ta = sd["args"]; assert ta.get("head") == "gated", "gated cursor LMs only"
    M = GatedCursorLM(blk=ta["blk"], d=ta["d"], n=ta["n"], lora=ta["lora"], iso=ta.get("iso", False), lattn=ta.get("lattn", False))
    M.load_state_dict(sd.get("ema", sd["M"])); return M.to(dev).eval(), ta, sd["lat_m"].to(dev), sd["lat_s"].to(dev)


@torch.no_grad()
def generate_batch(M, zp, idss, NB, cfg, steps, gen, adv_temp=1.0):
    """= eval_cursor_lm.py generate_batch without oracle timing: zp [R,nb0,D] prompts, idss R letter-id lists -> z [R,NB,D],
    R advance lists (blocks nb0..NB-1). cfg: lyric CFG scale (1 = off)."""
    R, nb0, D = zp.shape; usecfg = cfg != 1.0; RR = 2 * R if usecfg else R
    z = torch.zeros(R, NB, D, device=dev); z[:, :nb0] = zp; cache = M.cache_init(RR, dev)
    cs = np.zeros(R, np.int64); adv = [[] for _ in range(R)]; ts = torch.linspace(0, 1, steps + 1, device=dev)
    two = (lambda x: torch.cat([x, x])) if usecfg else (lambda x: x)
    for i in range(NB):
        W = torch.from_numpy(np.stack([windows(ids, cs[r:r + 1], M.wb, M.wf)[0] for r, ids in enumerate(idss)])).to(dev)
        WW = torch.cat([W, torch.full_like(W, NULL)]) if usecfg else W
        zprev = None if i == 0 else two(z[:, i - 1])
        with torch.autocast("cuda", dtype=torch.bfloat16): h = M.step_hidden(zprev, WW, cache)
        if i < nb0: continue
        samp = torch.multinomial(torch.softmax(M.adv(h[:R]).float() / adv_temp, -1), 1, generator=gen)[:, 0].tolist()
        ac = np.zeros(R, np.int64)
        for r, ids in enumerate(idss):
            a = min(samp[r], max(0, len(ids) + 1 - int(cs[r]))); ac[r] = min(a, M.max_adv - 1); adv[r].append(int(ac[r])); cs[r] += ac[r]
        A = torch.from_numpy(ac).to(dev); AA = torch.cat([A, torch.full_like(A, M.max_adv)]) if usecfg else A
        x = torch.randn(R, D, device=dev, generator=gen)
        for j in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16): v = M.step_velocity(two(x), ts[j].expand(RR), WW, AA, cache, zprev).float()
            if usecfg: v = v[R:] + cfg * (v[:R] - v[R:])
            x = x + (ts[j + 1] - ts[j]) * v
        z[:, i] = x
    return z, adv


def cmd_pairs(args):
    from vae.model import load_vae
    M, ta, lat_m, lat_s = load_lm(args.ckpt); V = load_vae(ta["vae"], dev)
    ekw = dict(blk=ta["blk"], keep=ta.get("enc_keep", 2), win=ta.get("enc_win", 3.0), ahead=ta.get("enc_ahead", 0.64))
    encode = (lambda w: V.encode_local(w, **ekw)) if ta.get("enc") == "local" and V.sem else V.encode
    blk = M.blk; g0 = int(round(args.ctx * RATE)) // blk * blk; nb0 = g0 // blk; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = Crops(args.manifest, holdout=0, seed=args.seed + 1000 * args.rank, seg=args.seg, with_lines=True, sr=V.sr)
    ds.rows = [r for r in ds.rows if len(r["lines"]) >= 4]; random.seed(args.seed + args.rank); it = iter(ds)
    items = []
    while len(items) < args.n // args.world:
        wav, tg, ok, cf, lines = next(it)
        own = [l for l in lines if l[0] >= g0 and l[2] <= args.seg * RATE]
        txt = romanise(" ".join(letters(l[1]) for l in own)).strip()
        if len(txt) >= 20: items.append((wav, txt))
    cands = [float(c) for c in args.cands.split(",")]
    print(f"rank {args.rank}: {len(items)} crops x {len(cands)} candidates (cfg {cands}), prompt {g0 / RATE:.1f} s", flush=True)
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(args.transcriptor)
    asr = WhisperForConditionalGeneration.from_pretrained(args.transcriptor, dtype=torch.float16).to(dev).eval()

    def transcribe(ys):                                                     # list of mono arrays at V.sr -> transcripts
        g = gcd(V.sr, 16000); ys = [resample_poly(y, 16000 // g, V.sr // g)[:16000 * 30] for y in ys]
        f = proc(ys, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
        with torch.inference_mode(): ids = asr.generate(f, task="transcribe", max_new_tokens=220)
        return [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]

    def cer(hyp, ref):
        h, r = romanise(hyp).replace(" ", ""), ref.replace(" ", ""); return edit(list(h), list(r)) / max(1, len(r))

    gen = torch.Generator(device=dev).manual_seed(args.seed + args.rank); recs = []; t0 = time.time(); S0 = g0 * V.hop
    for q in range(0, len(items), args.crops_per_batch):
        chunk = items[q:q + args.crops_per_batch]; zps, idss, NB = [], [], None
        for wav, txt in chunk:
            x = wav[None].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                mu = encode(x)[0].float(); mp = encode(x[..., :S0])[0].float()       # causal prompt, as the gate
            NB = mu.shape[1] // blk; zps.append(((mp[:, :g0] - lat_m) / lat_s).reshape(1, nb0, -1)); idss.append(encode_text(txt))
        zp = torch.cat(zps); Z = {}; ADV = {}
        for ci, c in enumerate(cands):
            z, adv = generate_batch(M, zp, idss, NB, c, args.steps, gen); Z[ci] = z.half().cpu(); ADV[ci] = adv
        ys, keys = [], []
        for ci in range(len(cands)):
            for r in range(len(chunk)):
                zz = Z[ci][r].float().to(dev).reshape(1, -1, 64) * lat_s + lat_m
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): y = V.decode(zz).float()[0].mean(0).cpu().numpy()
                ys.append(y[S0:]); keys.append((ci, r))
        trs = []
        for b in range(0, len(ys), 16): trs += transcribe(ys[b:b + 16])
        cm = np.zeros((len(chunk), len(cands)), np.float32)
        for (ci, r), t in zip(keys, trs): cm[r, ci] = cer(t, chunk[r][1])
        for r, (wav, txt) in enumerate(chunk):
            recs.append(dict(z=torch.stack([Z[ci][r] for ci in range(len(cands))]), adv=[ADV[ci][r] for ci in range(len(cands))],
                             ids=idss[r], txt=txt, cer=cm[r].tolist(), cfg=cands, nb0=nb0))
            if r == 0 and q % (4 * args.crops_per_batch) == 0:
                for (ci, rr), y in zip(keys, ys):
                    if rr == 0: sf.write(out / f"ex_r{args.rank}_{q:05d}_c{ci}.flac", y, V.sr)
        c1 = np.array([rc["cer"] for rc in recs])
        print(f"rank {args.rank}: {len(recs)}/{len(items)} crops, mean CER per cand {np.round(c1.mean(0), 3).tolist()}, "
              f"best-of-all {c1.min(1).mean():.3f} {time.time() - t0:.0f}s", flush=True)
        torch.save(dict(recs=recs, ckpt=args.ckpt, cands=cands), out / f"pairs_r{args.rank}.tmp"); (out / f"pairs_r{args.rank}.tmp").replace(out / f"pairs_r{args.rank}.pt")


def cmd_train(args):
    M, ta, lat_m, lat_s = load_lm(args.ckpt); ref = copy.deepcopy(M).eval().requires_grad_(False); M.train(); M.ckpt = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    recs = []
    for p in sorted(Path(args.pairs).glob("pairs_r*.pt")): recs += torch.load(p, map_location="cpu")["recs"]
    use = None if args.use_cfg == "all" else {float(c) for c in args.use_cfg.split(",")}
    pairs = []
    for rc in recs:
        ok = [i for i, c in enumerate(rc["cfg"]) if use is None or c in use]
        if len(ok) < 2: continue
        cr = [rc["cer"][i] for i in ok]; w, l = ok[int(np.argmin(cr))], ok[int(np.argmax(cr))]
        if rc["cer"][l] - rc["cer"][w] >= args.margin: pairs.append((rc, w, l))
    rng = random.Random(args.seed); rng.shuffle(pairs); nval = min(max(8, len(pairs) // 20), len(pairs) // 4); val, trn = pairs[:nval], pairs[nval:]
    gw = [p[0]["cer"][p[1]] for p in pairs]; gl = [p[0]["cer"][p[2]] for p in pairs]
    print(f"{len(recs)} crops -> {len(pairs)} pairs (margin {args.margin}, cfg {args.use_cfg}): winner CER {np.mean(gw):.3f}, "
          f"loser {np.mean(gl):.3f}; {len(trn)} train / {len(val)} held-out", flush=True)
    K_sem = 32; dw = torch.ones(64, device=dev); dw[:K_sem] = ta.get("sem_w", 1.0); dw = dw / dw.mean()
    opt = torch.optim.AdamW([p for p in M.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)

    def batch(ps):
        """-> z [2P,NB,D] (winners then losers), lyr [2P,NB,W], a_c [2P,NB], adv [2P,NB-nb0], nb0"""
        zs, ly, ac, av = [], [], [], []; nb0 = ps[0][0]["nb0"]
        for side in (1, 2):
            for p in ps:
                rc = p[0]; ci = p[side]; z = rc["z"][ci].float(); NB = z.shape[0]; a = np.asarray(rc["adv"][ci], np.int64)
                c = np.zeros(NB, np.int64); c[nb0 + 1:] = np.cumsum(a)[:NB - nb0 - 1]
                acn = np.full(NB, M.max_adv, np.int64); acn[nb0:] = a
                zs.append(z); ly.append(windows(rc["ids"], c, M.wb, M.wf)); ac.append(acn); av.append(a)
        T = lambda x: torch.from_numpy(np.stack(x)).to(dev)
        return torch.stack(zs).to(dev), T(ly), T(ac), T(av), nb0

    def errs(net, z, lyr, ac, avt, nb0, t, n):
        """sem-weighted squared velocity error per sequence [2P] (generated blocks only) + summed advance log-prob [2P]"""
        B, NB, D = z.shape; K = args.K; nar = range(nb0, NB); zt = z[:, nb0:, None].expand(B, NB - nb0, K, D)
        xt = (1 - t[..., None]) * n + t[..., None] * zt
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, v = net.velocity(z, lyr, ac, xt.reshape(B, -1, D), t.reshape(B, -1), nar, reps=K)
            lp = torch.log_softmax(net.adv(h[:, nb0:]).float(), -1).gather(-1, avt[..., None])[..., 0].sum(1)
        e = ((v.float().view(B, NB - nb0, K, -1, 64) - (zt - n).view(B, NB - nb0, K, -1, 64)) ** 2 * dw).mean((1, 2, 3, 4))
        return e, lp

    def dpo(ps, train=True):
        z, lyr, ac, avt, nb0 = batch(ps); P = len(ps); B, NB, D = z.shape
        t1 = torch.sigmoid(torch.randn(P, NB - nb0, args.K, device=dev)); n1 = torch.randn(P, NB - nb0, args.K, D, device=dev)
        t, n = torch.cat([t1, t1]), torch.cat([n1, n1])                      # same t / noise for winner and loser of a pair
        with torch.no_grad(): er, lr_ = errs(ref, z, lyr, ac, avt, nb0, t, n)
        with torch.set_grad_enabled(train): et, lt = errs(M, z, lyr, ac, avt, nb0, t, n)
        dfl = (et[:P] - er[:P]) - (et[P:] - er[P:]); dad = (lt[:P] - lr_[:P]) - (lt[P:] - lr_[P:])
        m = -0.5 * args.beta * dfl + args.beta_adv * dad
        loss = -F.logsigmoid(m).mean() + args.sft_w * et[:P].mean()
        return loss, dict(loss=loss.item(), acc=(m > 0).float().mean().item(), margin=m.mean().item(), dflow=dfl.mean().item() * 1e3,
                          dadv=dad.mean().item(), e_w=et[:P].mean().item(), e_l=et[P:].mean().item())

    log = open(out / "log.txt", "a"); t0 = time.time(); acc = {}
    for step in range(1, args.steps + 1):
        for g in opt.param_groups: g["lr"] = args.lr * min(1, step / 100)
        loss, st = dpo(rng.sample(trn, args.bs)); opt.zero_grad(set_to_none=True); loss.backward()
        st["gn"] = torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0).item(); opt.step()
        for k, v in st.items(): acc[k] = acc.get(k, 0) + v
        if step % 25 == 0:
            msg = f"step {step} " + " ".join(f"{k} {v / 25:.4f}" for k, v in acc.items()) + f" {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush(); acc = {}
        if step % args.save_every == 0 or step == args.steps:
            M.eval(); torch.manual_seed(7); vs = [dpo(val[i:i + args.bs], train=False)[1] for i in range(0, len(val), args.bs)]; M.train()
            msg = f"VAL step {step} " + " ".join(f"{k} {np.mean([v[k] for v in vs]):.4f}" for k in vs[0]); print(msg, flush=True); log.write(msg + "\n"); log.flush()
            sd = {k: v.detach().cpu() for k, v in M.state_dict().items()}
            torch.save(dict(M=sd, step=step, args=ta, dpo=vars(args), lat_m=lat_m.cpu(), lat_s=lat_s.cpu()), out / f"M_s{step}.pt")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("pairs"); a.add_argument("--ckpt", required=True); a.add_argument("--out", required=True)
    a.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/pilot_clm_train_clean.jsonl"))
    a.add_argument("--transcriptor", default=str(ROOT / "models/HeartTranscriptor-oss"))
    a.add_argument("--n", type=int, default=1200); a.add_argument("--rank", type=int, default=0); a.add_argument("--world", type=int, default=1)
    a.add_argument("--cands", default="1,1,1,1,3,3"); a.add_argument("--seg", type=float, default=24.0); a.add_argument("--ctx", type=float, default=4.0)
    a.add_argument("--steps", type=int, default=24); a.add_argument("--seed", type=int, default=777); a.add_argument("--crops-per-batch", type=int, default=64)
    b = sp.add_parser("train"); b.add_argument("--ckpt", required=True); b.add_argument("--pairs", required=True); b.add_argument("--out", required=True)
    b.add_argument("--use-cfg", default="all", help="candidate pool: 'all' or comma list of CFG values (1 = on-policy only)")
    b.add_argument("--margin", type=float, default=0.15); b.add_argument("--beta", type=float, default=2000.0); b.add_argument("--beta-adv", type=float, default=0.1)
    b.add_argument("--sft-w", type=float, default=0.0); b.add_argument("--lr", type=float, default=1e-5); b.add_argument("--bs", type=int, default=6)
    b.add_argument("--K", type=int, default=2); b.add_argument("--steps", type=int, default=1500); b.add_argument("--save-every", type=int, default=500)
    b.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); torch.backends.cuda.matmul.allow_tf32 = True
    cmd_pairs(args) if args.cmd == "pairs" else cmd_train(args)


if __name__ == "__main__":
    main()
