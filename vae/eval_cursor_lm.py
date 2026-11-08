#!/usr/bin/env python3
"""Gate-1 audio test for the block-flow cursor LM (vae/train_cursor_lm.py): does generated singing say the given words?

Held-out crops (slim_val.jsonl = real-corpus shard 0, rows with >= 4 timed lines), --seg s each. The first --ctx s are the
real latent (style / voice prompt, teacher-forced); the model then generates the rest block by block, given the romanised
letters of the lines that start after the prompt and end inside the crop:
    own  : the crop's own lines            swap : the lines of another crop (does it sing what it is told?)
    none : null lyrics (cursor) / empty prefix
Cursor mode samples its own advance (timing); prefix mode reads the whole letter string from its state.
Each output is decoded with the frozen VAE, transcribed by HeartTranscriptor (Whisper-large, language auto) and scored as
letter CER of the romanised transcript vs the romanised lyrics it was GIVEN (own / swap), plus vs the original's lyrics.
Reference rows: original audio and VAE round trip of the generated span.
usage: eval_cursor_lm.py --ckpt out/clm/cursor1/M_ema_s20000.pt --out out/clm_eval/cursor1_20k [--n 24]
"""
import argparse, json, sys, time
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE, LETTERS  # noqa: E402
from vae.data import Crops, romanise, encode_text, collate  # noqa: E402
from vae.train import edit  # noqa: E402
from vae.cursor_lm import CursorLM, windows, cursor_arrays, PAD, NULL  # noqa: E402
from vae.gated_nar import GatedCursorLM  # noqa: E402
from vae.align_letters import LetterAligner  # noqa: E402
from vae.graph_step import GraphedVelocity  # noqa: E402


def letters(ids): return "".join(LETTERS[i - 1] for i in ids)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--vae", default="", help="default: the VAE the LM was trained on (checkpoint args)")
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/slim_val.jsonl"))
    ap.add_argument("--transcriptor", default=str(ROOT / "models/HeartTranscriptor-oss"))
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--seg", type=float, default=24.0); ap.add_argument("--ctx", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=24); ap.add_argument("--adv-temp", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conds", default="own,swap,none")
    ap.add_argument("--causal-prompt", action="store_true", help="encode the prompt from its own audio only (the v4 latent sees the future through MERT)")
    ap.add_argument("--cfg", type=float, default=1.0, help="lyric CFG scale on the flow velocity (null = NULL windows + null advance); 1 = off")
    ap.add_argument("--slow", action="store_true", help="gated: old per-crop sampler (re-runs the prefix at every flow step)")
    ap.add_argument("--gen-bs", type=int, default=256, help="gated fast path: (crop, cond) rows per generation batch")
    ap.add_argument("--no-graph", action="store_true", help="gated fast path: eager step_velocity instead of a CUDA-graph replay")
    # cond "oracle" (cursor mode): own lyrics with the advance taken from the letter alignment of the real crop instead of sampled
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import random
    sd = torch.load(args.ckpt, map_location="cpu"); ta = sd["args"]
    if not args.vae: args.vae = ta["vae"]
    elif Path(args.vae).resolve() != Path(ta["vae"]).resolve(): print(f"WARNING: LM trained on {ta['vae']}, decoding with {args.vae}", flush=True)
    if args.vae == "msvae":                                                       # LM on MSVAE latents (tools/precompute_msvae.py)
        from vae.msvae import MSVAE; V = MSVAE(dev, amp=True); V.sem = False
    else:
        from vae.model import load_vae; V = load_vae(args.vae, dev)              # attaches MERT for sem (v4) VAEs
    ekw = dict(blk=ta["blk"], keep=ta.get("enc_keep", 2), win=ta.get("enc_win", 3.0), ahead=ta.get("enc_ahead", 0.64))
    enc_local = ta.get("enc") == "local" and V.sem                                # LM trained on bounded-lookahead latents
    encode = (lambda w: V.encode_local(w, **ekw)) if enc_local else V.encode
    LAT = 64
    if ta.get("asr_pca"):                                                         # LM predicts VAE 64 + Qwen3-ASR stream (decode: [0, 64))
        from vae.asr_stream import AsrStream; Q = AsrStream(pca=ta["asr_pca"], dev=dev); LAT = 64 + Q.dim; enc64 = encode
        def encode(w):
            mu = enc64(w)[0]
            return (torch.cat([mu, Q.local(w, V.sr, mu.shape[1], blk=ekw["blk"], keep=ekw["keep"], ahead=ekw["ahead"]).to(mu.dtype)], -1),)
    gated = ta.get("head", "mlp") == "gated"
    if ta.get("arch") == "liubai_h":                                              # vae/train_liubai_h.py (3B, bf16 export)
        from vae.gated_rwkv7 import LiuBaiH
        M = LiuBaiH(**(ta.get("tiny_cfg") or dict(lora=ta["lora"])), lattn=ta.get("lattn", False), iso=ta.get("iso", False), blk=ta["blk"])
    else: M = (GatedCursorLM(latent=LAT, blk=ta["blk"], d=ta["d"], n=ta["n"], lora=ta["lora"], iso=ta.get("iso", False), lattn=ta.get("lattn", False)) if gated else
         CursorLM(blk=ta["blk"], d=ta["d"], n=ta["n"], mode=ta["mode"], head_d=ta["head_d"], head_n=ta["head_n"]))
    M.load_state_dict(sd.get("ema", sd["M"])); M = M.to(dev, torch.bfloat16 if ta.get("arch") == "liubai_h" else torch.float32).eval(); lat_m, lat_s = sd["lat_m"].to(dev), sd["lat_s"].to(dev)
    blk = M.blk; rate = float(ta.get("rate", 12.5)); osr = getattr(V, "out_sr", V.sr)   # decoder output rate (MSVAE 48 kHz from 24 kHz)

    vd = Crops(args.manifest, holdout=0, seed=4242, seg=args.seg, with_lines=True, sr=V.sr, rate=rate); vd.rows = [r for r in vd.rows if len(r["lines"]) >= 4]
    random.seed(args.seed); it = iter(vd); items = []                              # seeded here, after load_vae (MERT init draws from random): same crops in every run
    g0 = int(round(args.ctx * rate)) // blk * blk                                   # first generated frame
    while len(items) < args.n:
        wav, tg, ok, cf, lines = next(it)
        own = [l for l in lines if l[0] >= g0 and l[2] <= args.seg * rate]          # lines sung completely in the generated span
        txt = romanise(" ".join(letters(l[1]) for l in own)).strip()
        if len(txt) >= 20: items.append((wav, own, txt))
    print(f"{len(items)} crops, prompt {g0 / rate:.1f} s, generate {(args.seg * rate - g0) / rate:.1f} s, mode {M.mode}", flush=True)

    @torch.no_grad()
    def sample_block(c, cn, gen):
        """M.sample_block with lyric CFG (cn = null condition or None); draws the same noise, so cfg 1 == M.sample_block"""
        if cn is None: return M.sample_block(c.float(), args.steps, gen)
        x = torch.randn(c.shape[0], M.D, device=dev, generator=gen); ts = torch.linspace(0, 1, args.steps + 1, device=dev)
        for j in range(args.steps):
            t = ts[j].expand(c.shape[0]); vc = M.head(x, t, c.float()).float(); vn = M.head(x, t, cn.float()).float()
            x = x + (ts[j + 1] - ts[j]) * (vn + args.cfg * (vc - vn))
        return x

    @torch.no_grad()
    def sample_gated(zc, W, ac, Wn, gen):
        """gated head: flow sample of block i = zc.shape[1] - 1 from the backbone's NAR rows; lyric CFG as a batch of 2
        (cond, null: NULL windows + null advance) when Wn is given; same noise draw as sample_block"""
        i = zc.shape[1] - 1; x = torch.randn(1, M.D, device=dev, generator=gen); ts = torch.linspace(0, 1, args.steps + 1, device=dev)
        WW = W if Wn is None else torch.cat([W, Wn]); nb = WW.shape[0]
        A = torch.full((nb, i + 1), M.max_adv, dtype=torch.long, device=dev); A[0, i] = ac
        for j in range(args.steps):
            v = M.velocity(zc.expand(nb, -1, -1), WW, A, x[:, None].expand(nb, 1, -1), ts[j].expand(nb, 1), [i])[1]; v = v[:, 0].float()
            x = x + (ts[j + 1] - ts[j]) * (v[:1] if Wn is None else v[1:] + args.cfg * (v[:1] - v[1:]))
        return x

    @torch.no_grad()
    def generate(z_ctx, ids, NB, gen, oracle=None):
        """z_ctx [1,nb_ctx,D] normalised prompt blocks; ids = letter ids (or None = null) -> z [1,NB,D]
        oracle = (c [NB], a [NB]) cursor / advance from the letter alignment (cursor mode): used instead of sampling"""
        nb0 = z_ctx.shape[1]; z = torch.zeros(1, NB + 1, M.D, device=dev); z[:, :nb0] = z_ctx
        c = 0; adv = []
        if M.mode == "prefix":
            P = torch.full((1, M.max_prefix), PAD, dtype=torch.long, device=dev)
            if ids: P[0, -len(ids[:M.max_prefix]):] = torch.tensor(ids[:M.max_prefix])
        for i in range(nb0, NB):
            cn = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if M.mode == "cursor":
                    W = np.zeros((1, i + 1, M.wb + M.wf), np.int64)
                    cs = list(oracle[0][:i + 1]) if oracle is not None else [0] * nb0 + [cc for cc in np.cumsum([0] + adv)[:i + 1 - nb0]]
                    W[0] = NULL if ids is None else windows(ids, np.asarray(cs), M.wb, M.wf)
                    h = M.hidden(z[:, :i + 1], torch.from_numpy(W).to(dev))[:, -1]
                    if ids is None: ac = torch.full((1,), M.max_adv, device=dev)
                    elif oracle is not None:
                        ac = torch.tensor([min(int(oracle[1][i]), M.max_adv - 1)], device=dev); adv.append(int(oracle[1][i]))
                    else:
                        p = torch.softmax(M.adv(h).float() / args.adv_temp, -1)
                        ac = torch.multinomial(p, 1, generator=gen)[:, 0]
                        ac = ac.clamp(max=max(0, len(ids) + 1 - cs[-1]))
                        adv.append(int(ac))
                    if gated:
                        Wt = torch.from_numpy(W).to(dev)
                        Wn = torch.full_like(Wt, NULL) if args.cfg != 1.0 and ids is not None else None
                        z[:, i] = sample_gated(z[:, :i + 1], Wt, int(ac), Wn, gen); continue
                    cnd = M.cond(h, ac.long())
                    if args.cfg != 1.0 and ids is not None:
                        Wn = torch.full((1, i + 1, M.wb + M.wf), NULL, dtype=torch.long, device=dev)
                        cn = M.cond(M.hidden(z[:, :i + 1], Wn)[:, -1], torch.full((1,), M.max_adv, device=dev))
                else:
                    h = M.hidden(z[:, :i + 1], P)[:, -1]; cnd = h
                z[:, i] = sample_block(cnd, cn, gen)
        return z[:, :NB], adv

    @torch.no_grad()
    def generate_batch(zp, idss, NB, gen, oracles):
        """gated head, all rows at once with the incremental state cache (GatedCursorLM.step_*): zp [R,nb0,D] prompts, idss
        R letter-id lists (None = null lyrics), oracles R (c, a) or None -> z [R,NB,D], R advance lists. Same model and CFG
        as generate(); only the noise draws differ (one batched draw per block instead of one per crop)"""
        R, nb0, D = zp.shape; cfg = args.cfg != 1.0; RR = 2 * R if cfg else R
        z = torch.zeros(R, NB, D, device=dev); z[:, :nb0] = zp; cache = M.cache_init(RR, dev)
        cs = np.zeros(R, np.int64); adv = [[] for _ in range(R)]; ts = torch.linspace(0, 1, args.steps + 1, device=dev)
        null = torch.tensor([ids is None for ids in idss], device=dev)[:, None]; nullw = np.full(M.wb + M.wf, NULL, np.int64)
        G = None; assert nb0 >= 1, "generation needs a prompt block (zprev)"
        two = (lambda x: torch.cat([x, x])) if cfg else (lambda x: x)
        for i in range(NB):
            for r, o in enumerate(oracles):
                if o is not None: cs[r] = o[0][i]
            W = torch.from_numpy(np.stack([nullw if ids is None else windows(ids, cs[r:r + 1], M.wb, M.wf)[0]
                                           for r, ids in enumerate(idss)])).to(dev)
            WW = torch.cat([W, torch.full_like(W, NULL)]) if cfg else W
            zprev = None if i == 0 else two(z[:, i - 1])
            with torch.autocast("cuda", dtype=torch.bfloat16): h = M.step_hidden(zprev, WW, cache)
            if i < nb0: continue
            with torch.autocast("cuda", dtype=torch.bfloat16): lg_a = M.adv(h[:R]).float()   # bf16 3B weights (fp32 208M unchanged)
            samp = torch.multinomial(torch.softmax(lg_a / args.adv_temp, -1), 1, generator=gen)[:, 0].tolist()
            ac = np.full(R, M.max_adv, np.int64)
            for r, ids in enumerate(idss):
                if ids is None: continue
                a = int(oracles[r][1][i]) if oracles[r] is not None else min(samp[r], max(0, len(ids) + 1 - int(cs[r])))
                ac[r] = min(a, M.max_adv - 1); adv[r].append(a)
                if oracles[r] is None: cs[r] += a
            A = torch.from_numpy(ac).to(dev); AA = torch.cat([A, torch.full_like(A, M.max_adv)]) if cfg else A
            x = torch.randn(R, D, device=dev, generator=gen)
            if not args.no_graph:                                                  # i >= nb0 >= 1, so zprev is a tensor
                if G is None: G = GraphedVelocity(M, cache, two(x), ts[0].expand(RR).contiguous(), WW, AA, zprev)
                else: G.load(cache, WW, AA, zprev)
            for j in range(args.steps):
                if G is not None: v = G(two(x), ts[j].expand(RR))
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        v = M.step_velocity(two(x), ts[j].expand(RR), WW, AA, cache, zprev).float()
                if cfg: v = torch.where(null, v[:R], v[R:] + args.cfg * (v[:R] - v[R:]))
                x = x + (ts[j + 1] - ts[j]) * v
            z[:, i] = x
        return z, adv

    @torch.no_grad()
    def decode(zn):
        z = (zn.reshape(1, -1, LAT) * lat_s + lat_m)[..., :64]
        with torch.autocast("cuda", dtype=torch.bfloat16): y = V.decode(z.float()).float()
        return y[0].cpu().numpy().T

    rows = []; gen = torch.Generator(device=dev).manual_seed(args.seed); conds = args.conds.split(","); LA = None
    assert "oracle" not in conds or M.mode == "cursor", "oracle timing needs a cursor-mode LM"
    fast = gated and hasattr(M, "step_hidden") and not args.slow; jobs = []
    for k, (wav, own, txt) in enumerate(items):
        x = wav[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): mu = encode(x)[0].float()
        NB = mu.shape[1] // blk; zn = ((mu[:, :NB * blk] - lat_m) / lat_s).reshape(1, NB, -1); nb0 = g0 // blk
        d = out / f"{k:03d}"; d.mkdir(exist_ok=True); S0 = g0 * V.hop
        sf.write(d / "A_original.flac", wav.numpy().T[S0:], V.sr)
        S0o = S0 * osr // V.sr; sf.write(d / "B_vae.flac", decode(zn)[S0o:], osr)
        zp = zn[:, :nb0]
        if args.causal_prompt:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): mp = encode(x[..., :S0])[0].float()
            assert mp.shape[1] >= g0, (mp.shape, g0); zp = ((mp[:, :g0] - lat_m) / lat_s).reshape(1, nb0, -1)
        other = items[(k + 1) % len(items)]
        given = dict(own=txt, swap=other[2], none="")
        row = dict(k=k, own=txt, swap=other[2])
        if "oracle" in conds:                                                       # letter alignment of the crop's own lines
            if LA is None: LA = LetterAligner(dev, sr=V.sr)
            o_ids, o_st, _ = LA(x, [own], NB * blk)[0]; oracle = cursor_arrays(o_st, NB, blk); row["oracle_letters"] = len(o_ids)
        for cnd in conds:
            if cnd == "oracle": ids, orc = o_ids, oracle
            else:
                ids = None if cnd == "none" and M.mode == "cursor" else ([] if cnd == "none" else encode_text(given[cnd])); orc = None
            if fast: jobs.append((k, cnd, zp, ids, orc, NB, S0o)); continue
            zg, adv = generate(zp, ids, NB, gen, orc)
            sf.write(d / f"C_{cnd}.flac", decode(zg)[S0o:], osr); row[f"adv_{cnd}"] = int(sum(adv))
        rows.append(row)
        if not fast: print(k, {c: row.get(f"adv_{c}") for c in conds}, "letters", len(txt), flush=True)
    for q in range(0, len(jobs), args.gen_bs):                                     # fast path: batched incremental generation
        jb = jobs[q:q + args.gen_bs]; assert len({j[5] for j in jb}) == 1, "batched generation needs equal crop lengths"; t0 = time.time()
        zg, advs = generate_batch(torch.cat([j[2] for j in jb]), [j[3] for j in jb], jb[0][5], gen, [j[4] for j in jb])
        for (k, cnd, _, _, _, _, S0), z1, a1 in zip(jb, zg, advs):
            sf.write(out / f"{k:03d}" / f"C_{cnd}.flac", decode(z1[None])[S0:], osr); rows[k][f"adv_{cnd}"] = int(sum(a1))
        print(f"generated {q + len(jb)} / {len(jobs)} ({time.time() - t0:.0f} s incl. decode)", flush=True)

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(args.transcriptor)
    asr = WhisperForConditionalGeneration.from_pretrained(args.transcriptor, dtype=torch.float16).to(dev).eval()

    def transcribe(p):
        y, sr = sf.read(p, dtype="float32", always_2d=True); y = y.mean(1); g = gcd(sr, 16000); y = resample_poly(y, 16000 // g, sr // g)
        f = proc(y[:16000 * 30], sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
        with torch.inference_mode(): ids = asr.generate(f, task="transcribe", max_new_tokens=220)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def cer(hyp, ref):
        h, r = romanise(hyp).replace(" ", ""), ref.replace(" ", "")
        return edit(list(h), list(r)) / max(1, len(r))

    res = {}
    for row in rows:
        d = out / f"{row['k']:03d}"
        for name in ["A_original", "B_vae"] + [f"C_{c}" for c in conds]:
            t = transcribe(d / f"{name}.flac"); row[f"tr_{name}"] = t
            row[f"cer_{name}"] = cer(t, row["own"])
            if name == "C_swap": row["cer_C_swap_given"] = cer(t, row["swap"])
        for kk, v in row.items():
            if kk.startswith("cer_"): res.setdefault(kk, []).append(v)
    summ = {k: float(np.mean(v)) for k, v in res.items()}
    json.dump(dict(summary=summ, rows=rows, args=vars(args)), open(out / "eval.json", "w"), indent=1, ensure_ascii=False)
    print("letter CER vs the crop's own lyrics (C_swap_given: vs the swapped lyrics it was given)")
    for k_, v in summ.items(): print(f"  {k_:22s} {v:.3f}")


if __name__ == "__main__":
    main()
