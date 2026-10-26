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
import argparse, json, sys
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE, LETTERS  # noqa: E402
from vae.data import Crops, romanise, encode_text, collate  # noqa: E402
from vae.train import edit  # noqa: E402
from vae.cursor_lm import CursorLM, windows, cursor_arrays, PAD, NULL  # noqa: E402
from vae.align_letters import LetterAligner  # noqa: E402


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
    # cond "oracle" (cursor mode): own lyrics with the advance taken from the letter alignment of the real crop instead of sampled
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import random
    sd = torch.load(args.ckpt, map_location="cpu"); ta = sd["args"]
    if not args.vae: args.vae = ta["vae"]
    elif Path(args.vae).resolve() != Path(ta["vae"]).resolve(): print(f"WARNING: LM trained on {ta['vae']}, decoding with {args.vae}", flush=True)
    from vae.model import load_vae; V = load_vae(args.vae, dev)                  # attaches MERT for sem (v4) VAEs
    ekw = dict(blk=ta["blk"], keep=ta.get("enc_keep", 2), win=ta.get("enc_win", 3.0), ahead=ta.get("enc_ahead", 0.64))
    enc_local = ta.get("enc") == "local" and V.sem                                # LM trained on bounded-lookahead latents
    encode = (lambda w: V.encode_local(w, **ekw)) if enc_local else V.encode
    M = CursorLM(blk=ta["blk"], d=ta["d"], n=ta["n"], mode=ta["mode"], head_d=ta["head_d"], head_n=ta["head_n"])
    M.load_state_dict(sd.get("ema", sd["M"])); M = M.to(dev).eval(); lat_m, lat_s = sd["lat_m"].to(dev), sd["lat_s"].to(dev)
    blk = M.blk; rate = 12.5

    vd = Crops(args.manifest, holdout=0, seed=4242, seg=args.seg, with_lines=True); vd.rows = [r for r in vd.rows if len(r["lines"]) >= 4]
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
                    cnd = M.cond(h, ac.long())
                    if args.cfg != 1.0 and ids is not None:
                        Wn = torch.full((1, i + 1, M.wb + M.wf), NULL, dtype=torch.long, device=dev)
                        cn = M.cond(M.hidden(z[:, :i + 1], Wn)[:, -1], torch.full((1,), M.max_adv, device=dev))
                else:
                    h = M.hidden(z[:, :i + 1], P)[:, -1]; cnd = h
                z[:, i] = sample_block(cnd, cn, gen)
        return z[:, :NB], adv

    @torch.no_grad()
    def decode(zn):
        z = zn.reshape(1, -1, 64) * lat_s + lat_m
        with torch.autocast("cuda", dtype=torch.bfloat16): y = V.decode(z.float()).float()
        return y[0].cpu().numpy().T

    rows = []; gen = torch.Generator(device=dev).manual_seed(args.seed); conds = args.conds.split(","); LA = None
    assert "oracle" not in conds or M.mode == "cursor", "oracle timing needs a cursor-mode LM"
    for k, (wav, own, txt) in enumerate(items):
        x = wav[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): mu = encode(x)[0].float()
        NB = mu.shape[1] // blk; zn = ((mu[:, :NB * blk] - lat_m) / lat_s).reshape(1, NB, -1); nb0 = g0 // blk
        d = out / f"{k:03d}"; d.mkdir(exist_ok=True); S0 = g0 * V.hop
        sf.write(d / "A_original.flac", wav.numpy().T[S0:], 24000)
        sf.write(d / "B_vae.flac", decode(zn)[S0:], 24000)
        zp = zn[:, :nb0]
        if args.causal_prompt:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): mp = encode(x[..., :S0])[0].float()
            assert mp.shape[1] >= g0, (mp.shape, g0); zp = ((mp[:, :g0] - lat_m) / lat_s).reshape(1, nb0, -1)
        other = items[(k + 1) % len(items)]
        given = dict(own=txt, swap=other[2], none="")
        row = dict(k=k, own=txt, swap=other[2])
        if "oracle" in conds:                                                       # letter alignment of the crop's own lines
            if LA is None: LA = LetterAligner(dev)
            o_ids, o_st, _ = LA(x, [own], NB * blk)[0]; oracle = cursor_arrays(o_st, NB, blk); row["oracle_letters"] = len(o_ids)
        for cnd in conds:
            if cnd == "oracle": zg, adv = generate(zp, o_ids, NB, gen, oracle)
            else:
                ids = None if cnd == "none" and M.mode == "cursor" else ([] if cnd == "none" else encode_text(given[cnd]))
                zg, adv = generate(zp, ids, NB, gen)
            sf.write(d / f"C_{cnd}.flac", decode(zg)[S0:], 24000); row[f"adv_{cnd}"] = int(sum(adv))
        rows.append(row); print(k, {c: row.get(f"adv_{c}") for c in conds}, "letters", len(txt), flush=True)

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
