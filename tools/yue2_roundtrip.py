#!/usr/bin/env python3
"""G0 gate (PLAN §8.4): round-trip real Suno songs through the YuE2 stack.

    mp3 (from a suno-94k shard tar) -> 24 kHz mono -> MERT-v2-FullSong layer-20 -> instance-norm
    -> Mothersuperior v4 head -> YuE2 semantic tokens (25 Hz)
    -> YuE2 NAR (flow matching, optional v4 NAR LoRA folded in) with a cot=off prefix from the song's own
       caption + lyrics -> [T,64] latents -> YuE2-Vae decode -> 48 kHz stereo flac

Also writes, per song: the original resampled to 48 kHz (A), the VAE-only round trip encode->decode (B, the
renderer ceiling), the token round trip (C), tokens .npy, and latent MSE(NAR latents, VAE.encode(original)).
Run inside venvs/yue2 (torch 2.10 + transformers 4.57.6). Stages run sequentially so peak GPU memory stays
under ~12 GiB (the stage-1 trainer occupies 25-27 GiB on every GPU).

usage: yue2_roundtrip.py --n 20 [--shard 0] [--nar-lora models/yue2/mothersuperior_v4/nar_lora_joint_v4.pt|none]
                          [--steps 32] [--out out/yue2_roundtrip] [--max-sec 200] [--ids id1,id2,...]
"""
import argparse, io, json, os, sys, tarfile, time, hashlib
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn as nn
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
YUE2 = ROOT / "models/yue2"
HEAD_PT = YUE2 / "mothersuperior_v4/tokenizer_head_joint_v4.pt"
VOCAB, WIN, D, L, H = 32768, 512, 512, 8, 8   # head hyper-parameters (train_v2.py)


class Tok(nn.Module):
    def __init__(s, din=1024):
        super().__init__(); s.inp = nn.Linear(din, D); s.pos = nn.Parameter(torch.zeros(1, WIN, D))
        layer = nn.TransformerEncoderLayer(D, H, 4 * D, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        s.enc = nn.TransformerEncoder(layer, L); s.norm = nn.LayerNorm(D); s.head = nn.Linear(D, VOCAB)
    def forward(s, x): return s.head(s.norm(s.enc(s.inp(x) + s.pos[:, :x.shape[1]])))


def instnorm(x):
    x = x.astype(np.float32); return (x - x.mean(0)) / (x.std(0) + 1e-5)


def pick_songs(shard, n, max_sec, ids):
    meta = [json.loads(l) for l in open(ROOT / f"data/meta/suno94k/shard-{shard:05d}.jsonl")]
    by_id = {m["id"]: m for m in meta}
    if ids:
        return [by_id[i] for i in ids]
    ok = [m for m in meta if not m.get("error") and 60 <= m["duration"] <= max_sec
          and len(m.get("lyrics", "")) > 200 and "[instrumental]" not in m["lyrics"].lower()
          and all(ord(c) < 0x250 for c in m["lyrics"])]            # Latin-script lyrics: PER via English ASR later
    ok.sort(key=lambda m: hashlib.md5(m["id"].encode()).hexdigest())
    return ok[:n]


def load_audio(tar, clip_id):
    for suffix in (".mp3", ".wav", ".flac"):
        try:
            data = tar.extractfile(f"{clip_id}{suffix}").read(); break
        except KeyError:
            continue
    else:
        raise KeyError(clip_id)
    a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if a.shape[1] == 1: a = np.repeat(a, 2, 1)
    g = gcd(sr, 48000); st48 = resample_poly(a, 48000 // g, sr // g, axis=0).astype(np.float32) if sr != 48000 else a
    g = gcd(sr, 24000); m24 = resample_poly(a.mean(1), 24000 // g, sr // g).astype(np.float32)
    return st48, m24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--ids", default=""); ap.add_argument("--max-sec", type=float, default=200)
    ap.add_argument("--nar-lora", default=str(YUE2 / "mothersuperior_v4/nar_lora_joint_v4.pt"))
    ap.add_argument("--steps", type=int, default=32); ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=str(ROOT / "out/yue2_roundtrip")); ap.add_argument("--vae", default="YuE2-Vae")
    ap.add_argument("--skip-b", action="store_true", help="skip the VAE-only round trip")
    ap.add_argument("--minted", default="", help="dir of yue2_mint.py outputs: round-trip those songs instead of the shard tar")
    ap.add_argument("--true-tokens", action="store_true", help="with --minted: render from the stored true tokens instead of the head")
    ap.add_argument("--head-ckpt", default="", help="our R1 checkpoint (scripts/train_inverse_tokenizer.py) instead of the v4 head")
    ap.add_argument("--tag", default="", help="extra tag for output file names")
    args = ap.parse_args()
    dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = ("lora" if args.nar_lora != "none" else "base") + ("_true" if args.true_tokens else "") + ("_r1" if args.head_ckpt else "") + (f"_{args.tag}" if args.tag else "")
    r1_layers = [20]
    if args.head_ckpt:
        sys.path.insert(0, str(ROOT / "scripts")); from train_inverse_tokenizer import InverseTokenizer, predict as r1_predict
        r1_ck = torch.load(args.head_ckpt, map_location=dev); r1_layers = r1_ck["layers"]
        r1 = InverseTokenizer(**r1_ck["cfg"]).to(dev); r1.load_state_dict(r1_ck["model"]); r1.eval()
        print(f"R1 head {args.head_ckpt} layers {r1_layers} step {r1_ck.get('step')} val top1 {r1_ck.get('top1')}", flush=True)
    if args.minted:
        songs = []
        for d in sorted(Path(args.minted).iterdir()):
            if (d / "latent.npy").exists():
                r = json.load(open(d / "request.json"))
                songs.append(dict(id=r["id"], title=r["source_title"], duration=r["audio_seconds"], caption=r["style"], lyrics=r["lyrics"],
                                  audio_path=d / "audio.flac", true_tokens=np.load(d / "semantic.npy")))
        songs = songs[:args.n]; tar = None
    else:
        songs = pick_songs(args.shard, args.n, args.max_sec, [s for s in args.ids.split(",") if s])
        tar = tarfile.open(ROOT / f"data/raw/suno94k/suno-various-94k-{args.shard:05d}.tar")
    print(f"{len(songs)} songs, NAR={tag}, steps={args.steps}", flush=True)
    t0 = time.time()

    # ---- stage 1: audio + MERT features + VAE encode (encoder needs decoder_only=False) ----
    from transformers import AutoModel, AutoFeatureExtractor
    from yue2.modeling_vae import YuE2VAE
    proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
    mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()
    vae = YuE2VAE.from_pretrained(str(YUE2 / args.vae), decoder_only=False, device=dev, local_files_only=True)
    items = []
    for m in songs:
        sid = m["id"]; d = out / sid; d.mkdir(exist_ok=True)
        if tar is None:
            a, sr = sf.read(m["audio_path"], dtype="float32", always_2d=True)
            if a.shape[1] == 1: a = np.repeat(a, 2, 1)
            g = gcd(sr, 48000); st48 = resample_poly(a, 48000 // g, sr // g, axis=0).astype(np.float32) if sr != 48000 else a
            g = gcd(sr, 24000); m24 = resample_poly(a.mean(1), 24000 // g, sr // g).astype(np.float32)
        else:
            st48, m24 = load_audio(tar, sid)
        # MERT L20, 30 s chunks batched, linear-resampled to exactly 25 Hz (prep_real.py recipe)
        CH = 24000 * 30; chunks = [m24[s:s + CH] for s in range(0, len(m24), CH)]
        chunks = [c for c in chunks if len(c) >= 24000]
        full = [c for c in chunks if len(c) == CH]; tail = [c for c in chunks if len(c) < CH]; feats = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for group in ([full] if full else []) + [[c] for c in tail]:
                inp = {k: v.to(dev) for k, v in proc(group, sampling_rate=24000, return_tensors="pt").items()}
                hs = mert(**inp, output_hidden_states=True).hidden_states
                feats.append(torch.stack([hs[l].reshape(-1, 1024) for l in sorted(set([20] + r1_layers))]))
        Hm = torch.cat(feats, 1).float(); T25 = int(round(len(m24) / 24000 * 25))
        Mk = torch.nn.functional.interpolate(Hm.transpose(1, 2), size=T25, mode="linear", align_corners=False).transpose(1, 2).half().cpu().numpy()
        klist = sorted(set([20] + r1_layers)); M = Mk[klist.index(20)]; M_r1 = Mk[[klist.index(l) for l in r1_layers]]
        # true VAE latents of the original, 60 s segments
        zs = []
        with torch.inference_mode():
            for s in range(0, len(st48), 48000 * 60):
                seg = st48[s:s + 48000 * 60]
                if len(seg) < 1920: break
                zs.append(vae.encode(torch.tensor(seg.T[None], device=dev))[0].T.float().cpu())
        Z = torch.cat(zs, 0).numpy(); n = min(len(M), len(Z)); M, Z = M[:n], Z[:n]
        np.save(d / "mert_l20.npy", M); np.save(d / "latent_true.npy", Z.astype(np.float16))
        if not (d / "A_original.flac").exists():
            sf.write(d / "A_original.flac", st48, 48000, subtype="PCM_16")
        if not args.skip_b and not (d / "B_vae_only.flac").exists():
            with torch.inference_mode():
                audio = vae.decode_tiled(torch.tensor(Z.T[None]).contiguous(), core_frames=750, halo_frames=16, output_device="cpu")
            sf.write(d / "B_vae_only.flac", audio[0].float().clamp(-1, 1).T.numpy(), 48000, subtype="PCM_16")
        items.append(dict(meta=m, mert=M, mert_r1=M_r1[:, :n], lat=Z, dir=d))
        print(f"  [{len(items)}/{len(songs)}] {sid} {m['duration']:.0f}s frames {n} mert-lat diff {len(M) - len(Z)} {time.time() - t0:.0f}s", flush=True)
    del mert, vae; torch.cuda.empty_cache()

    # ---- stage 2: head -> tokens ----
    head = Tok().to(dev); head.load_state_dict(torch.load(HEAD_PT, map_location=dev)["model"]); head.eval()

    @torch.no_grad()
    def predict(x):
        T = len(x); res = np.zeros(T, dtype=np.int64); starts = list(range(0, max(1, T - WIN + 1), WIN // 2))
        if starts[-1] + WIN < T: starts.append(max(0, T - WIN))
        for s0 in starts:
            xw = x[s0:s0 + WIN]; n = len(xw)
            if n < WIN: xw = np.pad(xw, ((0, WIN - n), (0, 0)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(torch.tensor(xw[None], device=dev))[0, :n].float().argmax(-1).cpu().numpy()
            lo = s0 + (0 if s0 == 0 else WIN // 4); hi = s0 + n - (0 if s0 + n >= T else WIN // 4)
            res[lo:hi] = pred[lo - s0:hi - s0]
        return res
    for it in items:
        if args.head_ckpt:
            toks = r1_predict(r1, it["mert_r1"], dev); np.save(it["dir"] / f"tokens_r1{('_' + args.tag) if args.tag else ''}.npy", toks.astype(np.int32))
        else:
            toks = predict(instnorm(it["mert"])); np.save(it["dir"] / "tokens_v4.npy", toks.astype(np.int32))
        if "true_tokens" in it["meta"]:
            tt = it["meta"]["true_tokens"].astype(np.int64); n = min(len(tt), len(toks))
            it["head_top1"] = float((tt[:n] == toks[:n]).mean())
            print(f"  head top-1 vs true tokens {it['meta']['id']}: {it['head_top1']:.3f} over {n} frames", flush=True)
            if args.true_tokens: toks = tt
        it["tokens"] = toks
        it["tok_stats"] = dict(unique=len(set(toks.tolist())) / len(toks), repeat=float((toks[1:] == toks[:-1]).mean()))
    del head; torch.cuda.empty_cache()
    print(f"tokens done {time.time() - t0:.0f}s", flush=True)

    # ---- stage 3: NAR (+LoRA) -> latents ----
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.protocol import SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from yue2.nar import synthesize
    snap = str(YUE2 / "YuE2-3B")
    model = YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(dev)
    model.requires_grad_(False); bb = model.model; tok = YuE2TextTokenizer(snap + "/qwen.tiktoken")
    if args.nar_lora != "none":
        ck = torch.load(args.nar_lora, map_location=dev); it_ = iter(ck["lora"]); n_merged = 0
        with torch.no_grad():
            for layer in bb.layers:
                for mod, names in ((layer.nar_self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.nar_mlp, ("gate_proj", "up_proj", "down_proj"))):
                    for nme in names:
                        A = next(it_).to(dev).float(); B = next(it_).to(dev).float(); lin = getattr(mod, nme)
                        lin.weight.add_((B @ A).to(lin.weight.dtype)); n_merged += 1
            model.vae2llm.load_state_dict({k: v.to(torch.bfloat16) for k, v in ck["io"]["vae2llm"].items()})
            model.llm2vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in ck["io"]["llm2vae"].items()})
        print(f"merged {n_merged} NAR LoRA linears + vae2llm/llm2vae", flush=True)
    for it in items:
        m = it["meta"]; style = " ".join(m["caption"].split())[:1500]; lyrics = m["lyrics"].strip() or "[instrumental]"
        prefix = token_prefixes(SongRequest(style=style, lyrics=lyrics, cot="off", seed=1, id=m["id"]), tok)
        with torch.inference_mode():
            z = synthesize(model, prefix, [int(v) for v in it["tokens"]], args.seed, steps=args.steps).float().cpu().numpy()
        it["lat_nar"] = z; np.save(it["dir"] / f"latent_nar_{tag}.npy", z.astype(np.float16))
        n = min(len(z), len(it["lat"])); it["mse"] = float(((z[:n] - it["lat"][:n]) ** 2).mean()); it["var"] = float(it["lat"][:n].var())
        print(f"  NAR {m['id']} prefix {len(prefix)} tokens {len(it['tokens'])} mse {it['mse']:.4f} (latent var {it['var']:.4f}) {time.time() - t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()

    # ---- stage 4: decode ----
    from yue2.modeling_vae import YuE2VAE
    vae = YuE2VAE.from_pretrained(str(YUE2 / args.vae), decoder_only=True, device=dev, local_files_only=True)
    report = []
    for it in items:
        with torch.inference_mode():
            audio = vae.decode_tiled(torch.tensor(it["lat_nar"].T[None]).contiguous(), core_frames=750, halo_frames=16, output_device="cpu")
        sf.write(it["dir"] / f"C_roundtrip_{tag}_s{args.steps}.flac", audio[0].float().clamp(-1, 1).T.numpy(), 48000, subtype="PCM_16")
        m = it["meta"]
        report.append(dict(id=m["id"], title=m["title"], duration=m["duration"], caption=m["caption"], frames=len(it["tokens"]),
                           tok_unique=it["tok_stats"]["unique"], tok_repeat=it["tok_stats"]["repeat"], latent_mse=it["mse"], latent_var=it["var"]))
    json.dump(dict(nar=tag, steps=args.steps, seed=args.seed, vae=args.vae, songs=report), open(out / f"report_{tag}_s{args.steps}.json", "w"), indent=1)
    mses = [r["latent_mse"] for r in report]
    print(f"DONE {len(report)} songs in {time.time() - t0:.0f}s | latent MSE mean {np.mean(mses):.4f} median {np.median(mses):.4f} | "
          f"token repeat mean {np.mean([r['tok_repeat'] for r in report]):.3f} | files under {out}", flush=True)


if __name__ == "__main__":
    main()
