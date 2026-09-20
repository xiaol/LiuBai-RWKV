#!/usr/bin/env python3
"""Generate YuE2 semantic tokens with the R3 stage-1 RWKV-7 model (PLAN §8.3 R3 / §8.6 G3 gate).

Prompt (scripts/yue2_layout.py):  world_tokens("[Genre] caption\n[Lyrics]\nlyrics\n")  EOD  SOA  YUE2CODEC
then sample codes (YUE2_BASE + c) until EOA or --max-frames. Sampling is restricted to the 32,768 codes + EOA.
GPT-mode prefill with the wkv7s CUDA kernel from RWKV-LM/RWKV-v7 (fp16), RNN-mode decode one token at a time.
Prompts come from a suno-94k meta shard (default: test shard 2, same 20 ids as out/yue2_roundtrip_s2 if --ids-from
is given) so the result can be rendered next to the real Suno song and scored with tools/yue2_roundtrip_eval.py.
Output: <out>/<id>/{semantic.npy (int32 codes), request.json (style, lyrics, sampling, timing)}.
Run inside the rwkv_py312 env (source /root/envs/env_rwkv) on one GPU.
usage: rwkv_generate.py --ckpt out/stage1_yue2/rwkv-final.pth --out out/gen/final_s2 --shard 2 --ids-from out/yue2_roundtrip_s2
         [--n 20] [--temperature 1.0] [--top-p 0.95] [--top-k 0] [--max-frames 6000] [--seed 4242] [--part 0 --nparts 1]
"""
import argparse, json, os, sys, time, types
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tools/RWKV-LM/RWKV-v5/tokenizer"))
from yue2_layout import EOD, SOA, EOA, YUE2CODEC, SOS, YUE2_BASE, N_CODES, VOCAB_SIZE  # noqa: E402
from build_yue2_sec_binidx import text_units  # noqa: E402
from build_stage1_manifest import prompt_text, is_instrumental, VOCAB_TXT  # noqa: E402
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402

DTYPE = torch.half
HEAD_SIZE = 64
CUDA_DIR = ROOT / "tools/RWKV-LM/RWKV-v7/cuda"


def load_kernel():
    from torch.utils.cpp_extension import load
    load(name="wkv7s", sources=[str(CUDA_DIR / "wkv7s_op.cpp"), str(CUDA_DIR / "wkv7s.cu")], is_python_module=False, verbose=False,
         extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])


def wkv7s(state, r, w, k, v, a, b):
    T, C = r.shape; H = C // HEAD_SIZE
    y = torch.empty((T, C), device=k.device, dtype=DTYPE)
    torch.ops.wkv7s.forward(1, T, C, H, state, r.contiguous(), w.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous(), y)
    return y


class RWKV7:
    """Weights as in rwkv_v7_demo_fast.py (transposed matrices, ln0 folded into emb); seq prefill + one-step decode."""
    def __init__(s, path, dev="cuda"):
        z = torch.load(path, map_location=dev, mmap=True, weights_only=True); z = {k: v for k, v in z.items()}
        s.n_layer = max(int(k.split(".")[1]) for k in z if k.startswith("blocks.")) + 1
        s.n_embd = z["emb.weight"].shape[1]; s.vocab = z["emb.weight"].shape[0]
        s.n_head, s.head_size = z["blocks.0.att.r_k"].shape; assert s.head_size == HEAD_SIZE
        for k in list(z.keys()):
            if any(t in k for t in ("key.weight", "value.weight", "receptance.weight", "output.weight", "head.weight")): z[k] = z[k].t()
            z[k] = z[k].squeeze().to(DTYPE).contiguous()
            if k.endswith("att.r_k"): z[k] = z[k].flatten()
        z["emb.weight"] = F.layer_norm(z["emb.weight"], (s.n_embd,), weight=z["blocks.0.ln0.weight"], bias=z["blocks.0.ln0.bias"])
        for n in ("v0", "v1", "v2"):
            if f"blocks.0.att.{n}" not in z: z[f"blocks.0.att.{n}"] = z[f"blocks.0.att.a{n[1]}"]
        s.z = z; s.dev = dev

    def new_state(s):
        st = []
        for _ in range(s.n_layer):
            st += [torch.zeros(s.n_embd, dtype=DTYPE, device=s.dev), torch.zeros((s.n_head, HEAD_SIZE, HEAD_SIZE), dtype=torch.float, device=s.dev), torch.zeros(s.n_embd, dtype=DTYPE, device=s.dev)]
        return st

    @torch.no_grad()
    def tmix(s, i, x, x_prev, v_first, state, seq):
        z = s.z; att = f"blocks.{i}.att."; H, N = s.n_head, HEAD_SIZE
        if seq: T = x.shape[0]; xx = torch.cat((x_prev.unsqueeze(0), x[:-1])) - x
        else: xx = x_prev - x
        xr, xw, xk, xv, xa, xg = (x + xx * z[att + n] for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))
        r = xr @ z[att + "receptance.weight"]; w = torch.tanh(xw @ z[att + "w1"]) @ z[att + "w2"]; k = xk @ z[att + "key.weight"]; v = xv @ z[att + "value.weight"]
        a = torch.sigmoid(z[att + "a0"] + (xa @ z[att + "a1"]) @ z[att + "a2"]); g = torch.sigmoid(xg @ z[att + "g1"]) @ z[att + "g2"]
        shp = (T, H, N) if seq else (H, N)
        kk = F.normalize((k * z[att + "k_k"]).view(*shp), dim=-1, p=2.0).view(*k.shape); k = k * (1 + (a - 1) * z[att + "k_a"])
        if i == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(z[att + "v0"] + (xv @ z[att + "v1"]) @ z[att + "v2"])
        if seq:
            w = -F.softplus(-(z[att + "w0"] + w)) - 0.5
            out = wkv7s(state, r, w, k, v, -kk, kk * a)
        else:
            w = torch.exp(-0.606531 * torch.sigmoid((z[att + "w0"] + w).float()))
            vk = v.view(H, N, 1) @ k.view(H, 1, N); ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
            state.copy_(state * w.view(H, 1, N) + state @ ab.float() + vk.float())
            out = (state.to(DTYPE) @ r.view(H, N, 1)).view(H * N)
        rows = T if seq else 1
        out = F.group_norm(out.view(rows, H * N), num_groups=H, weight=z[att + "ln_x.weight"], bias=z[att + "ln_x.bias"], eps=64e-5).view(*shp[:-2], H * N) if seq else \
              F.group_norm(out.view(1, H * N), num_groups=H, weight=z[att + "ln_x.weight"], bias=z[att + "ln_x.bias"], eps=64e-5).view(H * N)
        out = out + ((r * k * z[att + "r_k"]).view(*shp).sum(dim=-1, keepdim=True) * v.view(*shp)).view(*k.shape)
        return (out * g) @ z[att + "output.weight"], (x[-1] if seq else x), v_first

    @torch.no_grad()
    def forward(s, idx, state):
        """idx: list[int] (len>1 = GPT-mode prefill via CUDA kernel, len 1 = RNN step). Returns logits of the last position."""
        z = s.z; seq = len(idx) > 1
        x = z["emb.weight"][torch.tensor(idx, device=s.dev)] if seq else z["emb.weight"][idx[0]]
        v_first = torch.empty_like(x)
        for i in range(s.n_layer):
            b = f"blocks.{i}."
            xx = F.layer_norm(x, (s.n_embd,), weight=z[b + "ln1.weight"], bias=z[b + "ln1.bias"])
            xx, state[i * 3], v_first = s.tmix(i, xx, state[i * 3], v_first, state[i * 3 + 1], seq)
            x = x + xx
            xx = F.layer_norm(x, (s.n_embd,), weight=z[b + "ln2.weight"], bias=z[b + "ln2.bias"])
            if seq: dx = torch.cat((state[i * 3 + 2].unsqueeze(0), xx[:-1])) - xx
            else: dx = state[i * 3 + 2] - xx
            kx = xx + dx * z[b + "ffn.x_k"]; state[i * 3 + 2] = xx[-1] if seq else xx
            x = x + (torch.relu(kx @ z[b + "ffn.key.weight"]) ** 2) @ z[b + "ffn.value.weight"]
        x = x[-1] if seq else x
        x = F.layer_norm(x, (s.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
        return (x @ z["head.weight"]).float()


def sample(logits, mask, temperature, top_p, top_k, gen):
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4).masked_fill(~mask, float("-inf"))
    if temperature != 1.0: logits = logits / temperature
    probs = torch.softmax(logits, -1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0: return int(torch.argmax(logits).item())   # fp16 overflow guard
    if top_k > 0:
        thr = torch.topk(probs, top_k).values[-1]; probs = torch.where(probs >= thr, probs, torch.zeros_like(probs))
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True); cs = torch.cumsum(sp, -1)
        keep = cs - sp < top_p; sp = torch.where(keep, sp, torch.zeros_like(sp)); probs = torch.zeros_like(probs).scatter_(0, si, sp)
    probs = probs / probs.sum()
    if not torch.isfinite(probs).all() or probs.sum() <= 0: return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, 1, generator=gen).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "out/stage1_yue2/rwkv-final.pth")); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=2); ap.add_argument("--n", type=int, default=20); ap.add_argument("--ids", default="")
    ap.add_argument("--ids-from", default="", help="directory whose sub-dir names are the song ids to prompt with (e.g. out/yue2_roundtrip_s2)")
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=6000); ap.add_argument("--min-frames", type=int, default=250); ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1); ap.add_argument("--tag", default="")
    ap.add_argument("--sections", type=int, default=0, help="1 = R3.1 section-interleaved decoding: header EOD, then per lyric section SOS text SOA YUE2CODEC codes... EOA")
    ap.add_argument("--min-sec-frames", type=int, default=50); ap.add_argument("--max-sec-frames", type=int, default=1500)
    ap.add_argument("--unit", default="stanza", choices=["stanza", "line"], help="section granularity for --sections 1 (must match the training corpus)")
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = {m["id"]: m for m in (json.loads(l) for l in open(ROOT / f"data/meta/suno94k/shard-{args.shard:05d}.jsonl"))}
    if args.ids: ids = [i for i in args.ids.split(",") if i]
    elif args.ids_from: ids = sorted(d.name for d in Path(args.ids_from).iterdir() if d.is_dir() and d.name in meta)
    else: ids = [m["id"] for m in meta.values() if not m.get("error") and 30 <= m["duration"] <= 240][:args.n]
    ids = ids[:args.n]; ids = [i for j, i in enumerate(ids) if j % args.nparts == args.part]
    ids = [i for i in ids if not (out / i / "semantic.npy").exists()]
    print(f"{len(ids)} prompts, ckpt {args.ckpt}, T={args.temperature} top_p={args.top_p} top_k={args.top_k} max {args.max_frames} frames", flush=True)
    load_kernel(); model = RWKV7(args.ckpt, dev); tok = TRIE_TOKENIZER(str(VOCAB_TXT))
    assert model.vocab == VOCAB_SIZE, model.vocab
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=dev); mask[YUE2_BASE:YUE2_BASE + N_CODES] = True; mask_eoa = mask.clone(); mask_eoa[EOA] = True
    for n, sid in enumerate(ids):
        m = meta[sid]; lyrics = m.get("lyrics", "") or ""; inst = is_instrumental(lyrics)
        text = prompt_text(m.get("caption", "") or "", "[Instrumental]" if inst else lyrics)
        prompt = tok.encode(text) + [EOD, SOA, YUE2CODEC]
        gen = torch.Generator(device=dev).manual_seed(args.seed + n); t0 = time.time(); bounds = []
        if args.sections:
            units = ["[instrumental]\n"] if inst else text_units(lyrics, args.unit); prompt = tok.encode(text) + [EOD]
            state = model.new_state(); model.forward(prompt, state); codes = []
            for u in units:
                logits = model.forward([SOS] + tok.encode(u) + [SOA, YUE2CODEC], state); n0 = len(codes)
                while len(codes) - n0 < args.max_sec_frames and len(codes) < args.max_frames:
                    t = sample(logits, mask_eoa if len(codes) - n0 >= args.min_sec_frames else mask, args.temperature, args.top_p, args.top_k, gen)
                    if t == EOA: break
                    codes.append(t - YUE2_BASE); logits = model.forward([t], state)
                bounds.append(dict(text=u, f0=n0, f1=len(codes), eoa=len(codes) - n0 < args.max_sec_frames))
                if len(codes) >= args.max_frames: break
        else:
            state = model.new_state(); logits = model.forward(prompt, state); codes = []
            while len(codes) < args.max_frames:
                t = sample(logits, mask_eoa if len(codes) >= args.min_frames else mask, args.temperature, args.top_p, args.top_k, gen)
                if t == EOA: break
                codes.append(t - YUE2_BASE); logits = model.forward([t], state)
        dt = time.time() - t0; d = out / sid; d.mkdir(exist_ok=True)
        np.save(d / "semantic.npy", np.asarray(codes, dtype=np.int32))
        json.dump(dict(id=sid, source_title=m.get("title"), style=" ".join((m.get("caption") or "").split())[:1500], lyrics=lyrics.strip(), instrumental=inst,
                       ckpt=str(args.ckpt), temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed + n, n_prompt=len(prompt),
                       n_tokens=len(codes), stopped_by_eoa=len(codes) < args.max_frames, audio_seconds=len(codes) / 25, gen_seconds=dt, sections=bounds or None,
                       tok_repeat=float(np.mean(np.diff(codes) == 0)) if len(codes) > 1 else 0.0, tok_unique=len(set(codes))), open(d / "request.json", "w"), indent=1)
        print(f"  [{n + 1}/{len(ids)}] {sid} prompt {len(prompt)} -> {len(codes)} codes ({len(codes) / 25:.0f}s audio, eoa={len(codes) < args.max_frames}"
              f"{', ' + str(len(bounds)) + ' sections ' + '/'.join(str(b['f1'] - b['f0']) for b in bounds) if bounds else ''}) "
              f"repeat {np.mean(np.diff(codes) == 0) if len(codes) > 1 else 0:.3f} unique {len(set(codes))} in {dt:.0f}s ({len(codes) / max(dt, 1e-6):.1f} tok/s)", flush=True)
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    main()
