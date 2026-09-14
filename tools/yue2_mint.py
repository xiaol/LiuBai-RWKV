#!/usr/bin/env python3
"""Mint paired (tokens, latents, audio) songs with YuE2 from suno-94k captions + lyrics (PLAN §7.1 B2 / §8.3 R1).

Per song writes out/<id>/{audio.flac (48 kHz stereo), semantic.npy (int32 YuE2 codes), latent.npy (fp16 [T,64]),
request.json}. Prompts are drawn from a manifest shard (same distribution as the RWKV training corpus).
Run inside venvs/yue2 on one GPU (peak ~11-14 GiB).

usage: yue2_mint.py --n 5 [--shard 1] [--cot off] [--out data/yue2_minted_own] [--seed-base 1000] [--offset 0]
"""
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np, soundfile as sf, torch

ROOT = Path(__file__).resolve().parents[1]
YUE2 = ROOT / "models/yue2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--cot", default="off")
    ap.add_argument("--out", default=str(ROOT / "data/yue2_minted_own")); ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--vae", default="YuE2-Vae"); ap.add_argument("--max-sec", type=float, default=240)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(l) for l in open(ROOT / f"data/meta/suno94k/shard-{args.shard:05d}.jsonl")]
    ok = [m for m in meta if not m.get("error") and 60 <= m["duration"] <= args.max_sec and len(m.get("lyrics", "")) > 100
          and "[instrumental]" not in m["lyrics"].lower() and all(ord(c) < 0x250 for c in m["lyrics"])]
    ok.sort(key=lambda m: hashlib.md5(m["id"].encode()).hexdigest()); songs = ok[args.offset:args.offset + args.n]
    from yue2 import YuE2Pipeline
    pipe = YuE2Pipeline.from_pretrained(str(YUE2 / "YuE2-3B"), vae=str(YUE2 / args.vae), local_files_only=True, progress=False)
    t0 = time.time()
    for i, m in enumerate(songs):
        d = out / m["id"]
        if (d / "latent.npy").exists(): continue
        d.mkdir(exist_ok=True)
        style = " ".join(m["caption"].split())[:1500]; lyrics = m["lyrics"].strip(); seed = args.seed_base + i
        t1 = time.time()
        res = pipe(style=style, lyrics=lyrics, cot=args.cot, seed=seed, id=m["id"])
        audio = np.asarray(res.audio, dtype=np.float32)
        sf.write(d / "audio.flac", audio, 48000, subtype="PCM_16")
        np.save(d / "semantic.npy", np.asarray(res.semantic.tokens, dtype=np.int32))
        np.save(d / "latent.npy", np.asarray(res.latents, dtype=np.float16))
        json.dump(dict(id=m["id"], source_title=m["title"], style=style, lyrics=lyrics, cot=args.cot, seed=seed,
                       audio_seconds=len(audio) / 48000, n_tokens=len(res.semantic.tokens), gen_seconds=time.time() - t1),
                  open(d / "request.json", "w"), indent=1)
        print(f"  [{i + 1}/{len(songs)}] {m['id']} {len(audio) / 48000:.0f}s audio, {len(res.semantic.tokens)} tokens, {time.time() - t1:.0f}s gen, total {time.time() - t0:.0f}s", flush=True)
    pipe.close(); print("MINT DONE", flush=True)


if __name__ == "__main__":
    main()
