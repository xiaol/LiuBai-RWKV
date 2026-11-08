#!/usr/bin/env python3
"""Re-encode precomputed v5 latent shards (tools/precompute_latents.py) with MSVAE (vae/msvae.py), keeping their letter timings.

in  : <src>/r*_*.npz v5 shards (12.5 Hz latents + MMS letter starts / line table), the manifest they were built from
out : <out>/<same name>.npz, same keys and layout as the v5 shards (vae/train_liubai_h.py reads both): lat = MSVAE posterior
      mean [sumT,64] f16 at 25 Hz (bf16 autocast, nmse vs fp32 ~5e-4), st x 2, line frames (f0, f1raw, f1c) x 2, ids / i0 / i1
      unchanged. The letter alignment is the v5 one (MMS at 12.5 Hz frames), only rescaled: +-1 frame at 25 Hz.
      Songs whose MSVAE length differs from 2 x the v5 length by > 16 frames are dropped (logged).
      One output shard per input shard: resumable (existing outputs are skipped). meta.json.
usage: CUDA_VISIBLE_DEVICES=g venvs/yue2/bin/python tools/precompute_msvae.py --src DIR --out DIR --every 6 --rank r --world 4
"""
import argparse, glob, json, os, sys, time
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.msvae import MSVAE  # noqa: E402


class Songs(torch.utils.data.Dataset):
    def __init__(s, paths): s.paths = paths
    def __len__(s): return len(s.paths)
    def __getitem__(s, i):
        try:
            x, sr = sf.read(s.paths[i], dtype="float32", always_2d=True)
            x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]
            if sr != 24000: g = gcd(sr, 24000); x = resample_poly(x, 24000 // g, sr // g, axis=0).astype(np.float32)
            return i, np.ascontiguousarray(x.T)
        except Exception as e:
            return i, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="", help="default: <src>/meta.json manifest")
    ap.add_argument("--every", type=int, default=1, help="take every k-th source shard (sorted)")
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = args.manifest or json.load(open(Path(args.src) / "meta.json"))["manifest"]
    if not os.path.isabs(man): man = str(ROOT / man)
    files = sorted(glob.glob(os.path.join(args.src, "r*_*[0-9].npz")))[::args.every][args.rank::args.world]
    todo = [p for p in files if not (out / Path(p).name).exists()]
    if args.rank == 0:
        json.dump(dict(vars(args), manifest=man, vae="msvae", sr=MSVAE.sr, hop=MSVAE.hop, rate=MSVAE.rate, grid_frames=8,
                       letters="v5 MMS alignment x 2"), open(out / "meta.json", "w"), indent=1)
    srcs = [{k: v for k, v in np.load(p).items()} for p in todo]
    need = {int(r) for d in srcs for r in d["row"]}; paths = {}
    for n, l in enumerate(open(man, encoding="utf-8")):
        if n in need: paths[n] = json.loads(l)["audio_path"]
    print(f"rank {args.rank}: {len(todo)}/{len(files)} shards to do, {len(need)} songs", flush=True)
    V = MSVAE(bs=args.bs, amp=True); t0 = time.time(); secs = 0.0; n_ok = 0
    for p, d in zip(todo, srcs):
        rows = d["row"].tolist()
        dl = torch.utils.data.DataLoader(Songs([paths[r] for r in rows]), batch_size=None, num_workers=args.workers, prefetch_factor=2)
        buf = []
        for j, wav in dl:
            a, b = d["lat_off"][j], d["lat_off"][j + 1]; i0, i1 = d["id_off"][j], d["id_off"][j + 1]; l0, l1 = d["line_off"][j], d["line_off"][j + 1]
            if isinstance(wav, str): print(f"skip row {rows[j]}: {wav}", flush=True); continue
            mu = V.encode(torch.as_tensor(wav)[None].cuda())[0][0]
            if abs(len(mu) - 2 * (b - a)) > 16: print(f"skip row {rows[j]}: length {len(mu)} vs 2 x {b - a}", flush=True); continue
            tab = d["lines"][l0:l1].astype(np.int64); tab[:, :3] = np.minimum(tab[:, :3] * 2, 2**31 - 1)
            buf.append(dict(lat=mu.cpu().numpy().astype(np.float16), ids=d["ids"][i0:i1], st=d["st"][i0:i1] * 2, lines=tab.astype(np.int32),
                            row=rows[j], dur=float(d["dur"][j])))
            n_ok += 1; secs += wav.shape[1] / 24000
        cat = lambda k, dt: np.concatenate([x[k] for x in buf]).astype(dt)
        off = lambda k: np.cumsum([0] + [len(x[k]) for x in buf]).astype(np.int64)
        o = dict(lat=cat("lat", np.float16), lat_off=off("lat"), ids=cat("ids", np.uint8), st=cat("st", np.float32), id_off=off("ids"),
                 lines=np.concatenate([x["lines"] for x in buf]).reshape(-1, 5).astype(np.int32), line_off=off("lines"),
                 row=np.array([x["row"] for x in buf], np.int64), dur=np.array([x["dur"] for x in buf], np.float32))
        tmp = out / (Path(p).stem + ".tmp.npz"); np.savez(tmp, **o); os.replace(tmp, out / Path(p).name)
        el = time.time() - t0
        print(f"{Path(p).name}: {len(buf)}/{len(rows)} songs; total {n_ok} songs {secs / 3600:.1f} h, {secs / el:.0f}x realtime", flush=True)
    print(f"done {n_ok} songs {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
