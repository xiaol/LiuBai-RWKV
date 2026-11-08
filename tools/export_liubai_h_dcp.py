#!/usr/bin/env python3
"""Offline export of a LiuBai-H DCP checkpoint (vae/train_liubai_h.py <out>/dcp) to the M_ema_s{step}.pt format that
vae/eval_cursor_lm.py reads (M = bf16 EMA weights, step, args, lat_m, lat_s). CPU only, no process group: reads just the
"ema.*" entries of the checkpoint. lat_m / lat_s are recomputed exactly as the trainer does (first 6 train shards, every 7th
frame), so the result equals the in-training export.
usage: venvs/yue2/bin/python tools/export_liubai_h_dcp.py out/liubai_h/prod1 [--out FILE]
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out", default="")
a = ap.parse_args(); run = Path(a.run); ck = run / "dcp"
md = dcp.FileSystemReader(str(ck)).read_metadata().state_dict_metadata
ema = {k[4:]: torch.empty(v.size, dtype=v.properties.dtype) for k, v in md.items()
       if k.startswith("ema.") and isinstance(v, TensorStorageMetadata)}
st = dict(ema=ema, step=torch.tensor(0))
dcp.load(st, checkpoint_id=str(ck)); step = int(st["step"])
args = json.load(open(run / "config.json")); args["tiny_cfg"] = None
assert args.get("arch") == "liubai_h" and not args.get("tiny"), args.get("arch")
files = sorted(glob.glob(os.path.join(args["data"], "r*_*[0-9].npz")))
lat = np.concatenate([np.load(p)["lat"][::7].astype(np.float32) for p in files[:6]])
lat_m = torch.from_numpy(lat.mean(0)); lat_s = torch.from_numpy(lat.std(0)).clamp_min(1e-3)
out = Path(a.out) if a.out else run / f"M_ema_s{step}.pt"
torch.save(dict(M={k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in ema.items()}, step=step, args=args,
                lat_m=lat_m, lat_s=lat_s), out)
print(f"step {step}: {len(ema)} tensors, {sum(v.numel() for v in ema.values()) / 1e9:.3f}B params -> {out}")
