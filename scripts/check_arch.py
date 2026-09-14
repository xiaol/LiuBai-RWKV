#!/usr/bin/env python3
"""Build the RWKV-LM train_temp model with the stage-1 args and compare every tensor's shape with the
resized checkpoint. Also compiles the CUDA kernels (first run takes a few minutes) and, with --forward,
runs one bf16 forward on a real training window to report loss and peak memory on one GPU.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/check_arch.py [--ckpt out/stage1/rwkv-init.pth] [--forward]
"""
import argparse
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TT = ROOT / "tools" / "RWKV-LM" / "RWKV-v7" / "train_temp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "out" / "stage1" / "rwkv-init.pth"))
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--forward", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_KERNEL", "")
    os.environ.setdefault("RWKV_CTXLEN", str(a.ctx))
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    os.environ.setdefault("RWKV_HEAD_CE_VOCAB", "67072")
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
    for k, v in (("RWKV_D_DECAY_LORA", "96"), ("RWKV_D_AAA_LORA", "96"), ("RWKV_D_MV_LORA", "64"), ("RWKV_D_GATE_LORA", "320")):
        os.environ.setdefault(k, v)
    os.chdir(TT); sys.path.insert(0, str(TT))
    import torch
    from src.model import RWKV

    args = types.SimpleNamespace(n_layer=32, n_embd=2560, dim_att=2560, dim_ffn=10240, vocab_size=67072, head_size=64,
                                 ctx_len=a.ctx, my_testing="x070", grad_cp=1, weight_decay=0.1, lr_init=1e-4, lr_final=1e-5,
                                 betas=(0.9, 0.99), adam_eps=1e-18, precision="bf16", strategy="deepspeed_stage_2",
                                 head_chunk=0, kernel="", my_exit_tokens=0, warmup_steps=0, train_stage=3, layerwise_lr=1)
    model = RWKV(args)
    msd = model.state_dict()
    sd = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=True)
    missing = [k for k in msd if k not in sd]; extra = [k for k in sd if k not in msd]
    mism = [(k, tuple(msd[k].shape), tuple(sd[k].shape)) for k in msd if k in sd and tuple(msd[k].shape) != tuple(sd[k].shape)]
    print(f"model tensors {len(msd)}, ckpt tensors {len(sd)}; missing-in-ckpt {len(missing)}, extra-in-ckpt {len(extra)}, shape mismatches {len(mism)}")
    for x in (missing[:10], extra[:10], mism[:10]):
        if x:
            print("  ", x)
    if missing or extra or mism:
        sys.exit(1)
    print("state_dict layout matches the checkpoint")
    if not a.forward:
        return
    model.load_state_dict(sd)
    model = model.to("cuda", torch.bfloat16)
    from src.binidx import MMapIndexedDataset
    ds = MMapIndexedDataset(str(ROOT / "data" / "binidx" / "suno94k_stage1_val"))
    d = torch.tensor(ds.get(idx=0, offset=0, length=a.ctx + 1).astype("int64"))
    x, y = d[:-1][None].cuda(), d[1:][None].cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
    text_mask = (y < 65536)
    with torch.no_grad():
        lt = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1), reduction="none")
    print(f"forward ok: loss {loss.item():.3f} (text-token loss {lt[text_mask.view(-1)].mean().item():.3f}, "
          f"audio-token loss {lt[~text_mask.view(-1)].mean().item():.3f}), peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
