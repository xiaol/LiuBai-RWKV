#!/usr/bin/env python3
"""Unit test for the state-axes patch (docs/ARCHITECTURE_NOTES.md section 8, src/model.py slot_hold_mask).
1. prints the hold mask on a toy line-format document (slot heads must write only on their line's text);
2. checks that a held head's state really is frozen in the WKV kernel (same r at two held steps -> same y);
3. forward + backward of a 2-layer model with the patch on.
usage (rwkv_py312, one GPU): RWKV_SLOT_HEADS=4 RWKV_SLOT_M=2 python scripts/test_state_axes.py
"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
TT = Path(__file__).resolve().parents[1] / "tools/RWKV-LM/RWKV-v7/train_temp"
for k, v in dict(RWKV_MY_TESTING="x070", RWKV_KERNEL="", RWKV_HEAD_SIZE="64", RWKV_HEAD_L2WRAP_CE_CHUNK="0", RWKV_FLOAT_MODE="bf16",
                 RWKV_JIT_ON="1", RWKV_CTXLEN="64", RWKV_SLOT_HEADS="4", RWKV_SLOT_M="2").items():
    os.environ.setdefault(k, v)
os.chdir(TT); sys.path.insert(0, str(TT))
import torch, torch.nn.functional as F  # noqa: E402
from src.model import RWKV, slot_hold_mask, RWKV7_CLAMPW_CUDA, SLOT_HEADS, SLOT_M  # noqa: E402

EOD, SOA, EOA, YC, SOS, C0 = 65536, 65537, 65538, 65540, 65541, 65544
NAMES = {EOD: "EOD", SOA: "SOA", EOA: "EOA", YC: "YC", SOS: "SOS", 0: "0"}
seq = [10, 11, EOD, SOS, 20, 21, SOA, YC, C0 + 1, C0 + 2, EOA, SOS, 30, SOA, YC, C0 + 3, EOA, SOS, 40, SOA, YC, C0 + 5, EOA, 0]
H = 8
h = slot_hold_mask(torch.tensor([seq]), H)[0].int()
print(f"SLOT_HEADS {SLOT_HEADS} SLOT_M {SLOT_M}; 1 = hold. stock heads must be all 0")
for t, s in enumerate(seq):
    name = NAMES.get(s, f"c{s - C0}" if s >= C0 else f"txt{s}")
    print(f"  {name:>6}  slot heads {h[t, H - SLOT_HEADS:].tolist()}  stock {h[t, :H - SLOT_HEADS].tolist()}")
assert h[:, :H - SLOT_HEADS].sum() == 0

B, T, Hh, N = 1, 32, 2, 64; torch.manual_seed(0)
mk = lambda: (torch.randn(B, T, Hh * N, device="cuda") * 0.5).bfloat16()
r, w, k, v = mk(), mk(), mk(), mk()
kk = F.normalize(torch.randn(B, T, Hh, N, device="cuda"), dim=-1).view(B, T, Hh * N); a = torch.rand(B, T, Hh * N, device="cuda")
hold = torch.zeros(B, T, Hh, dtype=torch.bool, device="cuda"); hold[:, 8:24, 1] = True       # head 1 held for t = 8..23
hc = hold.unsqueeze(-1).expand(B, T, Hh, N).reshape(B, T, Hh * N); keep = (~hc).bfloat16()
r[:, 20] = r[:, 10]
y = RWKV7_CLAMPW_CUDA(r, w.masked_fill(hc, -30.0), k * keep, v, (-kk).bfloat16(), (kk * a).bfloat16() * keep).float()
d_hold = (y[0, 10, N:] - y[0, 20, N:]).abs().max().item(); d_free = (y[0, 10, :N] - y[0, 20, :N]).abs().max().item()
print(f"kernel: held head |y10 - y20| = {d_hold:.2e} (|y10| {y[0, 10, N:].abs().max().item():.2e}); free head {d_free:.2e}")
assert d_hold < 1e-2 * max(1e-6, y[0, 10, N:].abs().max().item()) and d_free > 10 * d_hold

args = SimpleNamespace(n_layer=2, n_embd=512, dim_att=512, dim_ffn=2048, head_size=64, vocab_size=98816, ctx_len=64, my_testing="x070", grad_cp=0)
m = RWKV(args).cuda().bfloat16()
out = m._forward_features(torch.tensor([seq + [0] * (32 - len(seq))], device="cuda")); out.float().sum().backward()
ok = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
print(f"model fwd/bwd {tuple(out.shape)}, grads finite: {ok}"); assert ok
print("PASS")
