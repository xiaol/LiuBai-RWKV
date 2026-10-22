"""Small RWKV-7-style blocks in plain PyTorch for the VAE (latent rate 12.5 Hz, so sequences are short: 100 frames per training
crop, ~3k for a whole song at inference; a Python-level recurrence over T is cheap next to the 24 kHz conv stacks).

TimeMix follows the RWKV-7 "generalised delta rule" state update per head (state S [N_v, N_k]):
    S_t = S_{t-1} diag(w_t) - (S_{t-1} kk_t) (kk_t * a_t)^T + v_t k_t^T,      y_t = S_t r_t   (removal uses the pre-decay state)
with data-dependent decay w_t in (0.545, 1), in-context learning rate a_t in (0, 1) and normalised removal key kk_t.
BiBlock runs one TimeMix forward and one on the reversed sequence (encoder/decoder: the whole crop is available);
CausalBlock is forward-only (the latent prior, which must not see the future).
"""
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F


_K = None
def _kernel():
    """Official RWKV-7 bf16 training kernel (tools/RWKV-LM/RWKV-v7/train_temp/cuda/rwkv7_clampw.*), head size 64, chunk 16."""
    global _K
    if _K is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "..", "tools", "RWKV-LM", "RWKV-v7", "train_temp", "cuda")
        flags = ["-res-usage", "-D_N_=64", "-D_CHUNK_LEN_=16", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
        load(name="vae_rwkv7_clampw", sources=[os.path.join(src, "rwkv7_clampw.cu"), os.path.join(src, "rwkv7_clampw.cpp")],
             is_python_module=False, verbose=False, extra_cuda_cflags=flags)
        _K = torch.ops.rwkv7_clampw
    return _K


class _WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        B, T, H, N = r.shape; K = _kernel()
        y = torch.empty_like(v); st = torch.empty(B, H, T // 16, N, N, dtype=torch.float32, device=r.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)
        K.forward(r, w, k, v, a, b, y, st, sa); ctx.save_for_backward(r, w, k, v, a, b, st, sa); return y
    @staticmethod
    def backward(ctx, dy):
        r, w, k, v, a, b, st, sa = ctx.saved_tensors; g = [torch.empty_like(x) for x in (r, w, k, v, a, b)]
        _kernel().backward(r, w, k, v, a, b, dy.contiguous(), st, sa, *g); return tuple(g)


def wkv7(r, w_raw, k, v, kk, a, reverse=False):
    """[B,T,H,64] each (w_raw = pre-sigmoid decay logit). Pads T to a multiple of 16; reverse = run on the flipped sequence."""
    T = r.shape[1]
    xs = [t.flip(1) if reverse else t for t in (r, w_raw, k, v, -kk, kk * a)]
    pad = (-T) % 16
    xs = [F.pad(t, (0, 0, 0, 0, 0, pad)).bfloat16().contiguous() for t in xs]
    y = _WKV.apply(*xs)[:, :T]
    return y.flip(1) if reverse else y


def token_shift(x, reverse=False):
    """x [B,T,C] -> previous frame (next frame if reverse), zero at the edge."""
    return F.pad(x, (0, 0, 0, 1))[:, 1:] if reverse else F.pad(x, (0, 0, 1, 0))[:, :-1]


class TimeMix(nn.Module):
    def __init__(s, d, head=64, reverse=False):
        super().__init__(); s.h = d // head; s.n = head; s.reverse = reverse
        s.mix = nn.Parameter(torch.full((6, d), 0.5))                    # token-shift interpolation for r,w,k,v,a,g
        s.r, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.w1, s.w2 = nn.Linear(d, 64, bias=False), nn.Linear(64, d)      # decay LoRA
        s.a1, s.a2 = nn.Linear(d, 64, bias=False), nn.Linear(64, d)      # in-context learning-rate LoRA
        s.g1, s.g2 = nn.Linear(d, 128, bias=False), nn.Linear(128, d, bias=False)
        s.k_k = nn.Parameter(torch.full((d,), 0.85)); s.ln_x = nn.GroupNorm(s.h, d, eps=64e-5)
        nn.init.zeros_(s.o.weight); nn.init.constant_(s.w2.bias, 0.5)
        for m in (s.w2, s.a2, s.g2): nn.init.zeros_(m.weight)

    def forward(s, x):
        B, T, C = x.shape; H, N = s.h, s.n
        dx = token_shift(x, s.reverse) - x
        xr, xw, xk, xv, xa, xg = (x + dx * s.mix[i] for i in range(6))
        r, k, v = s.r(xr), s.k(xk), s.v(xv)
        w_raw = s.w2(torch.tanh(s.w1(xw))).float()                                          # decay exp(-0.6065*sigmoid(w_raw))
        a = torch.sigmoid(s.a2(s.a1(xa))).float(); g = s.g2(torch.sigmoid(s.g1(xg)))
        kk = F.normalize((k * s.k_k).float().view(B, T, H, N), dim=-1)
        k = (k.float() * (1 + (a - 1) * 0.5)).view(B, T, H, N)                            # RWKV-7 k scaling by a
        r, v, w_raw, a = (t.float().view(B, T, H, N) for t in (r, v, w_raw, a))
        if x.is_cuda:
            y = wkv7(r, w_raw, k, v, kk, a, s.reverse).float().reshape(B * T, C)
        else:                                                                               # reference recurrence (CPU / tests)
            w = torch.exp(-0.6065 * torch.sigmoid(w_raw))
            S = x.new_zeros(B, H, N, N, dtype=torch.float32); ys = []
            for t in (range(T - 1, -1, -1) if s.reverse else range(T)):
                kt, at = kk[:, t], a[:, t]
                sa = torch.einsum("bhij,bhj->bhi", S, kt)                                  # removal read from the pre-decay state (official RWKV-7)
                S = S * w[:, t].unsqueeze(-2) - sa.unsqueeze(-1) * (kt * at).unsqueeze(-2) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
                ys.append(torch.einsum("bhij,bhj->bhi", S, r[:, t]))
            if s.reverse: ys = ys[::-1]
            y = torch.stack(ys, 1).reshape(B * T, C)
        y = s.ln_x(y).view(B, T, C).to(x.dtype)
        return s.o(y * g)


class ChannelMix(nn.Module):
    def __init__(s, d, mult=4):
        super().__init__(); s.mix = nn.Parameter(torch.full((d,), 0.5)); s.k = nn.Linear(d, d * mult, bias=False); s.v = nn.Linear(d * mult, d, bias=False)
        nn.init.zeros_(s.v.weight)
    def forward(s, x):
        xx = x + (token_shift(x) - x) * s.mix
        return s.v(torch.relu(s.k(xx)) ** 2)


class BiBlock(nn.Module):
    def __init__(s, d, head=64):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.fwd, s.bwd = TimeMix(d, head), TimeMix(d, head, reverse=True); s.ffn = ChannelMix(d)
    def forward(s, x):
        h = s.ln1(x); x = x + s.fwd(h) + s.bwd(h)
        return x + s.ffn(s.ln2(x))


class CausalBlock(nn.Module):
    def __init__(s, d, head=64):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d); s.tm = TimeMix(d, head); s.ffn = ChannelMix(d)
    def forward(s, x):
        x = x + s.tm(s.ln1(x))
        return x + s.ffn(s.ln2(x))
