"""CPU tests of vae/gated_rwkv7.py (tiny random config, fp32 reference recurrence). Run: python -m vae.test_gated_rwkv7

1. text rows = an independent fp32 re-implementation of tools/RWKV-LM/RWKV-v7/rwkv_v7_demo.py (official x070 math, softplus
   decay form, RNN loop) on the same weights -> the port of the official block is right
2. A rows bit-identical with / without F rows, under any F content; dA/dF exactly 0
3. F of block j: independent of blocks >= j and of other groups; reads history; in-block backward shift reaches frame 0
4. text prefix: text rows ignore A / F; A rows read the text
"""
import torch
import torch.nn.functional as F
from vae.gated_rwkv7 import LiuBaiH, layout

torch.manual_seed(0)
CFG = dict(n_layer=3, C=128, vocab=97, dims=(16, 16, 16, 32), lora=8, dc=16, da=64, blk=4)


def make(lattn):
    M = LiuBaiH(**CFG, lattn=lattn).double()
    with torch.no_grad():
        for n, p in M.named_parameters():                       # everything non-trivial, incl. the zero-init NAR parts
            if "ln" in n and n.endswith("weight") and "mod" not in n: p.copy_(1 + 0.1 * torch.randn_like(p))
            elif n.endswith(("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "ffn.x_k")): p.uniform_(0, 1)
            elif n.endswith("mix_b"): p.uniform_(0, 1)
            else: p.normal_(0, 0.3 if p.dim() < 2 else 1.0 / p.shape[-1] ** 0.5)
    return M.eval()


def ref_text(M, idx):
    """official demo math, fp32 double here, own code path (no gather / gate)"""
    sd = {k: v.double() for k, v in M.state_dict().items()}; L = CFG["n_layer"]; C = CFG["C"]; H = C // 64; N = 64
    ln = lambda x, n: F.layer_norm(x, (C,), sd[n + ".weight"], sd[n + ".bias"])
    shift = lambda x: F.pad(x, (0, 0, 1, -1))
    x = sd["emb.weight"][idx]; B, T, _ = x.shape; v_first = None
    for i in range(L):
        p = lambda n: sd[f"blocks.{i}.{n}"]
        if i == 0: x = ln(x, "blocks.0.ln0")
        h = ln(x, f"blocks.{i}.ln1"); xx = shift(h) - h
        xr, xw, xk, xv, xa, xg = (h + xx * p("att." + m) for m in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))
        r = xr @ p("att.receptance.weight").T
        w = -F.softplus(-(p("att.w0") + torch.tanh(xw @ p("att.w1")) @ p("att.w2"))) - 0.5
        k = xk @ p("att.key.weight").T; v = xv @ p("att.value.weight").T
        if i == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(p("att.v0") + (xv @ p("att.v1")) @ p("att.v2"))
        a = torch.sigmoid(p("att.a0") + (xa @ p("att.a1")) @ p("att.a2")); g = torch.sigmoid(xg @ p("att.g1")) @ p("att.g2")
        kk = F.normalize((k * p("att.k_k")).view(B, T, H, N), dim=-1).view(B, T, C); k = k * (1 + (a - 1) * p("att.k_a"))
        hv = lambda t: t.view(B, T, H, N); rr, ww, kq, vv, aa, bb = hv(r), torch.exp(-torch.exp(hv(w))), hv(k), hv(v), hv(-kk), hv(kk * a)
        S = torch.zeros(B, H, N, N, dtype=x.dtype); ys = []
        for t in range(T):
            S = S * ww[:, t, :, None, :] + S @ aa[:, t, :, :, None] @ bb[:, t, :, None, :] + vv[:, t, :, :, None] @ kq[:, t, :, None, :]
            ys.append((S @ rr[:, t, :, :, None])[..., 0])
        y = torch.stack(ys, 1).reshape(B * T, C)
        y = F.group_norm(y, H, p("att.ln_x.weight"), p("att.ln_x.bias"), eps=64e-5).view(B, T, C)
        y = y + ((hv(r) * hv(k) * p("att.r_k")).sum(-1, keepdim=True) * hv(v)).view(B, T, C)
        x = x + (y * g) @ p("att.output.weight").T
        h = ln(x, f"blocks.{i}.ln2"); kx = h + (shift(h) - h) * p("ffn.x_k")
        x = x + torch.relu(kx @ p("ffn.key.weight").T) ** 2 @ p("ffn.value.weight").T
    return ln(x, "ln_out") @ sd["head.weight"].T


def run(M, z, xt, t, nar, P=0, txt=None, lyr=None, a_c=None):
    B, NB, _ = z.shape
    lyr = torch.zeros(B, NB, M.wb + M.wf, dtype=torch.long) if lyr is None else lyr
    a_c = torch.zeros(B, NB, dtype=torch.long) if a_c is None else a_c
    return M.velocity(z, lyr, a_c, xt, t, nar, txt=txt)


def main():
    B, NB, blk = 2, 6, 4; D = 64 * blk
    for lattn in (False, True):
        M = make(lattn); z = torch.randn(B, NB, D, dtype=torch.double); nar = list(range(NB))
        xt = torch.randn(B, NB, D, dtype=torch.double); t = torch.rand(B, NB, dtype=torch.double)
        txt = torch.randint(0, CFG["vocab"], (B, 7))
        # 1. text rows vs the official demo math
        with torch.no_grad():
            _, _, ht, _ = run(M, z, xt, t, nar, txt=txt)
            lg = ht @ M.head.weight.T; lr = ref_text(M, txt)
        e1 = (lg - lr).abs().max().item(); assert e1 < 1e-8, e1
        # 2. A rows exact with / without F rows and under other F content; zero gradient A <- F
        with torch.no_grad():
            h0 = M.hidden(z, torch.zeros(B, NB, M.wb + M.wf, dtype=torch.long), txt=txt)
            h1, v1, _, _ = run(M, z, xt, t, nar, txt=txt); h2, v2, _, _ = run(M, z, torch.randn_like(xt), torch.rand_like(t), nar, txt=txt)
        assert torch.equal(h0, h1) and torch.equal(h1, h2), ((h0 - h1).abs().max(), (h1 - h2).abs().max())
        xg = xt.clone().requires_grad_(True); h, _, _, _ = run(M, z, xg, t, nar, txt=txt); h.sum().backward()
        assert xg.grad.abs().max().item() == 0.0
        # 3. F of block j: no dependence on blocks >= j or on other groups' frames; depends on history and on its own frames
        j = 3
        with torch.no_grad():
            z2 = z.clone(); z2[:, j:] += torch.randn_like(z2[:, j:]); xt2 = xt.clone(); xt2[:, [i for i in nar if i != j]] += 1.0
            _, va, _, _ = run(M, z, xt, t, nar, txt=txt); _, vb, _, _ = run(M, z2, xt2, t, nar, txt=txt)
            z3 = z.clone(); z3[:, :j - 1] += 1.0; _, vc, _, _ = run(M, z3, xt, t, nar, txt=txt)
            xt4 = xt.clone(); xt4[:, j, -64:] += 1.0; _, vd, _, _ = run(M, z, xt4, t, nar, txt=txt)
        assert torch.equal(va[:, j], vb[:, j]), (va[:, j] - vb[:, j]).abs().max()
        assert (va[:, j] - vc[:, j]).abs().max() > 1e-3, "F reads no history"
        assert (va[:, j, :64] - vd[:, j, :64]).abs().max() > 1e-6, "frame 0 does not see frame 3 (backward shift / lattn)"
        # 4. text rows ignore audio; A rows read text
        with torch.no_grad():
            _, _, hta, _ = run(M, z2, xt2, t, nar, txt=txt); txt2 = txt.clone(); txt2[:, 0] = (txt2[:, 0] + 1) % CFG["vocab"]
            hb = M.hidden(z, torch.zeros(B, NB, M.wb + M.wf, dtype=torch.long), txt=txt2)
        assert torch.equal(ht, hta) and (hb - h0).abs().max() > 1e-6
        print(f"lattn={lattn}: text vs official demo math max |dlogit| {e1:.2e}; A exact with / without F; dA/dF = 0; "
              f"F_j isolated from blocks >= j and other groups; reads history; in-block bidirectional; text causal", flush=True)
    print("all passed")


if __name__ == "__main__":
    main()
