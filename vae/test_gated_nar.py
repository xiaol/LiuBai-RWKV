"""Unit checks for vae/gated_nar.py (CPU, fp32 reference recurrence): layout, A<-F no leak (values + gradients), block isolation,
F->A causality, in-block receptive field (bidirectional / zero-init = causal / iso), inference layout = training layout.
run: venvs/yue2/bin/python vae/test_gated_nar.py"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.gated_nar import GatedNARLM, layout  # noqa: E402

L = layout(3, 2, [0, 2])
assert L["isf"].int().tolist() == [0, 1, 1, 0, 0, 1, 1], L["isf"].tolist()
assert L["prev"].tolist() == [-1, 0, 1, 0, 3, 4, 5], L["prev"].tolist()      # A_1 skips F_0, F_2,1 <- A_2
assert L["nxt"].tolist() == [-1, 2, -1, -1, -1, 6, -1], L["nxt"].tolist()
assert L["a_pos"].tolist() == [0, 3, 4] and L["f_pos"].tolist() == [[1, 2], [5, 6]]
Li = layout(3, 2, [0, 2], iso=True)
assert Li["prev"].tolist() == [-1, 0, 0, 0, 3, 4, 4] and (Li["nxt"] == -1).all()
Lr = layout(2, 2, [1], reps=2)
assert Lr["isf"].int().tolist() == [0, 0, 1, 1, 1, 1] and Lr["prev"].tolist() == [-1, 0, 1, 2, 1, 4], Lr["prev"].tolist()
assert Lr["nxt"].tolist() == [-1, -1, 3, -1, 5, -1] and Lr["f_pos"].tolist() == [[2, 3], [4, 5]]
print("layout ok")

torch.manual_seed(0); torch.set_num_threads(16)
# fresh init (no randomising, unlike the checks below): every time-mix must get gradients at step 0. The old init zeroed both
# o and g2, which is a fixed point (o's grad is scaled by g, g2's by o): WKV never trained in any pre-2026-10-31 model.
from vae.rwkv import TimeMix  # noqa: E402
from vae.gated_nar import GTimeMix  # noqa: E402
for name, m, run in (("rwkv.TimeMix", TimeMix(128), lambda m, x: m(x)), ("GTimeMix", GTimeMix(128), lambda m, x: m(x, layout(3, 2, [0, 2])))):
    x = torch.randn(2, 7, 128); run(m, x).sum().backward(); assert m.o.weight.grad.abs().max() > 0, name   # step 0: o moves
    with torch.no_grad(): m.o.weight -= 0.1 * m.o.weight.grad
    m.zero_grad(); run(m, x).sum().backward()                                                              # step 1: then g2 / r
    assert m.g2.weight.grad.abs().max() > 0 and m.r.weight.grad.abs().max() > 0, name
print("fresh time-mix init trains (o at step 0, g2 / r after) ok")

lat, blk, NB, B = 8, 4, 10, 2
M = GatedNARLM(latent=lat, blk=blk, d=128, n=4, head=64, lora=8)
with torch.no_grad():                                   # zero-init deltas / outputs would make every check trivial: randomise
    for name, p in M.named_parameters():
        if p.abs().sum() == 0: p.normal_(std=0.3 if "mix_b" in name else 0.05)
M.eval()
nar = list(range(NB)); lay = layout(NB, blk, nar)
z = torch.randn(B, NB, lat * blk); xt = torch.randn(B, NB, blk, lat); t = torch.rand(B, NB)
with torch.no_grad():
    h0, v0 = M(z, xt, t, lay)
    # A <- F: different noisy frames / flow times, or no F rows at all, give bit-identical A outputs
    h1, _ = M(z, torch.randn_like(xt) * 3, torch.rand(B, NB), lay)
    h2, _ = M(z, xt[:, :0], t[:, :0], layout(NB, blk, []))
    assert (h1 - h0).abs().max() == 0 and (h2 - h0).abs().max() == 0, ((h1 - h0).abs().max(), (h2 - h0).abs().max())
print("A <- F blocked (values, bit-exact) ok")

xg = xt.clone().requires_grad_(); tg = t.clone().requires_grad_()
h, v = M(z, xg, tg, lay); h.square().sum().backward()
assert xg.grad.abs().max() == 0 and tg.grad.abs().max() == 0
xg.grad = None; zg = z.clone().requires_grad_(); h, v = M(zg, xg, t, lay); v.square().sum().backward()
assert zg.grad.abs().max() > 0 and xg.grad.abs().max() > 0 and all(torch.isfinite(p.grad).all() for p in M.parameters() if p.grad is not None)
print("A <- F blocked (gradients exactly 0), F loss trains A stream ok")

with torch.no_grad():
    for j in (0, 4, 9):                                 # frames of block j only move the velocities of block j
        x2 = xt.clone(); x2[:, j] += 2; d = (M(z, x2, t, lay)[1] - v0).abs().amax((0, 2, 3))
        assert d[j] > 1e-3 and (d[torch.arange(NB) != j] == 0).all(), (j, d.tolist())
    for m in (2, 6):                                    # block m enters A_{m+1}: moves F of blocks > m only
        z2 = z.clone(); z2[:, m] += 2; d = (M(z2, xt, t, lay)[1] - v0).abs().amax((0, 2, 3))
        assert (d[: m + 1] == 0).all() and (d[m + 1:] > 1e-4).all(), (m, d.tolist())
print("block isolation + F -> A causal ok")


def recept(model, lay):
    """[blk, blk] |d v_frame_row / d x_frame_col| for block 5"""
    J = torch.zeros(blk, blk)
    for c in range(blk):
        x2 = xt.clone(); x2[:, 5, c] += 2
        with torch.no_grad(): J[:, c] = (model(z, x2, t, lay)[1] - model(z, xt, t, lay)[1])[:, 5].abs().amax((0, 2))
    return J


Jb = recept(M, lay); assert (Jb > 1e-4).all(), Jb
Ji = recept(M, layout(NB, blk, nar, iso=True)); assert (Ji.diag() > 1e-4).all() and (Ji - torch.diag(Ji.diag())).abs().max() == 0, Ji
mb = {n: p.detach().clone() for n, p in M.named_parameters() if "mix_b" in n}
with torch.no_grad():
    for n, p in M.named_parameters():
        if "mix_b" in n: p.zero_()
Jc = recept(M, lay); lo = torch.ones(blk, blk).tril().bool(); assert (Jc[~lo] == 0).all() and (Jc[lo] > 1e-4).all(), Jc
with torch.no_grad():
    for n, p in M.named_parameters():
        if n in mb: p.copy_(mb[n])
print("in-block receptive field: bidirectional full / mix_b = 0 lower-triangular (causal) / iso diagonal ok")

with torch.no_grad():                                   # inference: the live A stream up to A_i plus F_i only
    err = 0.0
    for i in (0, 3, 9):
        _, vi = M(z[:, : i + 1], xt[:, i: i + 1], t[:, i: i + 1], layout(i + 1, blk, [i]))
        err = max(err, ((vi[:, 0] - v0[:, i]).abs().max() / v0[:, i].abs().max()).item())
    assert err < 1e-5, err
print(f"inference layout = training layout ok (max rel diff {err:.1e})")

from vae.gated_nar import GatedCursorLM  # noqa: E402
C = GatedCursorLM(latent=lat, blk=blk, d=128, n=3, head=64, lora=8)
with torch.no_grad():
    for name, p in C.named_parameters():
        if p.abs().sum() == 0: p.normal_(std=0.3 if "mix_b" in name else 0.05)
C.eval(); R = 3
lyr = torch.randint(1, 30, (B, NB, C.wb + C.wf)); a_c = torch.randint(0, C.max_adv, (B, NB)); nb_ = [1, 4, 8]
xr = torch.randn(B, len(nb_) * R, lat * blk); tr = torch.rand(B, len(nb_) * R)
with torch.no_grad():
    h, v = C.velocity(z, lyr, a_c, xr, tr, nb_, reps=R)
    assert (C.hidden(z, lyr) - h).abs().max() == 0
    for g in (0, 5):                                    # each noise draw (group) alone = the same velocity
        _, vg = C.velocity(z, lyr, a_c, xr[:, g: g + 1], tr[:, g: g + 1], [nb_[g // R]])
        assert ((vg[:, 0] - v[:, g]).abs().max() / v[:, g].abs().max()) < 1e-5
    lyr2 = lyr.clone(); lyr2[:, 4] = 7; d = (C.velocity(z, lyr2, a_c, xr, tr, nb_, reps=R)[1] - v).abs().amax((0, 2))
    assert (d[:3] == 0).all() and (d[3:] > 1e-5).all(), d.tolist()   # window of block 4 reaches blocks >= 4 only
print("GatedCursorLM: hidden = A stream of velocity, reps groups independent, window causality ok")

# local attention (lattn): same mask guarantees, and with iso it alone gives the full in-block receptive field
for iso in (False, True):
    torch.manual_seed(1); Ma = GatedNARLM(latent=lat, blk=blk, d=128, n=4, head=64, lora=8, lattn=True)
    with torch.no_grad():
        for name, p in Ma.named_parameters():
            if p.abs().sum() == 0: p.normal_(std=0.3 if "mix_b" in name else 0.05)
    Ma.eval(); la = layout(NB, blk, nar, iso=iso)
    with torch.no_grad():
        h0, v0 = Ma(z, xt, t, la); h1, _ = Ma(z, torch.randn_like(xt) * 3, torch.rand(B, NB), la)
        h2, _ = Ma(z, xt[:, :0], t[:, :0], layout(NB, blk, [], iso=iso))
        assert (h1 - h0).abs().max() == 0 and (h2 - h0).abs().max() == 0
        for j in (0, 4, 9):
            x2 = xt.clone(); x2[:, j] += 2; d = (Ma(z, x2, t, la)[1] - v0).abs().amax((0, 2, 3))
            assert d[j] > 1e-3 and (d[torch.arange(NB) != j] == 0).all(), (j, d.tolist())
        for m in (2, 6):
            z2 = z.clone(); z2[:, m] += 2; d = (Ma(z2, xt, t, la)[1] - v0).abs().amax((0, 2, 3))
            assert (d[: m + 1] == 0).all() and (d[m + 1:] > 1e-4).all(), (m, d.tolist())
        err = 0.0
        for i in (0, 3, 9):
            _, vi = Ma(z[:, : i + 1], xt[:, i: i + 1], t[:, i: i + 1], layout(i + 1, blk, [i], iso=iso))
            err = max(err, ((vi[:, 0] - v0[:, i]).abs().max() / v0[:, i].abs().max()).item())
        assert err < 1e-5, err
    J = recept(Ma, la); assert (J > 1e-4).all(), J
    xg = xt.clone().requires_grad_(); h, _ = Ma(z, xg, t, la); h.square().sum().backward(); assert xg.grad.abs().max() == 0
    print(f"lattn iso={iso}: A bit-exact, dA/dF = 0, block isolation, F -> A causal, inference = training ({err:.1e}), full in-block field ok")

Ca = GatedCursorLM(latent=lat, blk=blk, d=128, n=3, head=64, lora=8, iso=True, lattn=True)
with torch.no_grad():
    for name, p in Ca.named_parameters():
        if p.abs().sum() == 0: p.normal_(std=0.05)
Ca.eval()
with torch.no_grad():
    h, v = Ca.velocity(z, lyr, a_c, xr, tr, nb_, reps=R); assert (Ca.hidden(z, lyr) - h).abs().max() == 0
    d = (Ca.velocity(z, lyr2, a_c, xr, tr, nb_, reps=R)[1] - v).abs().amax((0, 2)); assert (d[:3] == 0).all() and (d[3:] > 1e-5).all(), d.tolist()
    _, vg = Ca.velocity(z, lyr, a_c, xr[:, 5:6], tr[:, 5:6], [nb_[5 // R]]); assert ((vg[:, 0] - v[:, 5]).abs().max() / v[:, 5].abs().max()) < 1e-5
print("GatedCursorLM lattn: hidden exact, window causality, reps independent ok")

# incremental inference (step_hidden / step_velocity with the state cache) = full forward, every arm
for kw in (dict(), dict(iso=True), dict(lattn=True), dict(lattn=True, iso=True)):
    torch.manual_seed(2); Cs = GatedCursorLM(latent=lat, blk=blk, d=128, n=3, head=64, lora=8, **kw)
    with torch.no_grad():
        for name, p in Cs.named_parameters():
            if p.abs().sum() == 0: p.normal_(std=0.3 if "mix_b" in name else 0.05)
    Cs.eval(); cache = Cs.cache_init(B, "cpu"); eh = ev = 0.0
    with torch.no_grad():
        hf = Cs.hidden(z, lyr)
        for i in range(NB):
            h_i = Cs.step_hidden(None if i == 0 else z[:, i - 1], lyr[:, i], cache)
            eh = max(eh, ((h_i - hf[:, i]).abs().max() / hf[:, i].abs().max()).item())
            if i in (0, 3, 9):
                x1 = torch.randn(B, 1, lat * blk); t1 = torch.rand(B, 1)
                _, vf = Cs.velocity(z[:, : i + 1], lyr[:, : i + 1], a_c[:, : i + 1], x1, t1, [i])
                vs = Cs.step_velocity(x1[:, 0], t1[:, 0], lyr[:, i], a_c[:, i], cache, None if i == 0 else z[:, i - 1])
                ev = max(ev, ((vs - vf[:, 0]).abs().max() / vf[:, 0].abs().max()).item())
    assert eh < 1e-5 and ev < 1e-5, (kw, eh, ev)
    print(f"incremental {kw or 'default'}: h rel diff {eh:.1e}, v rel diff {ev:.1e} ok")
