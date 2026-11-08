"""CPU check (fp64) for vae/gated_rwkv7.py incremental inference: cache_init / step_hidden / step_velocity = the full forward
(hidden / velocity), with and without a text prefix, for default / iso / lattn / lattn + iso.
run: venvs/yue2/bin/python vae/test_liubai_h_step.py"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.gated_rwkv7 import LiuBaiH  # noqa: E402

torch.set_num_threads(16); B, NB, P = 2, 8, 5
for kw in (dict(), dict(iso=True), dict(lattn=True), dict(lattn=True, iso=True)):
    torch.manual_seed(0)
    M = LiuBaiH(n_layer=3, C=128, vocab=100, dims=(16, 16, 16, 32), lora=8, dc=32, da=64, **kw).double().eval()
    with torch.no_grad():
        for name, p in M.named_parameters():
            if p.abs().sum() == 0 or "x_" in name: p.normal_(std=0.3 if ("mix_b" in name or "x_" in name) else 0.05)
    z = torch.randn(B, NB, M.D, dtype=torch.float64); lyr = torch.randint(1, 20, (B, NB, M.wb + M.wf))
    a_c = torch.randint(0, M.max_adv, (B, NB)); txt = torch.randint(0, 100, (B, P))
    for tx in (None, txt):
        cache = M.cache_init(B, "cpu", tx); eh = ev = 0.0
        with torch.no_grad():
            hf = M.hidden(z, lyr, txt=tx)
            for i in range(NB):
                h = M.step_hidden(None if i == 0 else z[:, i - 1], lyr[:, i], cache)
                eh = max(eh, ((h - hf[:, i]).abs().max() / hf[:, i].abs().max()).item())
                if i in (0, 3, NB - 1):
                    x1 = torch.randn(B, 1, M.D, dtype=torch.float64); t1 = torch.rand(B, 1, dtype=torch.float64)
                    vf = M.velocity(z[:, : i + 1], lyr[:, : i + 1], a_c[:, : i + 1], x1, t1, [i], txt=tx)[1][:, 0]
                    vs = M.step_velocity(x1[:, 0], t1[:, 0], lyr[:, i], a_c[:, i], cache, None if i == 0 else z[:, i - 1])
                    assert vf.abs().max() > 1e-6, "velocity trivially zero: randomise more params"
                    ev = max(ev, ((vs - vf).abs().max() / vf.abs().max()).item())
        assert eh < 1e-9 and ev < 1e-9, (kw, tx is not None, eh, ev)
        print(f"LiuBaiH incremental {kw or 'default'} text={tx is not None}: h {eh:.1e} v {ev:.1e} ok")
