"""Unit checks for vae/lyric_renderer.py: interleave layout, frame-order gather, and that velocities stay local to their frame.
run: CUDA_VISIBLE_DEVICES=0 venvs/yue2/bin/python vae/test_lyric_renderer.py"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.lyric_renderer import LyricRenderer, interleave  # noqa: E402

pt, fi, li = interleave(6, [(0, [5, 6]), (3, [7])], 100)
assert pt.tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0], pt.tolist()
assert fi.tolist() == [-1, -1, -1, 0, 1, 2, -1, -1, 3, 4, 5], fi.tolist()
assert li.tolist() == [5, 6, 1, 0, 0, 0, 7, 1, 0, 0, 0], li.tolist()
print("interleave ok")

torch.manual_seed(0); dev = "cuda"
R = LyricRenderer(d=128, n=2, vocab=50).to(dev)
with torch.no_grad():                                   # RWKV output projections start at zero (no mixing until trained): randomise
    for name, p in R.named_parameters():
        if p.dim() == 2 and p.abs().sum() == 0: p.normal_(std=0.05)
codes = torch.randint(0, 50, (2, 6), device=dev); lines = [[(0, [5, 6]), (3, [7])], []]
lay = R.build(codes, lines); x = torch.randn(2, 6, 64, device=dev); t = torch.full((2,), 0.5, device=dev)
with torch.no_grad():
    v0 = R(x, t, lay); assert v0.shape == (2, 6, 64), v0.shape
    for b, f in ((1, 4), (0, 2), (0, 5)):
        x2 = x.clone(); x2[b, f] += 5; d = (R(x2, t, lay) - v0)[b].abs().sum(-1)
        assert d.argmax().item() == f, (b, f, d.tolist())
    # lyrics change the output of sample 0 but not sample 1 (which has none)
    lay2 = R.build(codes, [[(0, [9, 9, 9]), (3, [7])], []]); d = (R(x, t, lay2) - v0).abs().sum((1, 2))
    assert d[0] > 1e-3 and d[1] < 1e-5, d.tolist()
print("frame gather / locality / lyric dependence ok")
z = R.sample(codes, lines, steps=4, cfg_lyr=2.0); assert z.shape == (2, 6, 64)
print("sample ok")
