#!/usr/bin/env python3
"""Does the lyric renderer use the lyrics? On held-out crops, for t in (0.2, 0.5, 0.8):
  |v(lyrics) - v(no lyrics)| / |v(lyrics)|     and the same for codes (v(codes) vs v(null codes)),
  and the reader CTC loss of the clean estimate with own / swapped / no lyrics (lower = more readable as the true words).
usage: diag_lyric_use.py out/render/lyr1/R_ema_s50000.pt"""
import sys, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE, LyricCTC  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.lyric_renderer import LyricRenderer  # noqa: E402
from vae.data import Crops, collate  # noqa: E402

dev = "cuda"; torch.manual_seed(0)
ck = torch.load(ROOT / "out/vae/v1/G_s200000.pt", map_location="cpu"); a = ck["args"]
V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
reader = LyricCTC(); reader.load_state_dict(ck["ctc"]); reader = reader.to(dev).eval()
rk = torch.load(sys.argv[1], map_location="cpu"); ra = rk["args"]
tk = torch.load(ra["tok"], map_location="cpu"); TK = SemTokenizer(); TK.load_state_dict(tk["T"]); TK = TK.to(dev).eval()
R = LyricRenderer(d=ra["d"], n=ra["n"], vocab=TK.fsq.n_codes); R.load_state_dict(rk["R"]); R = R.to(dev).eval()
vd = Crops(str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"), holdout=0, max_rows=1000, seed=777, with_lines=True); it = iter(vd)
wav, tg, lens, ok, cf, lines = collate([next(it) for _ in range(16)])
print("crops with lyric lines:", sum(1 for l in lines if l), "of 16; letters per crop", [sum(len(i) for _, i in l) for l in lines])
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    mu = V.encode(wav.to(dev))[0].float(); c = TK.encode(mu)[1]; B = mu.shape[0]
    L_own = R.build(c, lines); L_none = R.build(c, [[] for _ in lines]); L_swap = R.build(c, lines[1:] + lines[:1])
    L_notok = R.build(c, lines, drop_tok=torch.ones(B, dtype=torch.bool, device=dev))
    idx = ok.nonzero().flatten(); offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
    tgt = torch.cat([tg[offs[i]:offs[i + 1]] for i in idx.tolist()]).to(dev)
    for t_ in (0.2, 0.5, 0.8):
        t = torch.full((B,), t_, device=dev); noise = torch.randn_like(mu); xt = (1 - t_) * noise + t_ * mu
        v = {k: R(xt, t, L).float() for k, L in (("own", L_own), ("none", L_none), ("swap", L_swap), ("notok", L_notok))}
        rel = lambda k: ((v["own"] - v[k]).norm() / v["own"].norm()).item()
        ctcl = {}
        for k in ("own", "none", "swap"):
            x1 = xt + (1 - t_) * v[k]; lp = reader(x1.float())
            ctcl[k] = F.ctc_loss(lp[idx].transpose(0, 1), tgt, torch.clamp(cf[idx], max=lp.shape[1]).to(dev), lens[idx].to(dev), blank=0, zero_infinity=True).item()
        print(f"t {t_}: |v_own - v_nolyr|/|v| {rel('none'):.4f}  swap {rel('swap'):.4f}  null codes {rel('notok'):.4f} | reader CTC own {ctcl['own']:.3f} none {ctcl['none']:.3f} swap {ctcl['swap']:.3f}")
