#!/usr/bin/env python3
"""Is MMS_FA a valid lyric judge? Held-out crops: MMS CTC loss + greedy letter CER for original / decode(mu) / token renderer
(r_fsq 100k) / lyric renderer (lyr1 50k). A valid judge ranks original < VAE << renders, like Whisper did (0.36 / 0.45 / 0.98)."""
import sys, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.renderer import Renderer  # noqa: E402
from vae.lyric_renderer import LyricRenderer  # noqa: E402
from vae.mms_judge import MMSJudge  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402

dev = "cuda"
ck = torch.load(ROOT / "out/vae/v1/G_s200000.pt", map_location="cpu"); a = ck["args"]
V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
tk = torch.load(ROOT / "out/tok/b1_ctc1/last.pt", map_location="cpu"); TK = SemTokenizer(); TK.load_state_dict(tk["T"]); TK = TK.to(dev).eval()
rf = torch.load(ROOT / "out/render/r_fsq/R_ema_s100000.pt", map_location="cpu"); RF = Renderer(d=rf["args"]["d"], n=rf["args"]["n"], vocab=TK.fsq.n_codes); RF.load_state_dict(rf["R"]); RF = RF.to(dev).eval()
rl = torch.load(ROOT / "out/render/lyr1/R_ema_s50000.pt", map_location="cpu"); RL = LyricRenderer(d=rl["args"]["d"], n=rl["args"]["n"], vocab=TK.fsq.n_codes); RL.load_state_dict(rl["R"]); RL = RL.to(dev).eval()
J = MMSJudge().to(dev)
vd = Crops(str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"), holdout=0, max_rows=1000, seed=4242, with_lines=True); it = iter(vd)
res = {}
for _ in range(4):
    wav, tg, lens, ok, cf, lines = collate([next(it) for _ in range(8)]); x = wav.to(dev); S = x.shape[-1]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = V.encode(x)[0].float(); c = TK.encode(mu)[1]
            zs = {"vae": mu, "render_tok": RF.sample(c, T=mu.shape[1], steps=24).float(), "render_lyr": RL.sample(c, lines, steps=24).float()}
        auds = {"original": x}
        for k, z in zs.items(): y = V.decode(z).float(); auds[k] = F.pad(y, (0, max(0, S - y.shape[-1])))[..., :S]
        mt, ml = J.targets(tg, lens); offs = torch.cat([torch.zeros(1, dtype=torch.long), ml.cumsum(0)])
        for k, y in auds.items():
            lp = J(y); l = J.ctc(lp, tg, lens, ok, cf.float() / 50)
            e = n = 0; fps = lp.shape[1] / 8.0
            for j in ok.nonzero().flatten().tolist():
                p = lp[j, :int(cf[j] / 50 * fps)].argmax(-1).tolist(); hyp = [t for i, t in enumerate(p) if t != 0 and (i == 0 or t != p[i - 1])]
                ref = mt[offs[j]:offs[j + 1]].tolist(); e += edit(hyp, ref); n += len(ref)
            r = res.setdefault(k, [0, 0, 0, 0]); r[0] += l.item() if l is not None else 0; r[1] += 1; r[2] += e; r[3] += n
for k, (l, nb, e, n) in res.items(): print(f"{k:12s} MMS CTC {l / nb:.3f}  letter CER {e / max(n, 1):.3f}")
