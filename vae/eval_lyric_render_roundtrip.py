#!/usr/bin/env python3
"""Full-song round trip for the lyric-conditioned renderer (vae/lyric_renderer.py) on the gate songs.

original -> 24 kHz -> VAE mu -> FSQ codes; lyric lines from the gate manifest (LRC times) -> Renderer.sample(codes, lines)
-> VAE decode -> C_roundtrip_<tag>.flac (30 s windows, 2 s cross-fade), scored by tools/yue2_roundtrip_eval.py.
--lyrics own | none | swap   (swap = the lines of the NEXT gate song, same timing positions: tests whether the words follow
                              the given lyrics; WER is then computed against the song's own lyrics, so swap should get WORSE)
usage: eval_lyric_render_roundtrip.py --vae out/vae/v1/G_s200000.pt --render out/render/lyr1/R_ema_s50000.pt --tag lyr1_50k [--cfg-lyr 3]
"""
import argparse, json, sys
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE  # noqa: E402
from vae.tokenizer import SemTokenizer  # noqa: E402
from vae.lyric_renderer import LyricRenderer  # noqa: E402
from vae.data import romanise, encode_text  # noqa: E402


def rs(x, a, b):
    g = gcd(a, b); return resample_poly(x, b // g, a // g, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vae", required=True); ap.add_argument("--render", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/gate20_shard0.jsonl"))
    ap.add_argument("--root", default=str(ROOT / "out/yue2_roundtrip_gate20")); ap.add_argument("--lyrics", default="own", choices=["own", "none", "swap"])
    ap.add_argument("--align", default="", choices=["", "line", "word"], help="default: what the renderer was trained with; word = "
                    "oracle word timing from the window's own audio (swap: the wrong words are force-aligned to this audio)")
    ap.add_argument("--cfg-lyr", type=float, default=1.0); ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--chunk", type=float, default=30.0); ap.add_argument("--overlap", type=float, default=2.0); args = ap.parse_args(); dev = "cuda"
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval()
    rk = torch.load(args.render, map_location="cpu"); ra = rk["args"]
    tk = torch.load(ra["tok"], map_location="cpu"); TK = SemTokenizer(levels=tuple(int(x) for x in tk["args"]["levels"].split(","))); TK.load_state_dict(tk["T"]); TK = TK.to(dev).eval()
    R = LyricRenderer(d=ra["d"], n=ra["n"], vocab=TK.fsq.n_codes); R.load_state_dict(rk["R"]); R = R.to(dev).eval()
    lat_m = rk.get("lat_m", torch.zeros(64)).to(dev); lat_s = rk.get("lat_s", torch.ones(64)).to(dev)   # normalised flow space (v2 runs)
    align = args.align or ra.get("align", "line")
    if align == "word":
        from vae.align_words import WordAligner
        AL = WordAligner(dev)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(args.manifest, encoding="utf-8")}
    ids = sorted(rows); nxt = {s: ids[(i + 1) % len(ids)] for i, s in enumerate(ids)}
    hop_s = V.hop / 24000

    @torch.no_grad()
    def render(seg, lines):
        x = torch.tensor(seg.T[None], device=dev); S = x.shape[-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = V.encode(x)[0].float(); c = TK.encode(mu)[1]
            if align == "word": lines = AL(x, [lines], mu.shape[1])[0]
            z = R.sample(c, [lines], steps=args.steps, cfg_lyr=args.cfg_lyr).float() * lat_s + lat_m; y = V.decode(z)[0].float()
        return F.pad(y, (0, max(0, S - y.shape[-1])))[..., :S].cpu().numpy().T

    for sid in ids:
        d = Path(args.root) / sid
        if not (d / "A_original.flac").exists(): continue
        src = rows[sid] if args.lyrics == "own" else rows[nxt[sid]]
        L = [] if args.lyrics == "none" else [(l["t0"], l["t1"], l["text"]) for l in src.get("lines") or []]
        x, sr = sf.read(d / "A_original.flac", dtype="float32", always_2d=True); x24 = rs(x, sr, 24000); x24 = x24[:len(x24) - len(x24) % V.hop]
        W = int(args.chunk * 24000) // V.hop * V.hop; O = int(args.overlap * 24000) // V.hop * V.hop; H = W - O
        out = np.zeros_like(x24); wsum = np.zeros((len(x24), 1), np.float32); fade = np.ones(W, np.float32); fade[:O] = np.linspace(0, 1, O); fade[W - O:] = np.linspace(1, 0, O)
        for s0 in range(0, max(1, len(x24) - O), H):
            seg = x24[s0:s0 + W]; n = len(seg); t0, t1 = s0 / 24000, (s0 + W) / 24000
            if n < W: seg = np.pad(seg, ((0, W - n), (0, 0)))
            lines = []
            for a_, b_, txt in L:
                if b_ <= t0 + 0.05 or a_ >= t1 - 0.05: continue
                e = encode_text(romanise(txt))
                if e: lines.append((int(round((a_ - t0) / hop_s)), e, int(round((b_ - t0) / hop_s))))   # raw frames: cut lines detectable
            y = render(seg, lines); f = fade[:n].copy()
            if s0 == 0: f[:O] = 1
            if s0 + W >= len(x24): f[max(0, n - O):] = 1
            out[s0:s0 + n] += y[:n] * f[:, None]; wsum[s0:s0 + n] += f[:, None]
        sf.write(d / f"C_roundtrip_{args.tag}.flac", np.clip(rs(out / np.maximum(wsum, 1e-6), 24000, 48000), -1, 1), 48000, subtype="PCM_16")
        print(sid, f"{len(x24) / 24000:.0f}s", flush=True)


if __name__ == "__main__":
    main()
