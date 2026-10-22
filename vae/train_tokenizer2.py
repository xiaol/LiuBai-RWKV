#!/usr/bin/env python3
"""Phase B v2: hybrid lyric-aware tokens, trained end-to-end with the generative renderer as the decoder.

  input   VAE latent mu (12.5 Hz x 64)  +  MERT-v2 layer 12 (the most lyric-rich representation in vae/probe_lyrics.py,
          continuous CER 0.40 vs 0.45 for mu), 25 Hz pooled 2:1 to 12.5 Hz, per-crop instance-normalised
  encoder BiRWKV (d 512, 6 blocks) -> FSQ (default 8,8,8,8,6,5 = 92,160 codes; --levels)
  losses  rectified-flow loss of a Renderer conditioned on the quantised code VECTORS (straight-through, so the codes are
          shaped by what the renderer needs, instead of an MSE regressor that collapses to the mean)
          + ctc_w * lyric CTC read from the code vectors (same reader as the VAE / probe)
The renderer trained here is usable directly (cond = code vectors); the LM later predicts the code ids.
usage: train_tokenizer2.py --vae out/vae/v1/G_s200000.pt --out out/tok2/h1 [--ctc-w 1.0] [--steps 100000]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import VAE, N_CTC  # noqa: E402
from vae.rwkv import BiBlock  # noqa: E402
from vae.tokenizer import FSQ  # noqa: E402
from vae.renderer import Renderer  # noqa: E402
from vae.data import Crops, collate  # noqa: E402
from vae.train import edit  # noqa: E402
YUE2 = ROOT / "models/yue2"


class HybridTok(nn.Module):
    def __init__(s, levels, d=512, n=6, mert_dim=1024, latent=64):
        super().__init__(); s.fsq = FSQ(levels); nl = len(levels)
        s.inp_mu = nn.Linear(latent, d); s.inp_m = nn.Linear(mert_dim * 2, d)          # 2 MERT frames (25 Hz) per latent frame
        s.enc = nn.ModuleList(BiBlock(d) for _ in range(n)); s.to_q = nn.Linear(d, nl)
        s.ctc_in = nn.Linear(nl, 256 * 4); s.ctc = nn.ModuleList(BiBlock(256) for _ in range(2)); s.ctc_out = nn.Linear(256, N_CTC)

    def encode(s, mu, m):                              # mu [B,T,64], m [B,2T,1024] -> q [B,T,nl], ids [B,T]
        B, T, _ = mu.shape; m = m[:, :2 * T].reshape(B, T, -1)
        h = s.inp_mu(mu) + s.inp_m(m)
        for b in s.enc: h = b(h)
        return s.fsq(s.to_q(h))

    def lyric_logp(s, q):
        B, T, _ = q.shape; h = s.ctc_in(q).view(B, T * 4, -1)
        for b in s.ctc: h = b(h)
        return F.log_softmax(s.ctc_out(h).float(), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/wangyiyun.jsonl"))
    ap.add_argument("--levels", default="8,8,8,8,6,5"); ap.add_argument("--ctc-w", type=float, default=1.0); ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--steps", type=int, default=100000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rd", type=int, default=768); ap.add_argument("--rn", type=int, default=12); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5000)
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1); torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.vae, map_location="cpu"); a = ck["args"]
    V = VAE(ch=a["ch"], mults=tuple(int(x) for x in a["mults"].split(",")), d=a["d"], n_rwkv=a["n_rwkv"]); V.load_state_dict(ck["G"]); V = V.to(dev).eval().requires_grad_(False)
    from transformers import AutoModel, AutoFeatureExtractor
    proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
    mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval().requires_grad_(False)
    levels = tuple(int(x) for x in args.levels.split(","))
    TK = HybridTok(levels).to(dev); R = Renderer(d=args.rd, n=args.rn, cond_dim=len(levels)).to(dev)
    params = list(TK.parameters()) + list(R.parameters()); opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    print(f"tokenizer {sum(p.numel() for p in TK.parameters()) / 1e6:.1f}M ({TK.fsq.n_codes} codes) renderer {sum(p.numel() for p in R.parameters()) / 1e6:.1f}M", flush=True)
    step = 0; ckp = out / "last.pt"
    if ckp.exists():
        sd = torch.load(ckp, map_location=dev); TK.load_state_dict(sd["TK"]); R.load_state_dict(sd["R"]); opt.load_state_dict(sd["opt"]); step = sd["step"]; print("resumed", step, flush=True)

    @torch.no_grad()
    def features(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = V.encode(x)[0].float()
            inp = proc([w.cpu().numpy() for w in x.mean(1).float()], sampling_rate=24000, return_tensors="pt")
            h = mert(**{k: v.to(dev) for k, v in inp.items()}, output_hidden_states=True).hidden_states[args.layer].float()
        h = F.interpolate(h.transpose(1, 2), size=2 * mu.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        return mu, (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-5)

    def ctc_loss(lp, tg, lens, ok, cf):
        idx = ok.nonzero().flatten()
        if len(idx) == 0: return torch.zeros((), device=dev)
        offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]); tgt = torch.cat([tg[offs[i]:offs[i + 1]] for i in idx.tolist()]).to(dev)
        return F.ctc_loss(lp[idx].transpose(0, 1), tgt, torch.clamp(cf[idx], max=lp.shape[1]).to(dev), lens[idx].to(dev), blank=0, zero_infinity=True)

    mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80, power=1.0).to(dev)
    mdb = lambda y, x: ((torch.log(mel(y.mean(1)) + 1e-5) - torch.log(mel(x.mean(1)) + 1e-5)).abs().mean() * 20 / math.log(10)).item()
    vd = Crops(args.manifest, holdout=0, max_rows=1000, seed=12345); vit = iter(vd); val = [collate([next(vit) for _ in range(16)]) for _ in range(4)]
    @torch.no_grad()
    def evaluate():
        TK.eval(); R.eval(); r = dict(code_ctc_cer=0, mel_render=0, lr_render=0, code_ppl=0); e = c = 0; hist = torch.zeros(TK.fsq.n_codes, device=dev)
        for i, (wav, tg, lens, ok, cf) in enumerate(val):
            x = wav.to(dev); mu, m = features(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q, ids = TK.encode(mu, m); lp = TK.lyric_logp(q.float())
                if i < 2:
                    z = R.sample(q.float(), T=mu.shape[1], steps=24); S = x.shape[-1]; y = V.decode(z.float()).float()
                    y = F.pad(y, (0, max(0, S - y.shape[-1])))[..., :S]; r["mel_render"] += mdb(y, x) / 2
                    r["lr_render"] += float(np.mean([torch.corrcoef(b)[0, 1].item() for b in y])) / 2
            hist += torch.bincount(ids.flatten(), minlength=TK.fsq.n_codes)
            offs = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
            for j in ok.nonzero().flatten().tolist():
                p = lp[j, :cf[j]].argmax(-1).tolist(); hyp = [t for k, t in enumerate(p) if t != 0 and (k == 0 or t != p[k - 1])]
                ref = tg[offs[j]:offs[j + 1]].tolist(); e += edit(hyp, ref); c += len(ref)
        pr = hist / hist.sum(); pr = pr[pr > 0]; r["code_ppl"] = math.exp(-(pr * pr.log()).sum().item()); r["code_ctc_cer"] = e / max(c, 1)
        TK.train(); R.train(); return r

    dl = iter(torch.utils.data.DataLoader(Crops(args.manifest, seed=21 + step), batch_size=args.bs, num_workers=args.workers, collate_fn=collate, prefetch_factor=4, persistent_workers=True))
    log = open(out / "log.txt", "a"); t0 = time.time(); acc = {}
    while step < args.steps:
        step += 1; wav, tg, lens, ok, cf = next(dl); x = wav.to(dev, non_blocking=True)
        for g in opt.param_groups: g["lr"] = args.lr * min(1, step / 2000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        mu, m = features(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, ids = TK.encode(mu, m); lf = R.loss(mu, q); lc = ctc_loss(TK.lyric_logp(q.float()), tg, lens, ok, cf)
        loss = lf + args.ctc_w * lc
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        for k, v in dict(flow=lf, ctc=lc).items(): acc[k] = acc.get(k, 0) + float(v)
        if step % 100 == 0:
            msg = f"step {step} " + " ".join(f"{k} {v / 100:.4f}" for k, v in acc.items()) + f" {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            print(msg, flush=True); log.write(msg + "\n"); log.flush(); acc = {}
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate(); msg = f"EVAL step {step} " + " ".join(f"{k} {v:.4f}" for k, v in e.items()); print(msg, flush=True); log.write(msg + "\n"); log.flush()
            torch.save(dict(TK=TK.state_dict(), R=R.state_dict(), opt=opt.state_dict(), step=step, args=vars(args)), out / "last.tmp"); os.replace(out / "last.tmp", ckp)


if __name__ == "__main__":
    main()
