#!/usr/bin/env python3
"""Joint R1 head + R2 NAR-LoRA training with YuE2's flow-matching teacher, with ONLINE MERT features (PLAN §8.6).

Same objective as train_joint_teacher.py (real window -> head -> straight-through codec embeddings -> frozen YuE2 AR+NAR
(+LoRA) -> flow MSE vs the song's true VAE latents; minted soft-CE anchor; minted flow windows with true tokens), but the
real-audio pool is no longer limited to tracks with 38 MB of stored MERT features:
  * stored real dirs   data/yue2_real/<shard>/<id>/{mert_L_12_16_20_23.npy, latent_true.npy, prefix.npy}   (as before)
  * G1 shards          data/yue2_corpus/suno94k/shard-NNNNN.{latents.npz,jsonl}[,audio/<id>.opus]
                       features are computed on the fly by a producer thread: opus audio if present (preferred: MERT on
                       real audio), else VAE.decode(latent) (≈ 11 % exact-token agreement with real-audio features, see §9)
Held-out eval stays on stored-feature real tracks (8, md5 order) + minted val, so numbers are comparable with out/joint/v2.
usage: train_joint_online.py --out out/joint/v3 --head-init out/joint/v2/head_best.pt --lora-init out/joint/v2/lora_best.pt
         --latent-shards 'data/yue2_corpus/suno94k/shard-*.latents.npz' [--wins-per-song 2] [--rescan-every 500] ...
"""
import argparse, glob, hashlib, json, math, random, sys, threading, queue, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
import torchaudio
from scipy.signal import resample_poly
from math import gcd
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
YUE2 = ROOT / "models/yue2"
sys.path.insert(0, str(ROOT / "scripts"))
from train_inverse_tokenizer import InverseTokenizer, Track  # noqa: E402
from online_mert import OnlineMERT, MemTrack  # noqa: E402
VOCAB = 32768
FEAT = "mert_L_12_16_20_23.npy"


class V4Head(nn.Module):
    """Mothersuperior v4/v5/v8/v9 tokenizer head (8-layer d512, 512-frame windows) on MERT layer 20, fed our [B, K=4, T, 1024]
    instance-normalised L12/16/20/23 windows (same per-track, per-channel normalisation as their instnorm)."""
    WIN, D, L, H = 512, 512, 8, 8
    def __init__(s):
        super().__init__(); s.tok = nn.Module(); t = s.tok
        t.inp = nn.Linear(1024, s.D); t.pos = nn.Parameter(torch.zeros(1, s.WIN, s.D))
        layer = nn.TransformerEncoderLayer(s.D, s.H, 4 * s.D, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        t.enc = nn.TransformerEncoder(layer, s.L, enable_nested_tensor=False); t.norm = nn.LayerNorm(s.D); t.head = nn.Linear(s.D, VOCAB)
        s.cfg = dict(arch="v4", win=s.WIN)
    def forward(s, x):
        t = s.tok; h = x[:, 2]; return t.head(t.norm(t.enc(t.inp(h) + t.pos[:, :h.shape[1]])))


class LoRALinear(nn.Module):
    def __init__(s, base, r):
        super().__init__(); s.base = base
        s.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        s.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(s, x): return s.base(x) + ((x.float() @ s.A.T) @ s.B.T).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--head-init", required=True)
    ap.add_argument("--train-head", type=int, default=1); ap.add_argument("--train-lora", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lora-init", default="none")
    ap.add_argument("--win", type=int, default=256); ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--real-dirs", nargs="*", default=[str(ROOT / "data/yue2_real/shard-00000"), str(ROOT / "data/yue2_real/shard-00001")])
    ap.add_argument("--latent-shards", default=str(ROOT / "data/yue2_corpus/suno94k/shard-*.latents.npz"), help="glob of G1 latent shards ('' = none)")
    ap.add_argument("--exclude-shards", default="2", help="G1 shard numbers never used for training (test material)")
    ap.add_argument("--min-sec", type=float, default=30); ap.add_argument("--max-sec", type=float, default=300)
    ap.add_argument("--require-opus", type=int, default=1, help="1 = only use G1 songs with stored opus audio (MERT on real audio); 0 = also VAE-decode songs without audio")
    ap.add_argument("--online-p", type=float, default=-1, help="prob. that a real step uses an online (G1) song; -1 = proportional to pool sizes")
    ap.add_argument("--wins-per-song", type=int, default=2, help="training windows (= optimizer steps) per online song")
    ap.add_argument("--rescan-every", type=int, default=500, help="re-glob the G1 shards every N steps (G1 is still running)")
    ap.add_argument("--minted-dirs", nargs="+", default=[str(ROOT / "data/yue2_minted/tracks"), str(ROOT / "data/yue2_minted_own")])
    ap.add_argument("--max-minted", type=int, default=0)
    ap.add_argument("--lr-head", type=float, default=3e-5); ap.add_argument("--lr-lora", type=float, default=3e-5); ap.add_argument("--lr-io", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ce-weight", type=float, default=1.0); ap.add_argument("--ce-bs", type=int, default=4)
    ap.add_argument("--minted-flow-p", type=float, default=0.25); ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--hold", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-jsonl", default="", help="G1-layout shard jsonl (excluded from training) whose first --hold songs (md5 order) form a "
                    "second held-out set, MERT from their own audio; when given, checkpoints are selected on it (real_nar2)")
    ap.add_argument("--head-arch", default="r1", choices=["r1", "v4"], help="v4 = Mothersuperior 8-layer d512 head on MERT L20 (v4/v5/v8/v9 releases, {'model': sd})")
    ap.add_argument("--aux-w", type=float, default=0.0, help="audio-domain loss weight (Mothersuperior joint_v6: VAE-decoded clean estimate vs the original recording)")
    ap.add_argument("--aux-tmax", type=float, default=0.4); ap.add_argument("--aux-fr", type=int, default=150); ap.add_argument("--aux-margin", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=1000, help="also keep head/lora_s<step>.pt every N steps (pick by round trip, not by flow loss)")
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = True
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.protocol import CODEC_OFFSET, MUSIC_END, SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from yue2.nar import attention as nar_attention

    # ---- YuE2 (frozen) + LoRA on the NAR branch ----
    snap = str(YUE2 / "YuE2-3B")
    model = YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(dev)
    model.requires_grad_(False); bb = model.model; tok = YuE2TextTokenizer(snap + "/qwen.tiktoken")
    Ecodec = bb.embed_tokens.weight[CODEC_OFFSET:CODEC_OFFSET + VOCAB]
    lora_params = []
    for layer in bb.layers:
        for mod, names in ((layer.nar_self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.nar_mlp, ("gate_proj", "up_proj", "down_proj"))):
            for n in names: l = LoRALinear(getattr(mod, n), args.rank); setattr(mod, n, l); lora_params += [l.A, l.B]
    model.vae2llm.float(); model.llm2vae.float(); io_params = list(model.vae2llm.parameters()) + list(model.llm2vae.parameters())
    def load_lora(path):
        ck = torch.load(path, map_location=dev)
        with torch.no_grad():
            for p, v in zip(lora_params, ck["lora"]): p.copy_(v.to(dev))
            model.vae2llm.load_state_dict({k: v.float() for k, v in ck["io"]["vae2llm"].items()}); model.llm2vae.load_state_dict({k: v.float() for k, v in ck["io"]["llm2vae"].items()})
    def save_lora(path): torch.save({"lora": [p.detach().cpu() for p in lora_params], "io": {"vae2llm": model.vae2llm.state_dict(), "llm2vae": model.llm2vae.state_dict()}, "rank": args.rank}, path)
    if args.lora_init != "none": load_lora(args.lora_init); print("loaded NAR LoRA", args.lora_init, flush=True)
    for p in lora_params + io_params: p.requires_grad_(bool(args.train_lora))

    # ---- head ----
    hck = torch.load(args.head_init, map_location=dev)
    if args.head_arch == "v4":
        head = V4Head().to(dev); head.tok.load_state_dict(hck["model"]); layers = [12, 16, 20, 23]
    else:
        head = InverseTokenizer(**hck["cfg"]).to(dev); head.load_state_dict(hck["model"]); layers = hck["layers"]
    assert layers == [12, 16, 20, 23], layers
    head.requires_grad_(bool(args.train_head)); head.train(bool(args.train_head))
    groups = []
    if args.train_head: groups.append({"params": list(head.parameters()), "lr": args.lr_head, "weight_decay": 0.05})
    if args.train_lora: groups += [{"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.0}, {"params": io_params, "lr": args.lr_io, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95)); base_lrs = [g["lr"] for g in opt.param_groups]
    NB = torch.tensor(np.load(ROOT / "data/yue2_minted/codec_nbr_idx.npy").astype(np.int64), device=dev)
    NW = torch.softmax(torch.tensor(np.load(ROOT / "data/yue2_minted/codec_nbr_cos.npy"), device=dev) / 0.05, dim=1)

    # ---- stored-feature data (as train_joint_teacher.py) ----
    def load_real(d):
        z = np.load(d / "latent_true.npy").astype(np.float32); tr = Track(d / FEAT, None, len(z))
        return dict(name=d.name, tr=tr, lat=z[:tr.n], prefix=[int(v) for v in np.load(d / "prefix.npy")])
    real_dirs = sorted(d for rd in args.real_dirs for d in Path(rd).iterdir() if (d / "prefix.npy").exists() and (d / FEAT).exists())
    real_dirs.sort(key=lambda d: hashlib.md5(d.name.encode()).hexdigest()); hold_dirs, real_dirs = real_dirs[:args.hold], real_dirs[args.hold:]
    real = [load_real(d) for d in real_dirs if len(np.load(d / "latent_true.npy", mmap_mode="r")) >= args.win]; hold = [load_real(d) for d in hold_dirs]
    def load_minted(d):
        y = np.load(d / "semantic.npy").astype(np.int64); tr = Track(d / FEAT, y, len(y))
        r = json.load(open(d / "request.json")) if (d / "request.json").exists() else {}
        it = dict(name=d.name, tr=tr, tok=tr.y, style=r.get("style", ""), lyrics=r.get("lyrics", ""))
        if (d / "latent.npy").exists(): it["lat_path"] = d / "latent.npy"
        return it
    minted_dirs = sorted(d for md in args.minted_dirs for d in Path(md).iterdir() if (d / FEAT).exists() and (d / "semantic.npy").exists())
    if args.max_minted: minted_dirs = minted_dirs[:args.max_minted]
    mheld = lambda d: int(hashlib.md5(d.name.encode()).hexdigest(), 16) % 20 == 0
    mtrain = [load_minted(d) for d in minted_dirs if not mheld(d)]; mval = [load_minted(d) for d in minted_dirs if mheld(d)]
    mflow = [m for m in mtrain if "lat_path" in m]

    # ---- online (G1) pool: song metadata only; features computed by the producer thread ----
    excl = {int(x) for x in args.exclude_shards.split(",") if x}
    online = {}                                    # id -> dict(npz, caption, lyrics, n, opus)
    def rescan():
        n_new = 0
        for npz in sorted(glob.glob(args.latent_shards)) if args.latent_shards else []:
            tag = Path(npz).name.split(".")[0]; shard = int(tag.split("-")[1]); jl = Path(npz).with_name(f"{tag}.jsonl")
            if shard in excl or not jl.exists(): continue
            adir = Path(npz).with_name(f"{tag}.audio")
            for line in open(jl):
                r = json.loads(line)
                if r.get("error") or r["id"] in online or "n_frames" not in r: continue
                sec = r["n_frames"] / 25; lyr = r.get("lyrics", "") or ""
                if not (args.min_sec <= sec <= args.max_sec) or "[instrumental]" in lyr.lower() or len(lyr) < 50: continue
                op = adir / f"{r['id']}.opus"
                if not op.exists() and r.get("audio_path") and Path(r["audio_path"]).exists(): op = Path(r["audio_path"])  # real corpus: original flac
                if args.require_opus and not op.exists(): continue
                online[r["id"]] = dict(npz=npz, caption=r.get("caption", ""), lyrics=lyr, n=int(r["n_frames"]), opus=op if op.exists() else None); n_new += 1
        return n_new
    rescan(); print(f"real stored {len(real)} held {len(hold)} | online {len(online)} ({sum(1 for v in online.values() if v['opus'])} with opus) | minted train {len(mtrain)} val {len(mval)} | win {args.win} rank {args.rank}", flush=True)
    om = OnlineMERT(dev=dev)
    npz_cache = {}
    def prefix_of(caption, lyrics, sid):
        style = " ".join(caption.split())[:1500] or "pop"; lyrics = lyrics.strip() or "[instrumental]"
        return token_prefixes(SongRequest(style=style, lyrics=lyrics, cot="off", seed=1, id=sid), tok)
    def build_online(sid, pool=None):
        it = (pool or online)[sid]
        if it["npz"] not in npz_cache: npz_cache[it["npz"]] = np.load(it["npz"])
        z = npz_cache[it["npz"]][sid].astype(np.float32)
        if it["opus"] is not None:
            a, sr = sf.read(it["opus"], dtype="float32")
            if a.ndim > 1: a = a.mean(1)
            if sr != 24000: g = gcd(sr, 24000); a = resample_poly(a, 24000 // g, sr // g).astype(np.float32)
            X = om.feats_from_mono24(a, T25=len(z))
        else:
            X = om.feats_from_latent(z)
        n = min(len(z), X.shape[1]); tr = MemTrack(X[:, :n], None, n)
        return dict(name=sid, tr=tr, lat=z[:n], prefix=prefix_of(it["caption"], it["lyrics"], sid), online=True,
                    audio=str(it["opus"]) if it["opus"] is not None and str(it["opus"]).endswith((".flac", ".mp3", ".wav")) else None)
    hold2 = []
    if args.hold_jsonl:
        hp = {}; hz = Path(args.hold_jsonl).with_name(Path(args.hold_jsonl).name.replace(".jsonl", ".latents.npz"))
        rows = [json.loads(l) for l in open(args.hold_jsonl, encoding="utf-8")]
        rows = sorted((r for r in rows if "n_frames" in r and r["n_frames"] >= args.win and len(r.get("lyrics") or "") >= 50 and r.get("audio_path")),
                      key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())[:args.hold]
        for r in rows: hp[r["id"]] = dict(npz=str(hz), caption=r.get("caption", ""), lyrics=r["lyrics"], n=r["n_frames"], opus=Path(r["audio_path"]))
        hold2 = [build_online(sid, hp) for sid in hp]; print(f"hold2: {len(hold2)} songs from {args.hold_jsonl}", flush=True)
    q = queue.Queue(maxsize=3); stop = threading.Event()
    def producer():
        s = torch.cuda.Stream()
        while not stop.is_set():
            if not online: time.sleep(5); continue
            sid = random.choice(list(online.keys()))
            try:
                with torch.cuda.stream(s): it = build_online(sid)
                s.synchronize(); q.put(it)
            except Exception as e:
                print(f"  online FAIL {sid}: {str(e)[:120]}", flush=True); online.pop(sid, None)
    threading.Thread(target=producer, daemon=True).start()

    # ---- YuE2 forward pieces ----
    def ar_layer(layer, x, cos_, sin_):
        q_, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos_, sin_); h = nar_attention(q_[0], k[0], v[0], causal=True)
        x = x + layer.self_attn.o_proj(h.flatten(1)[None]); return x + layer.mlp(layer.post_attention_layernorm(x)), k[0], v[0]
    def nar_layer(layer, h, ak, av, ncos, nsin):
        q_, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(h), ncos, nsin); a = nar_attention(q_[0], torch.cat((ak, k[0])), torch.cat((av, v[0])))
        h = h + layer.nar_self_attn.o_proj(a.flatten(1)[None]); return h + layer.nar_mlp(layer.nar_pre_mlp_layernorm(h))
    def flow_loss(prefix, codec_emb, x1, t, noise, grad_ar, grad_nar, full=False):
        pre = bb.embed_tokens(torch.tensor([prefix], device=dev))[0]; end = bb.embed_tokens(torch.tensor([MUSIC_END], device=dev))
        x = torch.cat((pre, codec_emb.to(pre.dtype), end), 0)[None]; Lq = x.shape[1]; cos_, sin_ = bb.rotary_emb(torch.arange(Lq, device=dev)[None]); cache = []
        if grad_ar:
            for layer in bb.layers: x, k, v = checkpoint(ar_layer, layer, x, cos_, sin_, use_reentrant=False); cache.append((k, v))
        else:
            with torch.no_grad():
                for layer in bb.layers: x, k, v = ar_layer(layer, x, cos_, sin_); cache.append((k, v))
        T = codec_emb.shape[0]; xt = t * noise + (1 - t) * x1; target = noise - x1; N = T + 2
        ncos, nsin = bb.rotary_emb(torch.arange(Lq, Lq + N, device=dev)[None])
        pe = model.latent_pos_embed(torch.arange(N, device=dev).clamp(max=model.config.max_latent_frames - 1))[None]
        sh = model._shift_t_value(float(np.clip(np.log(t / (1 - t)), -20, 20)), dev, torch.bfloat16)
        h = model.vae2llm(F.pad(xt, (0, 0, 1, 1))[None].float()).to(torch.bfloat16) + model.time_embedder(sh.expand(N))[None] + pe
        for layer, (ak, av) in zip(bb.layers, cache):
            h = checkpoint(nar_layer, layer, h, ak, av, ncos, nsin, use_reentrant=False) if (grad_ar or grad_nar) else nar_layer(layer, h, ak, av, ncos, nsin)
        v_hat = model.llm2vae(bb.norm(h)[0, 1:-1].float()); l = F.mse_loss(v_hat, target)
        return (l, v_hat, xt) if full else l

    # ---- audio-domain auxiliary loss (port of Mothersuperior joint_v6.py) ----
    HOP = 1920
    if args.aux_w > 0 or args.hold_jsonl:
        from yue2.modeling_vae import YuE2VAE
        vae_aux = YuE2VAE.from_pretrained(str(YUE2 / "YuE2-Vae"), decoder_only=True, device=dev, local_files_only=True); vae_aux.decoder.requires_grad_(False)
        _mel = torchaudio.transforms.MelSpectrogram(48000, n_fft=2048, hop_length=480, n_mels=128, power=1.0).to(dev)
        _wins = {n_: torch.hann_window(n_, device=dev) for n_ in (512, 1024, 2048)}
    def read48(path, f0, nfr):
        """original recording, latent frames [f0, f0+nfr) -> float32 [2, nfr*HOP] @48 kHz (None if short / unreadable)"""
        try:
            info = sf.info(path); sr = info.samplerate; a0 = int(f0 * HOP * sr / 48000); a1 = int((f0 + nfr) * HOP * sr / 48000) + 64
            x, _ = sf.read(path, start=a0, stop=a1, dtype="float32", always_2d=True)
        except Exception: return None
        x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]; a = torch.from_numpy(x.T.copy()).to(dev)
        if sr != 48000: a = torchaudio.functional.resample(a, sr, 48000)
        a = a[:, :nfr * HOP]; return a if a.shape[1] == nfr * HOP else None
    def spec_loss(pred, ref):
        pm, rm = pred.mean(0), ref.mean(0); mel = (torch.log(_mel(pm) + 1e-5) - torch.log(_mel(rm) + 1e-5)).abs().mean(); mr = 0
        for n_, w in _wins.items():
            A = torch.stft(pm, n_, hop_length=n_ // 4, window=w, return_complex=True).abs(); B = torch.stft(rm, n_, hop_length=n_ // 4, window=w, return_complex=True).abs()
            mr = mr + ((A - B).norm() / (B.norm() + 1e-6) + (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().mean()) / 3
        side = lambda x: torch.log(((x[0] - x[1]) / 2).pow(2).mean() + 1e-7) - torch.log(((x[0] + x[1]) / 2).pow(2).mean() + 1e-7)
        return mel + 0.5 * mr + 0.5 * (side(pred) - side(ref)).abs()
    def decode_center(xt, v_hat, t, s_win):
        """clean-latent estimate of the centre aux_fr frames (+margin) of a window starting at s_win -> (wav [2,S], abs start frame)"""
        T = xt.shape[0]; fr, mg = min(args.aux_fr, T - 2 * args.aux_margin), args.aux_margin
        s0 = max(0, (T - fr) // 2 - mg); x1 = (xt - t * v_hat)[s0:s0 + fr + 2 * mg]
        wav = vae_aux.decoder(x1.T[None].float())[0][:, mg * HOP:(mg + fr) * HOP]
        return wav, s_win + s0 + mg, fr
    def aux_loss(it, xt, v_hat, t, s_win):
        if not it.get("audio"): return None
        wav, f0, fr = decode_center(xt, v_hat, t, s_win); ref = read48(it["audio"], f0, fr)
        return spec_loss(wav, ref) if ref is not None and ref.shape == wav.shape else None
    @torch.no_grad()
    def audio_eval(items):
        """held-out audio-domain distances of the t=0.2 clean estimate vs the original: log-mel L1 dB and LTAS dB (the one that tracked listening)."""
        mel_t = ltas_t = cnt = 0
        for it in items:
            if not it.get("audio"): continue
            n = len(it["lat"]); s = max(0, min(n // 2, n - args.win)); _, m, z = window(it, "lat", s)
            with torch.autocast("cuda", dtype=torch.bfloat16): idx = head(T_(m)[None])[0].float().argmax(-1)
            g = torch.Generator(device="cpu").manual_seed(7); noise = torch.randn(len(z), 64, generator=g).to(dev)
            _, vh, xt = flow_loss(it["prefix"], Ecodec[idx], torch.tensor(z, device=dev), 0.2, noise, False, False, full=True)
            wav, f0, fr = decode_center(xt, vh, 0.2, s); ref = read48(it["audio"], f0, fr)
            if ref is None or ref.shape != wav.shape: continue
            mel_t += (20 / math.log(10)) * (torch.log(_mel(wav.mean(0)) + 1e-5) - torch.log(_mel(ref.mean(0)) + 1e-5)).abs().mean().item()
            P = lambda x: 10 * torch.log10(torch.stft(x.mean(0), 2048, hop_length=512, window=_wins[2048], return_complex=True).abs().pow(2).mean(1) + 1e-10)
            ltas_t += (P(wav) - P(ref)).abs().mean().item(); cnt += 1
        return mel_t / max(cnt, 1), ltas_t / max(cnt, 1)
    def st_embed(logits):
        p = torch.softmax(logits.float(), -1); idx = p.argmax(-1); hard = F.one_hot(idx, VOCAB).float()
        return ((hard + (p - p.detach())).to(Ecodec.dtype)) @ Ecodec, idx
    def sample_t(): return float(np.clip(torch.sigmoid(torch.randn(())).item(), 0.02, 0.98))
    def T_(m): return m if torch.is_tensor(m) else torch.tensor(m, device=dev)
    def window(item, key, s=None):
        n = min(len(item[key]), item["tr"].n); s = random.randint(0, max(0, n - args.win)) if s is None else s
        return s, item["tr"].window(s, args.win), item[key][s:s + args.win]
    def soft_ce(lg, y):
        logp = F.log_softmax(lg.float(), -1); return (0.7 * (-logp.gather(1, y[:, None])[:, 0]) + 0.25 * (-(logp.gather(1, NB[y]) * NW[y]).sum(1)) + 0.05 * (-logp.mean(1))).mean()

    @torch.no_grad()
    def hold_loss(items):
        g = torch.Generator(device="cpu").manual_seed(123); tot = 0; cnt = 0
        for it in items:
            n = len(it["lat"])
            for frac in (0.3, 0.6):
                s = int(n * frac) if n * frac + args.win <= n else max(0, n - args.win); _, m, z = window(it, "lat", s)
                if len(z) < args.win: continue
                with torch.autocast("cuda", dtype=torch.bfloat16): idx = head(T_(m)[None])[0].float().argmax(-1)
                noise = torch.randn(args.win, 64, generator=g).to(dev)
                for t in (0.2, 0.5, 0.8): tot += flow_loss(it["prefix"], Ecodec[idx], torch.tensor(z, device=dev), t, noise, False, False).item(); cnt += 1
        return tot / max(cnt, 1)

    def evaluate():
        head.eval(); g = torch.Generator(device="cpu").manual_seed(123); tot = 0; cnt = 0; reps = []
        for it in hold:
            n = len(it["lat"])
            for frac in (0.3, 0.6):
                s = int(n * frac) if n * frac + args.win <= n else max(0, n - args.win); _, m, z = window(it, "lat", s)
                if len(z) < args.win: continue
                with torch.autocast("cuda", dtype=torch.bfloat16): idx = head(T_(m)[None])[0].float().argmax(-1)
                reps.append(float((idx[1:] == idx[:-1]).float().mean())); noise = torch.randn(args.win, 64, generator=g).to(dev)
                for t in (0.2, 0.5, 0.8): tot += flow_loss(it["prefix"], Ecodec[idx], torch.tensor(z, device=dev), t, noise, False, False).item(); cnt += 1
        t1 = tot_ = 0
        for it in mval[:40]:
            _, m, y = window(it, "tok")
            with torch.autocast("cuda", dtype=torch.bfloat16): idx = head(T_(m)[None])[0].float().argmax(-1).cpu().numpy()
            t1 += (idx[:len(y)] == y).sum(); tot_ += len(y)
        e2 = hold_loss(hold2) if hold2 else float("nan")
        head.train(bool(args.train_head)); return tot / max(cnt, 1), t1 / max(tot_, 1), float(np.mean(reps)) if reps else 0.0, e2

    def save_all(tag):
        if args.head_arch == "v4": torch.save({"model": head.tok.state_dict(), "arch": "v4", "step": st, "tag": tag}, out / f"head_{tag}.pt")
        else: torch.save({"model": head.state_dict(), "cfg": head.cfg, "layers": layers, "step": st, "tag": tag}, out / f"head_{tag}.pt")
        if args.train_lora: save_lora(out / f"lora_{tag}.pt")

    def real_step(it, t, noise):
        s, m, z = window(it, "lat")
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = head(T_(m)[None])[0]
        emb, _ = st_embed(lg); noise = noise[:len(z)]
        ln, vh, xt = flow_loss(it["prefix"], emb, torch.tensor(z, device=dev), t, noise, bool(args.train_head), bool(args.train_lora), full=True)
        la = aux_loss(it, xt, vh, t, s) if (args.aux_w > 0 and t <= args.aux_tmax) else None
        if la is not None: ln = ln + args.aux_w * la; aux_hist.append(float(la))
        lc = torch.zeros((), device=dev)
        if args.train_head and args.ce_weight > 0:
            xs, ys = [], []
            for _ in range(args.ce_bs):
                mi = random.choice(mtrain); _, mm, yy = window(mi, "tok")
                if mm.shape[1] < args.win: mm = np.pad(mm, ((0, 0), (0, args.win - mm.shape[1]), (0, 0))); yy = np.pad(yy, (0, args.win - len(yy)), constant_values=-100)
                xs.append(mm); ys.append(yy)
            xb = torch.tensor(np.stack(xs), device=dev); yb = torch.tensor(np.stack(ys), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): lgb = head(xb)
            msk = yb != -100; lc = args.ce_weight * soft_ce(lgb[msk], yb[msk])
        return ln, lc

    aux_hist = []
    st = 0; e0, a0, r0, h0 = evaluate(); msg = f"EVAL step 0 real_nar {e0:.4f} minted_top1 {a0:.4f} real_repeat {r0:.3f} real_nar2 {h0:.4f}"
    if hold2: m0, l0 = audio_eval(hold2); msg += f" real_mel {m0:.2f}dB ltas {l0:.2f}dB"
    print(msg, flush=True)
    log = open(out / "log.txt", "a"); log.write(msg + "\n"); sel = (lambda e, h: h) if hold2 else (lambda e, h: e); best = sel(e0, h0); t0 = time.time()
    json.dump(vars(args) | dict(real=len(real), hold=len(hold), online=len(online), minted=len(mtrain)), open(out / "config.json", "w"), indent=1)
    pending = []; n_online = n_stored = 0; wait_s = 0.0
    for st in range(1, args.steps + 1):
        mult = min(1.0, st / args.warmup) * (0.5 * (1 + math.cos(math.pi * st / args.steps)) * 0.9 + 0.1)
        for g_, b in zip(opt.param_groups, base_lrs): g_["lr"] = b * mult
        t = sample_t(); noise = torch.randn(args.win, 64, device=dev); src = "minted"
        if args.train_lora and random.random() < args.minted_flow_p:
            it = random.choice(mflow); s, m, y = window(it, "tok")
            z = np.load(it["lat_path"]).astype(np.float32)[s:s + args.win]; n = min(len(z), len(y))
            ln = flow_loss(prefix_of(it["style"], it["lyrics"], it["name"]), Ecodec[torch.tensor(y[:n], device=dev)], torch.tensor(z[:n], device=dev), t, noise[:n], False, True); lc = torch.zeros((), device=dev)
        else:
            p_on = args.online_p if args.online_p >= 0 else len(online) / max(1, len(online) + len(real))
            if pending: it = pending.pop(); src = "online"
            elif online and random.random() < p_on:
                tw = time.time(); it = q.get(); wait_s += time.time() - tw; src = "online"
                pending.extend([it] * (args.wins_per_song - 1))
            else: it = random.choice(real); src = "stored"
            ln, lc = real_step(it, t, noise); n_online += src == "online"; n_stored += src == "stored"
        loss = ln + lc; opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g_ in opt.param_groups for p in g_["params"]], 1.0); opt.step()
        if st <= 3 or st % 25 == 0:
            print(f"step {st} {src} nar {ln.item():.4f} ce {lc.item():.3f} t {t:.2f} {time.time() - t0:.0f}s (wait {wait_s:.0f}s) online/stored {n_online}/{n_stored} aux {np.mean(aux_hist[-25:]) if aux_hist else 0:.3f} (n {len(aux_hist)}) mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if st % args.rescan_every == 0:
            k = rescan()
            if k: print(f"  rescan: +{k} online songs -> {len(online)}", flush=True)
        if st % args.eval_every == 0 or st == args.steps:
            e, a, rp, h = evaluate(); msg = f"EVAL step {st} real_nar {e:.4f} minted_top1 {a:.4f} real_repeat {rp:.3f} real_nar2 {h:.4f}"
            if hold2: mm_, ll_ = audio_eval(hold2); head.train(bool(args.train_head)); msg += f" real_mel {mm_:.2f}dB ltas {ll_:.2f}dB"
            msg += f" {time.time() - t0:.0f}s"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if sel(e, h) < best: best = sel(e, h); save_all("best")
            save_all("last")
        if args.save_every and st % args.save_every == 0: save_all(f"s{st}")
    stop.set(); print(f"RESULT best {'real_nar2' if hold2 else 'real_nar'} {best:.4f} (start {sel(e0, h0):.4f})", flush=True)


if __name__ == "__main__":
    main()
