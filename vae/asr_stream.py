"""Frozen Qwen3-ASR-1.7B audio encoder as an extra lyric stream next to the VAE latent (docs/VAE_PLAN.md, test 2026-11-01).

Idea: the block-flow LM binds lyrics only under CFG because nothing it must predict depends strongly on the words (the
MERT-aligned dims read letters at CER ~0.45). An ASR encoder's states are close to a function of what is sung and when, so
predicting them next to the VAE latent (latent 64 + 32 here, decoded from the first 64 only) should force the LM to read the
lyrics at w = 1. Same idea as MSVAE's ASR-aligned half, but as a predicted side stream instead of a retrained VAE.

Encoder facts (qwen_asr/core/transformers_backend/modeling_qwen3_asr.py): Whisper log-mel (16 kHz, 128 bins, 100 Hz) ->
3 stride-2 convs on independent 1 s chunks (13 frames per chunk, positions restart per chunk) -> 24 bidirectional layers
over fixed 8 s windows (up to 8 s lookahead) -> ln_post (1024) -> proj1 / proj2 to the LLM (2048).
Here: bounded lookahead exactly like VAE.mert_feats_local (the frames of blocks [j, j+keep) come from a `win` s window
ending `ahead` s after block j+keep), one batched forward over all windows with our own attention (the reference forward
passes no mask under sdpa, so batched windows would attend to each other), 13 Hz frame centres resampled to 12.5 Hz.
Weights stay frozen; Apache-2.0.
"""
import importlib.util, json, sys, types
from pathlib import Path
import torch, torch.nn.functional as F, torchaudio

QWEN = "/models/rwkv-music/models/Qwen3-ASR-1.7B"
_TB = "/root/envs/qwen-asr-venv/lib/python3.12/site-packages/qwen_asr/core/transformers_backend"
SR = 16000; HOP = 160; RATE = 12.5


def _modeling():
    """the qwen_asr transformers backend without its package __init__ (which pulls the processor and its deps)"""
    name = "qwen3_asr_tb"
    if f"{name}.modeling_qwen3_asr" in sys.modules: return sys.modules[f"{name}.modeling_qwen3_asr"]
    pkg = types.ModuleType(name); pkg.__path__ = [_TB]; sys.modules[name] = pkg
    for sub in ("configuration_qwen3_asr", "modeling_qwen3_asr"):
        spec = importlib.util.spec_from_file_location(f"{name}.{sub}", f"{_TB}/{sub}.py"); m = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.{sub}"] = m; spec.loader.exec_module(m)
    return sys.modules[f"{name}.modeling_qwen3_asr"]


def _mel_filters(n_mels=128):
    from transformers.audio_utils import mel_filter_bank
    return torch.from_numpy(mel_filter_bank(num_frequency_bins=201, num_mel_filters=n_mels, min_frequency=0.0, max_frequency=8000.0,
                                            sampling_rate=SR, norm="slaney", mel_scale="slaney")).float()       # [201, n_mels], = WhisperFeatureExtractor


class AsrStream(torch.nn.Module):
    def __init__(s, path=QWEN, pca=None, dev="cuda"):
        super().__init__()
        mq = _modeling(); from qwen3_asr_tb.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
        cfg = Qwen3ASRAudioEncoderConfig(**json.load(open(Path(path) / "config.json"))["thinker_config"]["audio_config"])
        cfg._attn_implementation = "sdpa"; s.enc = mq.Qwen3ASRAudioEncoder(cfg)
        from safetensors import safe_open
        idx = json.load(open(Path(path) / "model.safetensors.index.json"))["weight_map"]; pre = "thinker.audio_tower."; sd = {}
        for f in sorted({v for k, v in idx.items() if k.startswith(pre)}):
            with safe_open(str(Path(path) / f), "pt") as h:
                for k in h.keys():
                    if k.startswith(pre): sd[k[len(pre):]] = h.get_tensor(k)
        s.enc.load_state_dict(sd, strict=True); s.enc.requires_grad_(False)
        s.register_buffer("mel_fb", _mel_filters(cfg.num_mel_bins)); s.register_buffer("win", torch.hann_window(400))
        p = torch.load(pca, map_location="cpu") if pca else None
        s.register_buffer("pca_m", p["mean"].float() if p else None); s.register_buffer("pca_P", p["P"].float() if p else None)
        s.dim = cfg.d_model if p is None else s.pca_P.shape[1]
        s.to(dev).eval()

    def logmel(s, x):
        """x [N,S] 16 kHz -> [N,128,S/160] Whisper log-mel (per-item max - 8 dB floor, (x + 4) / 4)"""
        st = torch.stft(x.float(), 400, HOP, window=s.win, return_complex=True)[..., :-1].abs() ** 2
        m = (s.mel_fb.T @ st).clamp_min(1e-10).log10()
        m = torch.maximum(m, m.amax((1, 2), keepdim=True) - 8.0)
        return (m + 4.0) / 4.0

    @torch.no_grad()
    def hidden(s, mel):
        """mel [N,128,F], F a multiple of 100 -> ln_post states [N, 13 F/100, 1024]; full attention inside each item"""
        E = s.enc; N, C, Fr = mel.shape; nc = Fr // 100; assert nc * 100 == Fr, Fr
        x = mel.view(N, C, nc, 100).permute(0, 2, 1, 3).reshape(N * nc, 1, C, 100)
        x = F.gelu(E.conv2d1(x)); x = F.gelu(E.conv2d2(x)); x = F.gelu(E.conv2d3(x))
        b, c, f, t = x.shape
        x = E.conv_out(x.permute(0, 3, 1, 2).reshape(b, t, c * f)) + E.positional_embedding.positional_embedding[:t].to(x.dtype)
        x = x.reshape(N, nc * t, -1)
        for L in E.layers:
            a = L.self_attn; h = L.self_attn_layer_norm(x); B, T, D = h.shape
            q, k, v = (p(h).view(B, T, a.num_heads, -1).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
            x = x + a.out_proj(F.scaled_dot_product_attention(q, k, v, scale=a.scaling).transpose(1, 2).reshape(B, T, D))
            x = x + L.fc2(L.activation_fn(L.fc1(L.final_layer_norm(x))))
        return E.ln_post(x)

    def project(s, h):
        return h if s.pca_P is None else (h.float() - s.pca_m) @ s.pca_P

    @torch.no_grad()
    def local(s, wav, sr, T, blk=4, keep=2, ahead=0.64, win=4.0, bs=256, raw=False):
        """wav [B,C,S] or [B,S] at sr -> [B,T,dim] at 12.5 Hz: frames of blocks [j, j+keep) from the window
        [end of block j+keep - (win - ahead), end of block j+keep + ahead] only (same bound as VAE.mert_feats_local)"""
        x = wav.float().mean(1) if wav.dim() == 3 else wav.float()
        if sr != SR: x = torchaudio.functional.resample(x, sr, SR)
        B = x.shape[0]; fr = int(SR / RATE); L = blk * fr; nB = -(-T // blk)                  # 1280 samples per latent frame
        A = int(round(ahead * SR)) // L * L; Wt = int(round(win * SR)) // SR * SR; P = Wt - A; assert P >= keep * L
        x = F.pad(x, (P, A + L * keep))
        st = [(j + keep) * L for j in range(0, nB, keep)]
        segs = torch.stack([x[:, a:a + Wt] for a in st], 1).reshape(-1, Wt)                      # [B*nw, Wt]
        nc = Wt // SR; src = torch.tensor([c + min(8 * k + 4, 98) / 100 for c in range(nc) for k in range(13)], device=x.device)
        f0 = (P - keep * L) // fr; tgt = (torch.arange(f0, f0 + keep * blk, device=x.device) + 0.5) / RATE   # own frames' centres (s)
        i1 = torch.searchsorted(src, tgt).clamp(1, len(src) - 1); i0 = i1 - 1
        w1 = ((tgt - src[i0]) / (src[i1] - src[i0])).clamp(0, 1)[None, :, None]
        out = []
        for i in range(0, segs.shape[0], bs):
            with torch.autocast("cuda", dtype=torch.bfloat16): h = s.hidden(s.logmel(segs[i:i + bs])).float()
            h = h[:, i0] * (1 - w1) + h[:, i1] * w1                                               # [n, keep*blk, 1024]
            out.append(h if raw else s.project(h))
        h = torch.cat(out).reshape(B, -1, keep * blk, out[0].shape[-1]).flatten(1, 2)[:, :T]
        return h
