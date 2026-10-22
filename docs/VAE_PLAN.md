# LiuBai own stack: VAE → lyric-aware tokens → RWKV LM → RWKV renderer

Decided 2026-10-11 (user: "best way, not cheap way"). Replaces the YuE2 token space as the product token space; YuE2 stays as
teacher and quality bar. Corpus: NetEase real songs (≈2M incoming, >100k h; manifest `data/real_corpus/manifest/wangyiyun.jsonl`,
LRC line timestamps for every sung line).

## Why not stay in YuE2 space
* YuE2 ships no audio→token encoder; our inverse tokenizers reach 16–19 % exact codes. Round trip on 20 held-out real songs
  (shard 0): lyric WER/CER vs lyrics — original 0.36, YuE2-Vae only 0.46, best token round trip 0.62 (R1-a).
* No objective in that chain rewards intelligible words (flow loss, audio loss and token CE all reward sound).
* YuE2 weights are CC BY-NC 4.0 — an open release needs our own renderer anyway.

## Phase A — VAE (running: `vae/`)
`vae/model.py`: Oobleck conv codec (Snake, weight norm, strides 2·2·4·4·5·6 = 1920) + 4 bidirectional RWKV-7 blocks at the latent
rate on both sides (`vae/rwkv.py`, official `rwkv7_clampw` CUDA kernel, forward + reversed pass). Pilot: 24 kHz stereo →
12.5 Hz × 64-dim latent (half YuE2's frame rate → half the LM tokens per song). Losses: multi-scale mel (×15), multi-res STFT,
KL 1e-4, complex-STFT discriminator (hinge + feature matching) after 3k steps.

RWKV-specific signals, A/B/C/D pilot (40k steps, bs 8 × 8 s crops, one GPU each, `vae/run_pilot.sh`):

| run | lyric CTC on the latent | latent predictability (causal RWKV prior) |
|---|---|---|
| pilotA_base | probe only (detached) | probe only |
| pilotB_ctc | gradient to encoder (w 1.0) | probe only |
| pilotC_prior | probe only | encoder loss (w 1.0) |
| pilotD_both | yes | yes |

The probes are identical in every run, so `ctc_cer` (greedy CTC CER of a 2.7M BiRWKV reader on the latent, held-out shard-0
crops) and `prior_nmse` (normalised next-frame error of a 6M causal RWKV) are comparable across variants, next to `mel_db`.
CTC targets: romanised (uroman: pinyin-like for CJK) text of the LRC lines sung completely inside the crop.

Gate to scale up (48 kHz, larger channels, full corpus): mel/STFT close to the baseline (no quality cost) **and** a clear
`ctc_cer` / `prior_nmse` gain; then the full-song round trip on the 20 gate songs must reach YuE2-Vae (CER ≤ 0.46, MERT
cos ≥ 0.98, LTAS ≤ 2.3 dB).

### Phase A result and the v2 retrain (2026-10-20)
v1 (`out/vae/v1`, 171M, 200k steps) matches YuE2-Vae on the gate songs (CER 0.453 vs 0.458, MERT cos 0.983) but its latent
space is **too fragile for a generator**: posterior std = 0.2 % of the latent std (KL 1e-4 too small), and decoding
mu + 10 % relative noise already raises the MMS letter CER from 0.68 to 0.84 (35 %: 0.88; a 3-frame blur: 0.85). Every
renderer we trained lands at latent NMSE 0.31–0.36, i.e. exactly where the words are gone, which is why token renderers,
the lyric renderer and the audio-judged fine-tune all ended at gate20 CER ≈ 0.97 (`vae/diag_fragility.py`).
**v2** (`out/vae/v2`, init v1): KL 1e-2; 80 % of decodes perturbed with relative noise s² ~ U(0, 0.35) (scaled by the
sample's own latent variance, detached) and 30 % of those also time-blurred; MMS_FA CTC on the decoded perturbed audio
(`vae/mms_judge.py`) next to the latent lyric CTC. New eval columns `mms_n{0,0.1,0.3}` / `mel_n{…}`.
Gate: `mms_n0.3` ≤ ~0.70 (v1 0.88) with `mms_n0` and the clean round trip no worse than v1. Only then retrain renderers.

## Phase B — semantic tokens
FSQ codes (8–16k) at 12.5 Hz from VAE latent + MERT, trained with the same lyric CTC reader on the *codes* (+ MERT alignment).
Gate: CER of a token-only reader; round trip through the renderer ≤ 0.5.

## Phase C — RWKV renderer (NAR)
Flow matching, RWKV blocks bidirectional within ~4 s chunks with carried state (streaming, unbounded length), conditioned per
frame on tokens + the LM's hidden state + the time-aligned phoneme stream; losses: flow + audio-domain (VAE-decoded vs original,
the term that fixed R1-a's 8.5 dB spectral tilt) + lyric CTC on generated latents. Then 2–4-step distillation.

## Phase D/E — LM and RL
Stage-1 RWKV (G1k 2.9B base) on the full corpus in our tokens, section/line layout from LRC; YuE2-generated songs as extra
lyric-following data. Then GRPO with ASR (lyric CER) + style rewards. Target: lyric CER < 0.3, then WildSongBench vs Suno v6.

## Interim (YuE2 space) results kept as fallback
Audio-domain loss port (`scripts/train_joint_online.py --aux-w`), runs `out/joint/aux_r1a`, `out/joint/aux_v9`; gate20 numbers in
`out/yue2_roundtrip_gate20/eval.json`.
