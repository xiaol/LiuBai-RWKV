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

## Redesign 2026-10-22: LiuBai-RWKV = block-flow RWKV on the continuous latent + monotonic lyric cursor

User: SOTA, no YuE2 components, a novel RNN solution, beat Suno v6. Evidence that drove it:
* Token capacity test (Whisper gate20, `logs/rt_tokv2_q2q4.log`): CER 1 / 2 / 4 FSQ codes per frame = 0.855 / 0.764 / 0.709 vs
  VAE v2 0.478. Four times the bits closes ~40 % of the gap: the discrete step is where the words die, not only for capacity.
* Literature (full texts, 2026-10-22): MSVAE (2610.17187) — unaligned acoustic VAE + untimed lyrics = PER 139 %, ASR-aligned
  latent subspace = 13.6 % untimed / 6.9 % timed; DiffRhythm 2 (2510.22950) — no tokenizer, block flow matching on VAE latents,
  PER 0.11 (block 5) → 0.23 (block 100); StepAudio 3 Music / 2607.20253 — tokenizers that keep lyrics are 0.6B+ Conformers with
  CTC multi-task, renderers 0.9–8B. No RWKV / linear-attention continuous song LM and no lyric pointer for songs found.

Design (`vae/cursor_lm.py`): causal RWKV-7 steps once per block of 4 latent frames (0.32 s), a flow head (adaLN MLP) samples
the next block; no tokenizer, no separate renderer. Lyrics enter through a **cursor** into the romanised letter string:
block i sees letters [c_i − 4, c_i + 20), an advance head predicts how many letters block i sings (c_{i+1} = c_i + a_i), and the
flow head is conditioned on h_i + emb(a_i). Recall is replaced by an O(1) read, which a fixed-state RNN does exactly; the cursor
trajectory is also word-timed LRC for free and lets upcoming lyrics be edited live. Labels: letter-level MMS forced alignment
per LRC line (`vae/align_letters.py`; per-line windows, spans capped at 2 s + 0.2 s/letter). Baseline mode `prefix`: whole
letter string before the blocks (DiffRhythm-2-style untimed conditioning, must be recalled from the state).

Gate 1 (running 2026-10-23 01:20, 60k steps ≈ 18 h each, 208M, 24 s crops of 300k vocal + 33k instrumental rows,
`data/real_corpus/manifest/pilot_clm_train.jsonl`; val = test shard `slim_val.jsonl`): `out/clm/cursor1` (GPU 1),
`out/clm/prefix1` (GPU 2), logs `logs/clm_{cursor,prefix}1.log`. Teacher-forced eval every 2.5k: flow loss own / swap / null
lyrics (gap_swap must be > 0). Audio gate: `vae/eval_cursor_lm.py --ckpt out/clm/<run>/M_ema_s20000.pt` — 4 s real prompt,
20 s generated, Whisper letter CER vs the given lyrics for own / swap / none. Pass: cursor own CER < 0.6 and clearly below
swap and none, and below prefix.
Next if it passes: own singing-ASR encoder on the LRC corpus (alignment teacher + reward, replaces MMS/MERT), VAE v3 with an
MSVAE-style partitioned ASR/chroma latent at 48 kHz, 2.9B cursor LM, forked-state section GRPO, WildSongBench vs Suno v6.

### Gate 1 result and VAE v3 (2026-10-23)
Gate 1 at 25k steps (`out/clm_eval/{cursor1,prefix1}_25k`, 24 held-out crops, 4 s real prompt + 20 s generated, Whisper letter
CER): cursor own 0.855 / swap-vs-given 0.845 / none 0.840, prefix own 1.065 / none 0.910; chance ≈ 0.84, originals 0.41–0.49.
Teacher-forced gap_swap plateaued at 0.03 from 15k. Together with gate 0 (word-timed renderer, own 0.966 / swap 0.941 / none
0.937): two different generators, both given the right letters at the right time, ignore them → the v2 latent is the problem,
as MSVAE reports for unaligned VAEs. Runs stopped (25k checkpoints `out/clm/*/s25k.pt`). User: no own ASR; MERT-v2 is fine
for the first version (CC BY-NC).

**VAE v3** (`vae/run_v3.sh`, `out/vae/v3`, init v2, 4 GPUs, 60k steps ≈ 25 h, launched 2026-10-23 ~09:10): v2 losses + latent
dims 0–31 → MLP head → cosine to MERT-v2 L12 (online, instance-normalised, 2 frames per latent frame, w 2), dims 32–63 → chroma
(w 1) (`vae/align_targets.py`; heads at 10× LR). Eval adds `al_mert` / `al_chroma` (cosine on clean mu). Gate: clean round trip
at v2 level (Whisper gate20 CER ≤ 0.50, cos ≥ 0.97), then a short cursor run on the v3 latent: gap_swap must clearly exceed
cursor1's 0.03 and the audio own CER must drop below none / swap.

### VAE v4, leak fix and lyric CFG (2026-10-25/26)
**VAE v4** (`vae/run_v4.sh`, semantic-first posterior: dims 0–31 = frozen MERT-v2 L12 projection + bounded acoustic
correction, 60k steps): gate20 clean WER 0.440 (v2 0.478), cos 0.968; latent probe CER v4 0.456, v4[0:32] 0.471, v2 0.513,
MERT L12 0.434. Its latent is **not causal**: MERT attends over the whole crop, so the cursor LM's context and the eval prompt
carried future audio (`docs/REVIEW_2026-10-25.md`). Fixes: `vae/eval_cursor_lm.py --causal-prompt` (prompt encoded from its
own audio) and seeding after `load_vae` (eval crops were not identical across runs before).

Audio gate on 96 held-out crops, causal prompt, Whisper letter CER medians (originals 0.426), `out/clm_eval/*c96*`:

| LM | CFG | own | swap (vs given) | own < none | swap follows given |
|---|---|---|---|---|---|
| cursor1 25k (v2) | 1 | 0.806 | 0.833 | 63/96 | 51/96 |
| cursor1 25k (v2) | 3 | 0.506 | 0.577 | 93/96 | 86/96 |
| cursor1 25k (v2) | 5 | 0.349 | 0.395 | 94/96 | 91/96 |
| cursor_v4 40k (leaky latents) | 2 | 0.521 | 0.467 | 88/96 | 84/96 |
| cursor_v4 40k (leaky latents) | 3 | 0.389 | 0.378 | 94/96 | 93/96 |

Without guidance no LM follows swapped lyrics; **lyric CFG** (null = NULL letter windows + null advance) makes both sing what
they are given, gate passed at CFG ≥ 3. At equal CFG v4 beats v2 (CFG 3, paired: lower own CER on 70/96, median −0.12).
CER below the originals at CFG 5 is suspicious (stripped accompaniment / over-articulation?): judge by listening
(`samples/cursor_cfg_20261026/player.html`, generated clips only) and add an audio-similarity score before picking a scale.

**cursor_v4b** (`scripts/run_cursor_v4b.sh`, running): v4 + `VAE.encode_local` (MERT on 3 s windows, 2 blocks per window,
≤ ~1 s lookahead; leak exactly 0 beyond the bound, `vae/check_causal.py`), lyrics cleaned by `tools/clean_lyrics.py`
(18.7 % of train songs: credits, titles, singer tags, role prefixes, inline translations, furigana; 145 val dups dropped),
flow loss on the MERT dims × 3 (`--sem-w 3`), lyric-dim gaps `sgap_*` in the eval, 20k steps.
