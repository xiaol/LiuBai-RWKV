# LiuBai own stack: VAE → lyric-aware tokens → RWKV LM → RWKV renderer

> **⚠ SERIOUS BUG (found 2026-10-31): every RWKV time-mix trained before 2026-10-31 was dead.** `vae/rwkv.py` TimeMix and
> `vae/gated_nar.py` GTimeMix zero-initialised both `o` and `g2`: a fixed point (o's grad is scaled by g2's output, g2's by o),
> so WKV never trained, and weight decay kept it at 0. Checked: `o` / `g2` exactly 0 in VAE v1–v5 (enc / dec BiRWKV, LyricCTC,
> LatentPrior), every CursorLM (cursor1, cursor_v4b, e1_mlp), every GatedNARLM (e1_gated_s{0,1}, e1_lattn_s0), and so in the
> renderers / tokenizers / sing_asr built on BiBlock. **Every result below that involves an "RWKV" here is a conv / token-shift +
> FFN model without state.** That includes "the gated read-only NAR reads S", the E1 gated-vs-MLP gate, the lattn gate, the toy
> arms, and the "BiRWKV" readers and probes. The 3B port (`vae/gated_rwkv7.py`, pretrained time-mix) is not affected. Fix:
> `rwkv.init_gate` (official g1 = 0, g2 = orthogonal × 0.1), regression test in `vae/test_gated_nar.py`, `rwkv.revive_wkv`
> for warm starts (train.py `--init`). Plan: lat_v5 precompute stopped, **VAE v6 = v5 + live WKV** (`vae/run_v6.sh`), then
> retrain the LMs on v6. Live WKV in the **decoder, LyricCTC and prior only**: the encoder's time-mix stays dead (exact fixed
> point), because a live backward WKV would let each latent frame see the whole crop's future and break `encode_local`'s
> bounded lookahead. The v6 encoder is therefore conv + 4 frame-wise FFN blocks (one-frame-back shift), as in v1–v5.

## Goal and current plan (2026-10-31) — read this first
**Goal (user, 2026-10-31):** a SOTA song model, caption + lyrics → full song, that beats **Suno v6 and YuE2**. The method is
free to change. **Benchmark:** WildSongBench (192 prompts, 94 zh / 98 en) with YuE2's protocol: PER from greedy Qwen3-ASR
(tone-marked pinyin / English phonemes), lower-PER output of 2 candidates per prompt; SongBench / SongEval quality; MuQ-MuLan,
AllMusicCaps and Qwen3-Omni (Q3O) for prompt match. Published bar (YuE2 paper 2609.33757): YuE2 PER 0.0844, MiniMax Music 3
0.0627, ACE-Step 1.5 0.0746; Suno v5 leads text alignment (MuLan / AMCaps). Our gates (20 s continuations, VAE round trip) are
proxies; a full-song WildSongBench harness has to exist before any SOTA claim.

**Method now:** no discrete audio tokens and **no audio→token tokenizer** (the "Phase B — semantic tokens" section below is
superseded). LiuBai-H: an RWKV-7 backbone (3B pretrained = main line, 208M for ablations) predicts our own VAE latent
(12.5 Hz × 64; dims 0–31 semantic = MERT-v2 L12-aligned + lyric CTC, dims 32–63 acoustic) in blocks of 4 frames (0.32 s):
A rows causal over clean blocks, F rows = flow expert on noisy frames. Lyrics enter as text tokens in the prefix plus the
cursor's letter window per block. Latent I/O projections already exist (`vae/gated_rwkv7.py`: `inp` 4×64 → C for A rows,
`fin` 64 → C for F rows, `out` C → 64 zero-init, flow-time AdaLN). Our VAE encoder is needed only to encode the corpus and audio
prompts. Our latent merges the jobs of YuE2's semantic codes (dims 0–31) and YuE2-Vae (whole latent).

**Status:** VAE **v6 on hold** (user 2026-10-31: test the backbone first; v5 is used as is, it passed its gate: WER 0.460 vs
YuE2-Vae 0.458). Running 2026-10-31: `out/clm/e1_gated_s0_wkv` (E1 gated head re-run with live WKV, v4 latents, `--sem-w 3`,
20k steps) and a 6-rank `tools/precompute_latents.py`.

**YuE2 for comparison** (paper §2.2, §3.3, App. B.1/B.2; checked 2026-10-31):
* Sequence: `[caption + lyrics text] [ABC_START] lead sheet [ABC_END] [MUSIC_START] semantic codes [MUSIC_END]` | continuous
  VAE latents (NAR). Lyrics are ordinary text tokens, whole string, untimed, not interleaved; codes use a disjoint id range;
  type masks stop cross-emission; text and lyrics dropped separately for CFG. Words appear three times: lyric text (what to
  sing), semantic codes (what is sung, lyric-CTC supervised), VAE latent (sound only, no lyric supervision).
* Semantic tokenizer (unreleased): targets = MuQ + Qwen2-Audio features fused and 4-level RVQ'd (16,384 per level) → 4 cached
  25 Hz code streams. Encoder = log-mel → ConvNeXt (25 Hz) → 24-layer Conformer (1024 wide). Stage 1 masked prediction of the
  4 streams (bidirectional, = MERT2-30s); branch 2-FS full songs, bidirectional (= released MERT-v2-FullSong); tokenizer branch:
  Stage 2 causal attention, Stage 3 + lyric CTC + mel + chroma on full songs, Stage 4 a 32,768 × 32-dim cosine CVQ between
  layers 13 and 14 with the CTC / mel / chroma heads kept above it. Deployed: front end + layers 0–13 + codebook, 375 bit/s.
  Its retention test probes tags / emotion / beat only, so how much lyric content the codes keep is not reported.
* At inference YuE2 does not need the tokenizer: the AR generates the codes from text; only training data preparation (and
  audio-input tasks: edit, cover, continuation) needs audio → codes.

**Next experiments (208M, matched steps, 96-crop causal gate, CFG 1 own CER decides):**
1. E1 re-run with live WKV (running), then the lattn arm re-run on the fixed init.
2. **Semantic-first F head:** within each block, flow on dims 0–31 first, then 32–63 conditioned on them (YuE2's c → z order
   inside one model), vs the joint head. Motivation: YuE2 binds lyrics on a compact, mostly phonetic stream; our flow loss is
   half acoustic. (Loss weighting of the semantic dims already exists: `--sem-w`, 3 in the current runs.)
3. **YuE2-Vae as a latent arm** (never tested): (a) gate20 fragility, YuE2-Vae latents + 30 % relative noise vs v5's 0.548;
   (b) the same backbone on YuE2-Vae latents (25 Hz, so blocks of 8 frames = 0.32 s) vs v5 latents at matched steps. If
   YuE2-Vae ties, the own-VAE work is not what makes the difference; if v5 wins, that is the first direct evidence that it is.
4. WildSongBench harness (Qwen3-ASR PER as YuE2, YuE2 run locally as the baseline, Suno v6 samples per prompt).
5. **Lyric-tuned MERT probe** (no VAE retrain yet): the semantic half is computed from *frozen* MERT-v2 L12, so its lyric
   content is capped by L12 (reader CER 0.399; v5 latent probe 0.45). YuE2's tokenizer fine-tunes those layers with lyric CTC
   (+ mel / chroma) before quantising. Fine-tune MERT-v2-FullSong layers 0–12 with uroman lyric CTC from the LRC corpus
   (+ mel / chroma heads), then the same small reader on L12: CER clearly below 0.399 → a VAE with the tuned MERT branch is
   worth it (re-encode the corpus); otherwise the latent is not the lyric bottleneck. Optional second half = tokenizer
   test: a 32k cosine VQ / FSQ between layers 13 and 14, heads above it, CTC CER from codes vs pre-VQ (YuE2 never reported
   it). Our earlier tok2 (`vae/train_tokenizer2.py`: frozen L12 + latent → BiRWKV with dead WKV → FSQ, CTC on codes, small
   flow renderer) lost the words (gate20 CER 0.71–0.86), but it differed from YuE2 in frozen MERT, quantizer placement,
   bitrate, a WKV-less encoder and renderer, and data, so it does not settle the question.
5b. **Decoder-only live-WKV fine-tune of v5** (when a GPU is free): encoder frozen, so the latent, the lat_v5 precompute and
   every backbone run stay valid; revive the decoder WKV only. Gate20 WER / LTAS / MERT cos vs v5. Target: the LTAS gap
   (2.87 vs YuE2-Vae 2.25 dB). May gain nothing.
5c. **v6 = one combined VAE retrain, near the 3B run, not before.** Candidates, each decided by its own evidence:
   live decoder WKV (5b); lyric-tuned MERT branch (5, if L12 CER drops clearly below 0.399); **25 Hz frame rate** (user
   suggestion 2026-10-31). For 25 Hz: native MERT rate (no 2-frame stacking), more room for fast lyrics and spectral detail,
   same rate as YuE2-Vae / MiniMax. Against: 2× backbone sequence (4 min = 6,000 frames; attention layers 4×), block of 8
   frames to keep 0.32 s, and gate20 shows no clear 12.5 Hz cost (fast-lyric half +0.015 WER, n = 20). Evidence to decide:
   the YuE2-Vae backbone arm (3b, 25 Hz at the same 0.32 s block) and a rap / fast-lyric gate set. One retrain only: the
   181k h corpus is re-encoded once.
6. **Symbolic plan (later):** YuE2's ABC lead sheet gives +15 points of expert preference (49.3 vs 34.6 %). Needs an
   audio → lead-sheet transcriber for training labels (their SheetSage2 is their own); candidate prefix span for LiuBai-H.

**Frame rate (12.5 Hz) — checked 2026-10-31, not an issue on current evidence.** Chosen to halve the backbone's steps
(4 min = 3,000 frames = 750 A tokens), never A/B'd against 25 Hz. Gate20 split by lyric rate (lyric words / song duration,
0.7–3.6 words/s, n = 20): slow half WER YuE2-Vae (25 Hz) 0.407 vs v5 0.396, fast half 0.508 vs 0.523; corr(rate, v5 − YuE2-Vae)
0.31 (n = 20, not significant). At most a small cost on fast songs. DiffRhythm 2 reconstructs at **5 Hz** with PER 0.121
(better than Stable Audio 2 VAE at 21.5 Hz, 0.148). Remaining risk: rap / fast melisma, not in gate20.

**Other systems (checked 2026-10-31, `/tmp/lit/`, MiniMax HF card):**
* **DiffRhythm 2** (2510.22950): Music VAE 24 kHz mono in → 48 kHz mono out, **5 Hz**, encoder = Stable Audio 2 (Oobleck conv,
  as ours), a transformer block before the decoder only (none after the encoder: bidirectional encoder context would leak
  future into blocks — same reason as our bounded-lookahead encoder), BigVGAN decoder, mel + STFT losses, MPD / MSD / CQT
  discriminators. No semantic dims in the VAE; semantics enter the generator through stochastic block REPA to MuQ. Generator:
  ~1B DiT, block flow matching (block 10 = 2 s; PER 0.11 at block 5 → 0.23 at 100), lyrics as untimed full text, DPO
  (w/o DPO PER 0.27 → 0.13). 70k h, 1.4M songs. Differences from ours: 2.5× lower rate, mono, acoustic-only latent, attention
  generator, DPO doing much of the lyric work.
* **MiniMax Music 3** (open weights, WSB PER 0.0627 = best public): 8-codebook RVQ tokenizer at 25 Hz (semantic codebook 16,384
  trained first, then 7 × 1,024 acoustic); **8B Global LLM (Qwen3-8B init)** predicts codebook 1 per frame, 0.6B Local LLM the
  other 7; synthesis fuses both LLMs' **hidden states** → 2.4B flow matching → Flow-VAE latent → 123M decoder, 32 kHz stereo.
  Discrete tokens are only the AR targets; the renderer reads continuous hidden states. Lyrics with section tags + a
  structured caption (metadata / vocals / arrangement). Pretrained text-LLM init is the same bet as our RWKV-7 3B.
* **YuE2 planning:** not a prompt CoT; the model itself writes an ABC lead sheet (melody + chords, ordinary text tokens)
  after text + lyrics and before the semantic codes. Training labels = SheetSage2 transcriptions of each song. Four tasks mixed
  in balanced ratio: score → codes → latents; codes → latents; score → latents; direct. Next-token loss on score and codes,
  flow loss on latents. Inference: create samples the score; edit uses a user-modified score; cover uses a score transcribed
  from the reference.

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

## Phase B — semantic tokens (SUPERSEDED 2026-10-22: no discrete tokens, see "Goal and current plan" at the top)
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

**cursor_v4b gate (2026-10-26, 20k steps, same 96 causal crops, medians, paired bootstrap 95 % CI):** CFG 1 own 0.892 / none
0.958 (no control without guidance, as before). CFG 3 own **0.401** (v2 cursor1 0.506, leaky cursor_v4 40k 0.389), swap vs
given 0.438; CFG 5 own **0.289** (v2 0.349, leaky 0.274). v4b − v2: −0.068 [−0.103, −0.046] at CFG 3, −0.068 [−0.095, −0.021]
at CFG 5 → the v4 advantage is not the leak. v4b − leaky v4: +0.05 (half the training steps). User listening: v4 sounds
better. **v4 chosen.**

### VAE v5 = v4 at 48 kHz (launched 2026-10-26 ~23:00, `vae/run_v5.sh`, `out/vae/v5`, 4 GPUs, bs 4/GPU, 80k steps)
Same 12.5 Hz × 64 latent so the cursor LMs carry over: one extra outer stride-2 stage (strides 2,2,2,4,4,5,6 = 3840, mults
1,1,2,4,8,16,32, 172.6M), init v4 (inner stages shifted by one; first conv and output conv reused; new stage fresh; D adds a
4096 scale), inner stages frozen for 3k steps, GAN from 2k. MERT and its targets get 24 kHz mono internally (`VAE._mono24`);
mel / STFT / chroma windows scaled ×2 (same seconds). `--enc local`: the semantic half is computed from bounded-lookahead MERT
in training too, so the decoder learns the latents the LM produces. `--teacher v4 --anchor-w 1`: normalised MSE from our mu
to frozen v4's mu on the 24 kHz crop (step 0: 0.55) keeps the latent space v4's. Data: full slim train (2.06M songs) with
cleaned lyrics (`data/real_corpus/manifest/v5_{train,val}_clean.jsonl`). Bandwidth (`tools/bandwidth_scan.py`, 2k random
songs, cut = highest 250 Hz bin within 60 dB of the 1–4 kHz level): median 21.5 kHz, 89 % ≥ 19 kHz, 96 % ≥ 16 kHz → the
audit's "12 kHz" was its own test, the corpus is mostly full-band; no bandwidth weighting (Crops supports a row weight "w").
Gate: held-out mel / ctc_cer / anchor, gate20 round trip at 48 kHz (Whisper CER ≤ v4's 0.44, full-band LTAS vs YuE2-Vae),
then decode cursor_v4b samples through v5 (should work if the anchor holds) and fine-tune the cursor LM on v5 latents.

## Planned 2026-10-27: LiuBai-H = hybrid cursor LM (YuE2's hybrid mask on a recurrent block LM)
User: novelty and SOTA are the goal, streaming is not. Not implemented yet. YuE2 reference (TeX §3.3, `docs/yue2_reference/
modeling_yue2.py:668`, `yue2_infer_pkg/nar.py`): one shared attention, AR→AR causal, AR→NAR blocked, NAR→AR full, NAR→NAR
bidirectional; separate AR / NAR experts (norms, QKV/O, MLP) per layer; NAR experts initialised from the AR ones after 60k AR-only
updates (random init, Qwen3-1.7B shape and Qwen vocab only).

Gaps in `vae/cursor_lm.py` it targets: (1) lyric binding comes from CFG, not the model (v4b CFG 1 own 0.892 vs none 0.958; CFG 5
0.289), and the flow head sees the 24-letter window only as one flattened vector summed into h_i; (2) no exact recall (236M A/B:
attention wins only on copying, 1.85 vs 2.75 nats) for returning choruses and long captions; (3) MLP head on one flattened block,
no access to the previous frames; (4) task is a 20 s continuation, SOTA needs text + lyrics → full song.

Design. Three position types:
* C = caption + whole letter string with line / section markers → small bidirectional encoder (~4 layers) → memory M.
* B = one token per block (today's RWKV stream), causal. Backbone 24 layers, pattern [RWKV, RWKV, RWKV, Attn] × 6.
* F = the 4 noisy frames of block i (NAR expert = flow head as a 6–8 layer transformer, d 1024, adaLN on t).

| query \ key | C | B | F |
|---|---|---|---|
| B_i | full | causal (≤ i) | blocked |
| F_i | cursor letters c_i−4 … c_i+20 only (contextualised from M) | taps of h_i from layers 6/12/18/24 + clean frames of block i−1 | bidirectional |

Difference from YuE2: F reads lyrics through the cursor, not the whole string, so timing stays a pointer (O(1)). Attention cost
is small: 3.125 blocks/s → 4 min ≈ 750 B + ~2k C positions (YuE2 24,576). Condition dropout for separate lyric / caption CFG.

Expected roles: letter cross-attention + previous frames in the head → binding without CFG, seams, sound; attention layers →
chorus recall, caption, full-song structure (not lyrics: RWKV bound lyrics better, 0.41 vs 0.28); cursor → timing (prefix-only
failed in the RNN, 1.065; untested in a hybrid).
Candidate claims (re-check literature): first linear-recurrent + attention continuous song LM; cursor for local lyric timing +
attention for global recall; block-rate attention. Main risk: hybrid + prefix-only lyrics matches the cursor → claim 2 dies.

Experiments (208M, matched steps, 96-crop causal gate, ≥ 3 seeds before claims):
* **E1 head**: MLP head vs attention head (letter cross-attention + previous block). Metric: own / swap / none CER at CFG 1 and 3,
  MERT cos. Contained change to `FlowHead` / `CursorLM.cond`. First.
* **E2 cursor necessity**: hybrid + cursor vs hybrid + prefix only. Decides the novelty claim.
* **E3 backbone**: pure RWKV vs 3:1 hybrid on 60–120 s crops with repeated sections. Metric: chorus-repeat self-similarity vs the
  original, CER of late vs early blocks (drift; DiffRhythm 2 PER 0.11 → 0.23).
* Then ~1B on v5 latents with caption conditioning, WildSongBench with YuE2's protocol. Deferred: whole-song bidirectional refine
  pass (SDEdit from t ≈ 0.5) — it is YuE2's NAR pass, little novelty; only if E3 shows drift.
Architecture alone is not SOTA: YuE2 also has 346k h and ≈ 0.5T positions; scale, data and PER-reward RL still matter.

### Revision 2026-10-27: main line = pretrained RWKV-7 3B backbone + gated read-only NAR expert
User: use the RWKV-7 3B pretrained weights (`models/rwkv-g1k-3b-temp`, `models/rwkv7-g1j-2.9b`) as the main backbone, with
the hybrid NAR / masked attention on top. RWKV layers cannot take a mask, so each cell of YuE2's mask is rebuilt inside one
interleaved sequence `[caption + lyrics] A_0 F_0,1..4 A_1 F_1,1..4 …` (A_i = AR block token that predicts block i, F_i,j =
noisy frame j of block i, NAR expert):

| YuE2 cell | RWKV-3B version |
|---|---|
| A→A causal | normal RWKV over the A tokens; their token shift skips the F frames (A_i reads A_{i−1}) |
| A→N blocked | **write gate**: on F rows force w = 1 (w_raw = −1e4), k = 0, a = 0 → the WKV update leaves S exactly unchanged, nothing reaches later A tokens |
| N→A full | F_i rows read S (after A_i) through their own r in every layer = recurrent analogue of NAR queries on the AR KV cache |
| N→N bidirectional | forward shift F_i,j ← F_i,j−1 (F_i,1 ← A_i) + a new backward shift F_i,j ← F_i,j+1 with zero-init mix (step 0 = pretrained 3B), in TimeMix and ChannelMix; 2 shifts per layer → the 4 frames see each other after 2 layers |
| AR / NAR experts | NAR = AR weights + LoRA on r / g / out / FFN + own norms + adaLN on t; LoRA zero-init. F's own k / v / w / a are unused (except the per-token bonus term, a self-only read) |
| shared masked attention | optional 4–8 attention layers, zero-init output proj, same mask; kept only if E3 says so |

Checked 2026-10-27 on the stock `rwkv7_clampw` kernel (GPU, random inputs, 12 blocks × 4 frames, `/tmp/tsmask/test_gate.py`):
A outputs and A gradients interleaved vs A-only bit-exact (max diff 0.0); F outputs = S_after(A_i)·r to bf16 rounding (0.03 on
values ≤ 7). So no kernel change: no state in / out, no state gradients, no snapshots; all NAR losses in one forward, and the NAR
loss still trains the A stream through S. Inference: S after A_i is fixed, so each flow step is a 4-token pass reading it.
Limits: F frames have no scratch memory (mixing among them only via shift + FFN); exact recall still missing (attention layers);
literature check for "read-only gated tokens as a recurrent mask" not done.
Dropped (user, 2026-10-27): the state fork (copy every layer's WKV + shift state after B_{i−1}, run F from the copy, N→N by
feeding the block twice). It needed a new kernel and ~21 MB snapshots per sampled block, and the gated version gives the same
read without either.
Toy gated vs fork (2026-11-07, `vae/toy_fork.py`, CPU, d 64, 3 layers, blk 16, 1500 steps, 2 seeds, `out/toy_fork/results.jsonl`):
frames j and j + 8 share a sign (beyond the shift reach of 6), history needed as in the earlier toy.

| arm | params | eval loss | mode err | pair agree (chance 0.5) | block agree (chance 0.004) |
|---|---|---|---|---|---|
| bi (gated) | 382k | 0.81 / 0.81 | 0.35 / 0.35 | 0.49 / 0.49 | 0.001 / 0.003 |
| fork1 (F writes into the copy, 1 pass) | 382k | 0.70 / 0.69 | 0.34 / 0.34 | 0.56 / 0.64 | 0.008 / 0.036 |
| fork2 (block fed twice) | 382k | 0.66 / 0.60 | 0.38 / 0.36 | 0.54 / 0.77 | 0.010 / 0.153 |
| lattn (gated + local attention) | 581k | 0.53 / 0.51 | 0.26 / 0.26 | 0.93 / 0.95 | 0.70 / 0.85 |

Read: scratch state lowers the flow loss and can carry in-block coordination (fork2 seed 1), but unreliably across seeds and with
no gain in per-frame precision; gated + local attention wins on every metric (at +52 % params in this toy). Keeps the drop; the
remaining reason to build snapshots is branching rollouts (section GRPO / search), not the training mask. Toy only.

"Mask token" = the noisy latent at flow time t (t = 1 fully masked); no discrete [MASK] (tokens lost the words: 0.855 vs 0.478).
Audio blocks enter via new `Linear(256 → 2560)` (A) / `Linear(64 → 2560)` per frame (F) + modality embedding; the cursor stays on
romanised letters for timing. Pretrained weights are expected to help the text side (multilingual captions / lyrics), not audio
(YuE2 used random init; our stage-1 3B had no random-init control). Candidate claim: gated read-only NAR tokens = the hybrid
mask on a pure recurrent model (literature check needed).

Cost: 4 min song ≈ 2k text + 750 A + 4 × 750 F (or F on K ≈ 64 sampled blocks only: + 256) positions (stage-1 already ran
g1k-3b at ctx 16k, DeepSpeed 2, 4 × A100). Inference ≈ 750 blocks × 32 flow steps × 3B over 4 tokens ×2 for CFG → minutes per
song; if too slow, NAR only in the top half of the layers or fewer steps.
Blockers: (1) the shift gather (A skips F, F forward / backward) in TimeMix + ChannelMix; (2) 3B LoRA dims 96/96/64/320 (env
overrides exist); (3) fla `chunk_rwkv7` still broken in rwkv_py312 (triton `AttrsDescriptor`), not needed now.
Order: (1) CPU unit tests of the gated stack (no leak A←F, block isolation, in-block bidirectional, incremental decode = parallel
forward, toy flow task) while the GPUs run VAE v5 — **done 2026-10-28**, see below; (2) 208M gated NAR on `cursor_lm.py` vs the MLP head (= E1); (3) 3B port,
warm start g1k-3b, train on v5 latents with captions; (4) E2 / E3 at 3B (or 1B if 3B is too slow on this box).

CPU results 2026-10-28 (`vae/gated_nar.py`, GPUs busy with VAE v5). `vae/test_gated_nar.py` (fp32 reference recurrence, all
pass): A outputs bit-identical under any F frames / flow times / no F rows at all; ∂A/∂F exactly 0; F loss reaches the A stream;
F of block j depends only on block j's frames and blocks < j; in-block receptive field full (bi), lower-triangular with mix_b = 0,
diagonal with iso; incremental inference (A_0..A_i + F_i) = training layout to 5e-7. Side fix: `vae/rwkv.py` `wkv7_ref` uses
mul + sum instead of einsum (CPU bmm backward was ~300× slower; matches the old code to 1e-7).
Toy flow task (`vae/toy_gated_nar.py`, d 64, 3 layers, 1500 steps, 2 seeds, `out/toy_gated_nar/results.jsonl`): block i =
±(roll of block i−1's mean + frame offset), one random sign per block. Needs F→A (history) and F→F (frames share the sign).

| arm | final loss | mode err | sign agree (chance 0.125) |
|---|---|---|---|
| bi (forward + backward shift) | 0.68 / 0.66 | 0.29 / 0.30 | 0.96 / 0.95 |
| causal (mix_b frozen 0) | 0.82 / 0.81 | 0.29 / 0.29 | 0.95 / 0.95 |
| iso (no in-block shift) | 0.66 / 0.63 | 0.17 / 0.17 | 0.14 / 0.12 |
| nohist (history zeroed) | 1.07 / 1.05 | 1.17 / 1.18 | 0.55 / 0.55 |

Read: the read-only gate passes history (mode err 1.17 → 0.29 with it); the in-block shift is what makes frames agree
(0.13 → 0.95). The backward shift lowers the loss (0.82 → 0.67) but not the sampled metrics: forward-only in-block shift already
coordinates 4 frames. Cost: mixing frames made each frame less precise here (mode err 0.29 vs 0.17 for iso); watch per-frame
quality at 208M. Toy, d 64 — mechanism check only, no claim about audio.

### Literature check and review of the LiuBai-H plan (2026-10-28)
Full texts in `/tmp/lit2` (recurrent masks) and `/tmp/lit3` (song SOTA, guidance). ✓ = mechanism read in the full text by me;
otherwise taken from a search agent's reading (abstract page opened, numbers not re-checked).

**Novelty of the gated read-only NAR is narrower than written above.**
* ✓ FLARE (2606.01774, 2026-06): AR → block-diffusion conversion of Gated DeltaNet + attention hybrids, one pass, clean causal
  stream + noisy blocks. Its words: on recurrent layers the clean / noisy mask "becomes a schedule of writes, state propagation,
  and state resets". Noisy block b starts from the clean boundary state and writes only into that copy (= the state fork we
  dropped), custom two-stream GDN kernels. → "hybrid AR / NAR mask on a linear-recurrent model in one forward" is published.
* ✓ B³D-RWKV (2605.25969, 2026-05): RWKV-7 7.2B, no architectural change; each block appears three times (masked, masked with
  loss, clean). Masked copies write the WKV state, so noise reaches later blocks; 3× sequence length; discrete text.
* Also cite: Fixed State, Long Reach (2609.11998, block diffusion on Mamba-2, one block per example); linear-attention dLLM
  retrofit (2608.06628, state updated only after a block is committed); speculative decoding on SSM / GDN (STree 2505.14969,
  2608.20961) = read without commit, inference only; TiDAR / BD3-LM / I-DLM = the attention mask template.
* Nothing found: "query-only" / zero-write tokens as a training mask in SSMs; continuous-latent AR + flow on RWKV / Mamba; RWKV
  text-to-music audio.
Defensible claim: *zero-write gating* (w = 1, k = 0, a = 0) on the stock RWKV-7 kernel replaces FLARE's state fork and B³D's
block triplication — no new kernel, no 3× length, exact A isolation — applied to continuous flow on audio with the full backbone
as flow head. Reviewer question it invites: F rows get no scratch state (only token shift), FLARE's noisy rows do. Needs the
ablation gated vs fork at 208M (fork = FLARE recurrence, reference implementation is enough at that size) even though the fork
is not the product.

**Song SOTA (WildSongBench PER from YuE2 Table 2, best of 2 × min of 4 ASR):** MiniMax Music 3 0.063, ACE-Step 1.5 (2602.00744,
LM planner + DiT) 0.075, YuE2 0.084, HeartMuLa 0.107, DiffRhythm 2 0.184 (closest to our design: block flow on a VAE),
SongBloom 0.192, LeVo 2 0.261, Muse (Qwen3-0.6B init, 7.8k h) 0.334. Diffusion models can top PER; block flow on a VAE so far
does not.

**Our real bottleneck is not addressed by the 3B port.** Lyric binding exists only under guidance (v4b CFG 1 own 0.892 vs none
0.958; CFG 5 0.289 with over-compression). Production systems run CFG 1.25–1.5 (HeartMuLa, SongBloom, LeVo 2, VibeVoice 1.3);
HeartMuLa already finds CFG 1.5 harsh to listeners. No paper sweeps lyric CFG against PER. What moved binding in the
literature is the objective: Khala (2605.01790) explicit alignment task PER 78.7 → 21.2 % (CN); JDCM flow loss through LM
readouts 20.3 → 14.5 %; MSVAE ASR-aligned latent. RL on PER polishes only (LeVo 2 DPO 10.6 → 9.2 %; recipe 2508.05011).
No music paper ablates text-LLM init vs scratch (YuE2: random init) → the 3B start is a bet on caption / multilingual text, not
on binding. Ordered changes to the plan:
1. **E1 as planned, but judged at CFG 1** (own vs none gap) besides CFG 3: the MLP head sees the letter window as one flattened
   vector summed into h_i; if the gated head (letters per F row + previous frames) binds at CFG 1, the CFG dependence was the
   head. `GatedCursorLM` (`vae/gated_nar.py`) is ready; GPUs are free (v5 ended 2026-10-28 11:58).
2. **Inference-only guidance fixes on cursor_v4b now** (no training): APG (2410.02416, drop the parallel component of the
   guidance → no oversaturation) and CFG-Zero* (2503.18886), CFG 3 / 5, 96 causal crops, CER + MERT cos + RMS / crest /
   clipping. Cheap answer to the CFG-5 loudness.
3. **Guidance-baked training** (2504.20334: target u − w·sg(v_c − v_u), CFG-free flow TTS at F5 quality, 9× faster) for the
   NAR expert: halves the 3B inference cost (no ×2 for CFG) and makes "binds at w = 1" a training target.
4. **Objective-side binding** if 1–3 leave a CFG-1 gap: the letter reader from the frozen VAE on the predicted clean block
   (x1_hat at t ≥ 0.5) — earlier fooled on the v1 latent (reader 0.49 vs Whisper 0.97), so only with the v4 / v5 latent and
   checked against Whisper; or REPA of F rows to a phonetic SSL model (ACE-Step 1 uses mHuBERT; unverified).
5. PER-reward GRPO / DPO last (expect 1–2 points).
Timing: keep the learned advance head. JAM reports bad timestamps triple PER, but our own oracle-timing test was *worse* than
sampled (0.545 vs 0.507 at CFG 3); no song generator with a learned cursor found (closest: SegTune 2606.02638 predicts LRC line
starts with Qwen3-4B; stochastic monotonic alignment for TTS 2502.01084) → claim survives.
Order after this check (v5 already passed its gate, 2026-10-28: gate20 WER 0.460, LTAS 2.87 dB; cursor_v4b through v5 CFG 3
0.361 / CFG 5 0.281): APG / CFG-Zero* on cursor_v4b + v5 decode ∥ E1 on v5 latents (MLP vs gated, matched steps) → gated vs fork
ablation → 3B port with guidance-baked loss → E2 / E3.

**VAE v5 gate (2026-10-28, `scripts/gate20_v5.sh`, `logs/gate20_v5_80k.log`; 80k = final):** gate20 Whisper WER clean 0.460
(v4 0.440, v2 0.478), +30 % noise 0.548 (v2 0.594); MERT cos 0.961 (v4 0.968); **LTAS 2.87 dB (v4 12.5, YuE2-Vae 2.25)**, stereo
0.81 (original 0.81). Held-out 80k: mel 3.13 dB, ctc_cer 0.45, anchor 0.087. cursor_v4b 20k (trained on v4) decoded through v5
with no retraining (`out/clm_eval/cursor_v4b_20k_v5dec_c96_cfg{3,5}`, same 96 causal crops, medians, paired vs v4 decode):
own CER CFG 3 0.361 (v4 0.401, −0.032 [−0.047, 0.000]), CFG 5 0.281 (v4 0.289, +0.006 [−0.014, +0.037]); swap follows the
given text (0.461 / 0.333). → v5 replaces v4: full band at the same lyric level, latent compatible. `vae/eval_cursor_lm.py`
now reads / writes audio at the VAE's rate.

2026-10-28: VAE v5 done (80k; held-out mel 3.13 dB @48k, ctc_cer 0.45). gate20 Whisper WER 0.460 (v4 0.440, YuE2-Vae 0.458),
+30 % noise 0.548 (v2 0.594), LTAS 2.9 dB (24 kHz v2/v4 12.3–12.5: the band limit is gone), stereo 0.81. cursor_v4b decoded
through v5 (96 crops, means): CFG 3 own 0.417 / swap-given 0.472, CFG 5 0.355 / 0.374 → the v4-trained LM carries over to v5.
E1 launched (`scripts/run_e1.sh GPU HEAD SEED`, cursor_v4b config, v4 latents so the 20k gate is paired with cursor_v4b):
`out/clm/e1_gated_s0` (GPU 0), `e1_gated_s1` (GPU 1), `e1_mlp_s1` (GPU 2, second MLP seed; cursor_v4b = first). Gated = 12
layers like the MLP run's backbone, 224M vs 208M (LoRA 64 + adaLN), K = 4 noisy copies of all 75 blocks per crop as NAR rows
(1275 positions, per-layer checkpointing, +0.9 s/step). Gate: CFG 3 / 5 own + swap on the 96 causal crops, after 20k.
E1 interim (teacher-forced held-out eval, fixed t grid + noise, step 7500): fl_own gated 1.0289 / 1.0292 (2 seeds) vs MLP
1.0704 / 1.0713 (v4b, s1) — gated at 7.5k = MLP at 20k (1.0292); gap_swap 0.0285 / 0.0287 vs 0.0245 / 0.0249, sgap_swap
0.044 / 0.045 vs 0.039 / 0.040, gap_null 0.016 vs 0.012. At 2.5k gated gap_swap is already 0.027–0.029 (MLP s1 0.014).
Seeds agree to 3e-4. Cost 2.34 vs 1.55 s/step. Not yet audio — the 96-crop CFG gate decides.

**E1 gate (2026-10-29, 96 causal crops, medians):** gated own CER CFG 3 0.244 / 0.237 (s0 / s1), CFG 5 0.136 / 0.162; MLP
(e1_mlp_s1) 0.395 / 0.268; VAE round trip 0.506. The gated NAR head beats the MLP head clearly.
Paired, 2-seed means per arm (MLP = cursor_v4b + e1_mlp_s1, gated = s0 + s1), 96 crops, bootstrap 95 % CI: own CER gated − MLP
CFG 3 −0.130 [−0.168, −0.091] (gated better on 77/96), CFG 5 −0.130 [−0.159, −0.101] (82/96); swap scored vs the given text
CFG 3 −0.155 [−0.187, −0.123] (80/96), CFG 5 −0.131 [−0.169, −0.095] (83/96) → it sings what it is told, not the crop's song.
Means: gated CFG 3 own 0.298 / 0.309, CFG 5 0.226 / 0.239; MLP CFG 3 0.436 / 0.430, CFG 5 0.358 / 0.367. Gated CFG 3 beats
MLP CFG 5 (0.30 vs 0.36), so less guidance is needed. Teacher-forced lyric gaps were equal at 20k (sgap_swap 0.044–0.045 for
both), so that eval does not predict the audio gain. Not checked yet: sound quality (MERT cos / LTAS / crest / clipping,
`/tmp/e1q/e1_quality.py` written, not run) and listening. Gated sampling re-runs the full sequence each flow step (~3× the
MLP gate time); needs the incremental state path before longer use.
**Local-attention arm (`--lattn`, 2026-10-29):** the shift is the only F→F path and it blends inputs (toy mode err 0.29 vs iso
0.17); letters reach F only as one flattened vector. `vae/gated_nar.py` LocalAttn: per layer, F rows only, the 4 frames of a
group attend bidirectionally to [24 cursor letters (GatedCursorLM.lmem), block i−1's clean frames (cin + cpos), own frames];
narrow (256 wide, 4 heads), zero-init out, added by index_add on F rows → A stays bit-exact (all unit tests pass, both with
shifts and iso). 237M vs 224M. Toy (`--arm lattn` = shifts + attn, `lattn_iso`): seed 0 loss 0.43 (bi 0.68), mode err 0.17
(= iso), agree 0.97 (bi 0.96) → frames agree without losing per-frame precision. Run `out/clm/e1_lattn_s0`
(`scripts/run_e1_lattn.sh 3 0`, GPU 3, 20k, then CFG 1 / 3 / 5 own + swap); baseline CFG 1 for e1_gated_s0 run alongside
(`out/clm_eval/e1_gated_s0_20k_c96_cfg1`). Decides: own CER at CFG 1 (binding without CFG) and CFG 3 vs e1_gated_s{0,1}.

**3B port, code (2026-10-29/30):** `vae/gated_rwkv7.py` LiuBaiH = official x070 block (g1k-3b loads strictly) + write gate +
fwd / bwd in-block shift + optional `--lattn` (8 heads × 64 = 512 wide); `vae/train_liubai_h.py` (FSDP2, `scripts/run_liubai_h.sh`)
on precomputed v5 latents (`tools/precompute_latents.py`); `vae/eval_cursor_lm.py` loads its exports (`arch liubai_h`). CPU
tests `vae/test_gated_rwkv7.py` pass (text rows = official demo math to 4e-15). Review 2026-10-30, tiny 2-GPU smoke test
(train → eval → save → resume → export → 96-crop gate script on 2 crops) found and fixed: (1) EMA swap crashed after the
no-grad eval forward (FSDP2 leaves the root unsharded without a backward → reshard first); (2) resume failed: layer 0's
`att.v0/v1/v2` are unused (v_first), get no grad, so no AdamW state → frozen; step-1 assert that every trainable param gets a
grad; (3) log averages wrong after resume; (4) instrumental share in the shards is 2.1 %, not ~10 % → `--p-instr 0.1` as in
train_cursor_lm.py (E1 recipe). Not run yet at 3B scale under FSDP (GPUs busy with precompute + e1_lattn_s0); estimate
~25–28 GB / GPU at bs 4 on 4 GPUs (fp32 shards + grads + AdamW + EMA ≈ 15.5 GB, activations ≈ 9 GB) → needs the GPUs free.
Known mismatch: training latents are whole-song encodes, the gate's causal prompt is a 3.8 s stand-alone encode (crop-edge
effect over its first ~3 s); same kind of prompt as the 208M gates, but those were also trained on crop encodes.
Speed (2026-10-30): lattn 2.57 s/step alone vs gated 2.34 (+10 %); the 3.7 s/step of steps 0–15k was the CFG 1 baseline eval
sharing GPU 3. Teacher-forced flow own at 15k 0.981 (gated 0.995 at 15k, 0.9875 at 20k). Baseline gated s0 CFG 1 own 0.776 /
swap-given 0.792 (CFG 3 0.244, CFG 5 0.136) → CFG 1 decides. Gate now `scripts/gate_e1_lattn.sh` (CFG 1 / 3 / 5 in parallel,
GPU 3 + any free GPU of 0–2), log `logs/gate_e1_lattn.log`.

**E1 lattn gate (2026-10-30, same 96 causal crops, medians, paired vs e1_gated_s0, bootstrap 95 % CI of the median diff):**
CFG 1 own 0.777 vs 0.776 (−0.005 [−0.038, +0.030]); CFG 3 0.194 vs 0.244 (+0.000 [−0.050, +0.011]); CFG 5 0.132 vs 0.136
(−0.004 [−0.039, 0.000]); swap-given CFG 5 0.158 vs 0.176. → NEGATIVE for binding: local attention on the cursor letters
does not bind lyrics without CFG; at most a small gain at CFG 5, within noise. Lower flow loss (0.975 vs 0.9875) did not
reach the words. Not worth carrying into the 3B port on this evidence (+10 % step time).
**Fast gated sampler (2026-10-30):** `GatedNARLM.cache_init / step_a / step_f`, `GatedCursorLM.step_hidden / step_velocity`:
per-layer WKV state + shift rows cached after each A token, each flow step = a blk-row pass against S (F never writes S, so
exact). CPU tests: = full forward to ≤ 6e-7 for default / iso / lattn / lattn+iso; GPU bf16 vs the CUDA-kernel forward on the
real checkpoints: rel L2 ≤ 5e-3. `eval_cursor_lm.py` uses it by default for gated LMs (all crops × conds in one batch,
`--gen-bs`, `--slow` = old path; noise draws differ from the old path). 96 crops × 2 conds at CFG 3: generation ~95 s vs
~7.7 h (289 s/crop) → ~300×. Same scores within noise: gated s0 CFG 3 own 0.215 (old 0.244), swap-given 0.259 (0.262).

**Init bug, fixed 2026-10-31:** `vae/rwkv.py` TimeMix and `vae/gated_nar.py` GTimeMix zero-initialised both `o` and `g2`.
That is a fixed point (o's grad is scaled by the gate g = g2(·), g2's by o), so the WKV time-mix never trained: `tm.o` / `tm.g2`
are exactly 0 in every layer of VAE v4 / v5, cursor_v4b, e1_gated_s{0,1}, e1_lattn_s0. All those models are conv / shift + FFN
nets without state (the gated "read-only NAR" never read S; the A stream sees ≤ n_layer blocks via the FFN shifts). Fix =
official RWKV-7 gate init (`rwkv.init_gate`: g1 = 0, g2 = orthogonal × 0.1); test in `vae/test_gated_nar.py` on the untouched
init. Old checkpoints load and behave as before; resuming one keeps the zeros (re-init g2 to revive). The 3B port is not
affected (pretrained time-mix). Conclusions to redo with live WKV: E1 gated vs MLP, lattn gate, toy shift / iso arms.
**Correction 2026-10-31 (init bug, found by another session; `vae/rwkv.py` init_gate):** o and g2 were both zero-init, so
both got exactly zero gradient forever. Checked on the checkpoints: max |tm.o| = max |g2| = 0.0 after 20k steps in
e1_gated_s0, e1_lattn_s0 and cursor_v4b. In all of them the WKV time-mix never contributed: the stacks were FFN + token shift
only (history = 1 block per layer through the shift, ~12 blocks ≈ 3.8 s), and F rows never read S. So E1 compared two
WKV-less models: "gated > MLP" holds only as deep F rows vs a separate MLP head, and the read-only gate itself (F→A through
S) is still untested. The lattn arm's local attention was the only working cross-token mixer besides the shift; its null
result has to be re-run on the fixed init before any conclusion. LiuBaiH (3B) is not affected: output / g2 come from the
pretrained weights. Fast sampler ported to `vae/gated_rwkv7.py` (LiuBaiH cache_init(txt) / step_hidden / step_velocity;
`vae/test_liubai_h_step.py`, fp64, = full forward to 1e-15 for default / iso / lattn / lattn + iso, with and without a text
prefix); `eval_cursor_lm.py` picks it up automatically.

**E1 live-WKV gate (2026-11-01, `out/clm_eval/e1_gated_s0_wkv_20k_c96_cfg{1,3,5}`, same 96 causal crops, paired vs dead-WKV
e1_gated_s0, medians, bootstrap 95 % CI of the median diff):** held-out flow own 0.9455 (dead 0.9875), sem dims 0.907 (0.946),
gap_swap 0.0255 (0.0296). Max |tm.o| 0.16–0.30 per layer (dead: 0), so the state is live.
| CFG | own live / dead | diff [CI] | swap-given live / dead | diff [CI] |
|---|---|---|---|---|
| 1 | 0.819 / 0.776 | +0.029 [−0.015, +0.126] | 0.785 / 0.792 | +0.021 [−0.062, +0.080] |
| 3 | **0.167** / 0.244 (fast sampler 0.215) | −0.059 [−0.087, −0.023] | 0.269 / 0.262 | −0.040 [−0.074, −0.017] |
| 5 | **0.110** / 0.136 | −0.031 [−0.067, 0.000] | 0.153 / 0.176 | −0.035 [−0.054, −0.008] |
→ live WKV improves guided binding (CFG 3 own −0.06, holds against the fast-sampler baseline too: −0.059 [−0.083, −0.014])
but **not binding without CFG**: CFG 1 is unchanged within noise, and its mean is worse (0.915 vs 0.790) because 14/96 crops
have CER > 1.2 (dead: 3/96), i.e. babble that Whisper over-transcribes. Better model of the song's own history, same
dependence on guidance. Pending: seed 1 (`e1_gated_s1_wkv`, GPU 2) and the lattn live-WKV arm (`e1_lattn_s0_wkv`, GPU 0, CFG 1
decides). CFG-1 binding remains the open problem → objective-side fixes (guidance-baked loss, PER-ranked DPO) move up.

**DPO trial (launched 2026-11-01, `scripts/run_dpo.sh 1 3 1200`, `vae/dpo_cursor_lm.py`):** Diffusion-DPO in flow form on
e1_gated_s0_wkv. Pairs: 1,200 training crops (pilot_clm_train_clean, not the gate crops), 4 s causal prompt + 20 s own
lyrics, 6 candidates each (CFG 1 ×4, CFG 3 ×2), Whisper letter CER; winner = lowest, loser = highest CER, gap ≥ 0.15.
Loss −logσ(−½β·Δflow + β_adv·Δlogp(advance)), β 2000, β_adv 0.1, same t / noise per pair, frozen reference, lr 1e-5,
bs 6 pairs, K 2, 1,500 steps. Arms: **a_onpol** (CFG-1 candidates only) and **b_mixed** (all, so CFG-3 samples can win:
distils guided binding into CFG 1). Gate: same 96 crops, CFG 1 / 3 own + swap, vs e1_gated_s0_wkv (CFG 1 0.819, CFG 3 0.167).
Smoke test (8 crops, CFG 1, 1, 3): mean CER per candidate 0.917 / 0.798 / 0.371, best-of-3 0.364.

**DPO trial result (2026-11-01, 96 gate crops, paired vs e1_gated_s0_wkv, medians, bootstrap 95 % CI):** pairs: 1,200 crops,
CFG-1 candidates mean CER ~1.10, CFG-3 ~0.40, best-of-6 0.30; a_onpol 1,048 pairs (winner 0.73 / loser 1.71), b_mixed 1,199
(0.30 / 1.62). Held-out preference acc a 0.86, b 1.00 (b separates CFG-3 vs CFG-1 samples trivially).
| arm | CFG 1 own | CFG 1 swap-given | CFG 3 own | CFG 3 swap-given |
|---|---|---|---|---|
| base (live WKV) | 0.819 | 0.785 | 0.167 | 0.269 |
| **a_onpol** | **0.576** (−0.236 [−0.292, −0.163], 78/96) | **0.686** (−0.125 [−0.161, −0.083]) | 0.152 (−0.014 [−0.034, 0.000]) | 0.142 (−0.043 [−0.074, −0.016]) |
| b_mixed | 0.665 (−0.223 [−0.296, −0.154]) | 0.750 (−0.138 [−0.204, −0.050]) | 0.141 (−0.014 [−0.039, +0.021]) | 0.161 (−0.056 [−0.084, −0.022]) |
| lattn_wkv (no DPO) | 0.787 (−0.037 [−0.087, +0.039]) | 0.790 | 0.140 (−0.013 [−0.032, +0.007]) | 0.181 |
Babble crops (CER > 1.2) at CFG 1: base 14, a 1, b 2, lattn 11. → **DPO is the first thing that moves CFG-1 binding**, and it
generalises to unseen crops and to swapped lyrics (it sings the given text, not a memorised song); CFG 3 does not get
worse. On-policy (CFG-1-only pairs) is at least as good as mixing in CFG-3 winners. lattn with live WKV: still no CFG-1 gain →
drop local attention. Not checked yet: sound quality (MERT cos / LTAS / crest), listening, reward hacking (Whisper-judged,
same ASR family as the gate: re-score with Qwen3-ASR before trusting the size of the gain), second seed, iterated DPO rounds.
