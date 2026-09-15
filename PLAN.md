# RWKV-7 Song LM — Plan

Goal: a **pure RWKV-7 (attention-free) lyrics+style → song model**, following the YuE
recipe (text → semantic audio tokens → acoustic rendering), trained on the 4×A100-40GB box.
Project root: `/root/x/rwkv-music`. Written 2026-09-12.

## 0. Decisions already made (and why)

| Decision | Choice | Reason |
|---|---|---|
| Backbone | RWKV-7, no attention layers | fits WKVM state-native engine; fixed-size state; user preference |
| Init | continue-pretrain **rwkv-g1k-3b-temp-5441.pth** (in-progress g1k data run, L32-D2560, world vocab 65536; from `BlinkDL/temp-latest-training-models`) — user's choice 2026-09-12. `rwkv7-g1j-2.9b-20260831` is downloaded as the released fallback | competitive from scratch is impossible on 4 GPUs. Caveat: the temp checkpoint is mid-training (step 5441) and will be superseded; record its filename in every run config |
| Audio tokens | **Current run:** X-Codec cb0 (YuE1 `m-a-p/xcodec_mini_infer`, Apache-2.0): 16 kHz, 50 Hz, 1024 entries; codebooks 0–7 kept on disk. **Target:** YuE2 semantic tokens (25 Hz, 32,768 entries, ids `151853..184620` in YuE2's vocab), reached by *distilling an inverse tokenizer* (§7 Track B), because YuE2 ships no audio→token encoder | X-Codec is the only open song tokenizer with a released encoder. YuE2 / HeartMuLa / MiniMax ship generators only: their AR models emit semantic tokens from text, no audio→token encoder, so their token spaces cannot be used as a training target without §7. YuE2's NAR only renders YuE2 tokens; X-Codec and YuE2 token spaces are unrelated (vocab, rate, meaning), no adapter exists. YuE2-Vae does include an encoder, but it yields continuous 64-dim latents, not tokens |
| Renderer (stage-2) | **target = YuE2 quality (48 kHz stereo)**: YuE2 NAR + YuE2-Vae via Track B in §7, or own NAR → YuE2-Vae via Track A. YuE1 `YuE-s2-1B-general` + X-Codec/Vocos stays as the 16 kHz **baseline** for the current cb0 run | user decision 2026-09-14: YuE2-level output is the goal, experiments first |
| Corpus | `webshart/suno-various-94k` original config (94,174 songs, style caption + structured lyrics) as primary; `humair025/suno-audio` (49.7k, MIT) and MTG-Jamendo (55k real tracks, tags) as secondary | only corpora with aligned lyrics+tags+full songs reachable from this host |
| License | **deferred** (user 2026-09-14: experiments first, do not let licensing block them). Record: YuE2-3B / YuE2-Vae / YuE2-Vae-legacy / MERT-v2 are CC BY-NC 4.0; X-Codec, YuE1 s2, RWKV-7 are Apache-2.0 | **the RWKV song model is intended to be open source**, so re-check this row before any public release: a release that depends on YuE2 weights needs a permissively licensed replacement renderer first (§5) |
| Storage | stream shards: download → tokenize → delete audio; keep tokens + metadata only | 454 GB of MP3 vs 497 GB free disk; tokens are ~140 KB/song |

Measured on this box (2026-09-12): X-Codec whole-song encode ≈ 200× realtime on one A100,
peak 7.8 GiB for a 167 s song; hf-mirror download with `aria2c -x8` ≈ 100 MB/s.

**Download path caveat (2026-09-12 evening):** hf-mirror only redirects; the tar bytes come from
HF's Xet store via CloudFront (`cas-bridge.xethub.hf.co`). The system resolver returned the
`108.156.221.x` edge (332 ms RTT, 25 % loss, 0.2–0.7 MB/s per stream) and downloads fell to
20–60 min per shard. Edge `18.155.68.x` (what Cloudflare 1.1.1.1 resolves to) does 12 MB/s per
stream, 70+ MB/s with 8. Fix in place: `/etc/hosts` pins `18.155.68.69 cas-bridge.xethub.hf.co`
(backup `/etc/hosts.bak-20260912`; remove the line when the dataset phase is over). If downloads
slow down again, re-measure with `dig @1.1.1.1 cas-bridge.xethub.hf.co` +
`curl --resolve cas-bridge.xethub.hf.co:443:<ip> <signed-url>` and repin. HF direct and
ModelScope are not options (blocked / dataset absent).
Chunked encoding (60 s windows) disagrees with whole-song encoding on ~17 % of codebook-0
frames, so **songs are encoded in one pass** (chunked only above 8 min).

## 1. Environment (done / to do)

Done: `soundfile`, `librosa`, `omegaconf`, `tensorboard`, `matplotlib`, `julius`, `argbind`
installed from the Tsinghua PyPI mirror; `scripts/shims/audiotools` stubs the heavy
`descript-audiotools` package so X-Codec imports without torchaudio.

To do before training:
- `flash-linear-attention 0.5.2` does not import: triton 3.8 is paired with torch 2.6.
  Either pin triton to 3.2 for torch 2.6, or move torch forward. RWKV-LM's own CUDA kernel
  does **not** need fla, so training is unblocked; fla is needed for the wkvm/eval side.
- [done] `models/rwkv-g1k-3b-temp/rwkv-g1k-3b-temp-5441.pth` and `models/rwkv7-g1j-2.9b/rwkv7-g1j-2.9b-20260831-ctx16384.pth` downloaded via hf-mirror (HF direct is blocked). Verify sizes: both must be 5.9 GB.
- [done] `tools/RWKV-LM` cloned (RWKV-v7/train_temp has train.py + src/binidx.py; RWKV-v5/make_data.py builds binidx).
- Optional: `ffmpeg` static build (GitHub release asset download stalled once; conda-forge
  mirror at tuna works as fallback). Not needed for mp3 decode — libsndfile 1.2.2 handles mp3.

## 2. Dataset phase (this is the phase being executed now)

### 2.1 Sources

| Source | Songs | Audio | Text fields | License | Status |
|---|---|---|---|---|---|
| webshart/suno-various-94k `original` | 94,174 | MP3, 85 tars × ~5.3 GB | `caption` (style tags), `lyrics` with `[verse]/[chorus]` markers, title, duration | creator rights retained; research use | **running**: shard 0 done, shards 1–84 tokenizing on GPUs 0/1 via `scripts/prepare_suno94k.py` (`scripts/run_all_shards.sh`) |
| humair025/suno-audio | 49,698 | MP3 in Arrow, 213 GB | `tags`, `prompt` (= lyrics), title | MIT (as declared) | phase 2.4 |
| rkstgr/mtg-jamendo | 55k full tracks | tars, 117 GB | genres / instruments / moods, **no lyrics** | CC (Jamendo), Apache packaging | phase 2.5, lyrics via HeartTranscriptor for vocal tracks |
| m4singer / opencpop / GTSinger | small | WAV | Chinese lyrics, aligned | MIT / CC | eval only |

### 2.2 Token format on disk

- `data/tokens/<source>/shard-NNNNN.npz`: `clip_id -> np.int16 [8, T]`, 50 Hz, values 0..1023.
- `data/meta/<source>/shard-NNNNN.jsonl`: `id, title, caption, lyrics, duration, n_frames, creator, model_name, …, error?`
- Codebook 0 alone is the stage-1 target. Keeping all 8 lets us train / verify a stage-2 renderer later.

### 2.3 Sequence format for stage-1 (defined here, built in phase 3)

RWKV world vocab (65,536) + new ids appended:

```
65536      <EOD>
65537      <SOA>   start of audio
65538      <EOA>   end of audio
65539      <xcodec> codec-type marker (YuE convention)
65540–65543 reserved (stage markers, ICL prompt, …)
65544–66567 codebook-0 codes 0..1023   (= 65544 + code)
```
Vocab padded to **67,072** (65,544 + 1,024 = 66,568 rounded up to a multiple of 512; the earlier
figure 66,560 was too small by 8 and tripped an assert in `build_binidx.py` on 2026-09-13).
Embedding rows for new ids initialised to the mean of existing rows plus small noise; head rows likewise.
Constants live in `scripts/build_stage1_manifest.py` and are imported by `build_binidx.py`.

One training document per song (YuE "segment" style, which is what makes 3-min songs fit
a 16k context):

```
[Genre] {caption}\n[Lyrics]\n{full lyrics}\n<EOD>
<SOS>[verse]\n{verse lyrics}<SOA><xcodec>{cb0 codes for that section}<EOA>
<SOS>[chorus]\n...
```
Because Suno lyrics carry section markers but **no timestamps**, sections cannot be aligned to
audio yet. Phase 3 therefore starts with the simpler whole-song format
`prompt text <EOD> <SOA><xcodec> all cb0 codes <EOA>` (≈ 9k tokens for 3 min), and adds
section alignment later using HeartTranscriptor word timestamps.

### 2.4 Steps

1. **[done]** Shard 0 of suno-94k → tokens: 1,135 clips, 0 failures, 12.8 min, 177 MiB (see §6).
2. **[running]** `bash scripts/run_all_shards.sh 0 1` → shards 1–42 on GPU 0, 43–84 on GPU 1.
   ~13 min/shard/GPU → ≈ 9–10 h wall-clock. Token output ≈ 15 GB total. Resumable: rerun the
   same command; finished shards are skipped, shards with >10 % failures are redone.
   Check: `grep -h '^\[shard' logs/prep_suno94k_gpu*.log; ls data/tokens/suno94k | wc -l` (expect 85).
   **[running]** `scripts/prefetch_shards.py --ahead 2` (log `logs/prefetch_suno94k.log`) downloads the
   next 2 tars per worker range to `.tar.part` and renames them in place, so the GPU never idles on the
   network. It never touches the worker's current shard; the worker's `download()` sees a complete tar
   and skips its own download. Disk cost ≤ 4 extra tars ≈ 21 GB. Relaunch it together with
   `run_all_shards.sh` if either dies.
3. **[done 2026-09-13]** `python scripts/dataset_report.py --source suno94k --check-npz` (output in
   `logs/dataset_report_suno94k.txt`). Filter list is implemented in `scripts/build_stage1_manifest.py`:
   30 s ≤ duration ≤ 300 s, whole document ≤ 16,384 tokens, instrumental-only lyrics (< 20 chars after
   stripping `[...]` markers) kept as caption-only docs with lyrics `[Instrumental]`, exact-duplicate
   lyrics kept in train (different audio renders) but excluded from val.
4. humair025/suno-audio: Arrow batches → same pipeline (`prepare_suno_audio.py`, to write;
   reuse `XCodecTokenizer`, read `audio.bytes` from Arrow, `prompt` → lyrics, `tags` → caption).
5. MTG-Jamendo: download `rkstgr/mtg-jamendo` tars, tokenize, run HeartTranscriptor on the
   vocal subset (`instruments` contains `voice`/`vocal`) to produce lyrics; keep instrumental
   tracks as caption-only documents.
6. **[done 2026-09-13]** Held-out split: 1 % of creators by md5 hash (`build_stage1_manifest.py`) →
   `data/manifests/suno94k_stage1_{train,val}.jsonl` + `_summary.json`. The m4singer/opencpop Chinese
   OOD probe is still to add.
7. **[done 2026-09-13]** `scripts/build_binidx.py` → `data/binidx/suno94k_stage1_{train,val}.{bin,idx}`,
   int32 (vocab 67,072 > uint16), document = `text EOD SOA XCODEC cb0+65544… EOA 0` (trailing 0 is the
   RWKV-LM doc separator). Sidecar `suno94k_stage1_binidx.json` holds token counts and `magic_prime`
   for ctx 16,384. Note RWKV-LM's `MyDataset` samples random ctx-length windows from the flat stream
   (documents are cut arbitrarily, as in pretraining); a document-aligned loader can be added later.

## 3. Stage-1 training (started 2026-09-13)

- Trainer: RWKV-LM v7 (`tools/RWKV-LM/RWKV-v7/train_temp/train.py`), launched by `scripts/train_stage1.sh`
  (`out/stage1/`, logs `logs/train_stage1_*.log`, `out/stage1/train_log.txt`). Init `out/stage1/rwkv-init.pth`
  = g1k-temp-5441 with emb/head grown to 67,072 rows by `scripts/resize_vocab.py` (new rows = mean + 2 % noise).
  ctx_len 16384, bf16, DeepSpeed stage 2, grad_cp 1, head_chunk 65536, micro_bsz 1 × 4 GPUs.
- Environment fixes needed (all done): `pytorch-lightning==1.9.5`, `deepspeed==0.16.9` (0.19 breaks with
  torch 2.6: `BaseMuonWithAuxAdam` NameError), `setuptools<81` (lightning_fabric imports `pkg_resources`),
  pydantic-core pinned to pydantic's requirement. Installed with `pip --no-deps` so torch stays untouched.
- **Local patches to RWKV-LM** (originals kept as `*.orig`): (1) `src/model.py` reads
  `RWKV_D_{DECAY,AAA,MV,GATE}_LORA` env overrides because the g1k-3b checkpoint uses LoRA dims 96/96/64/320
  while the formula in train_temp gives 128/128/96/256 for D=2560; (2) three CUDA kernels
  (`rwkv7_cmix_bf16_v5.cu`, `rwkv7_tmix_mix6_bf16_v5.cu`, `rwkv7_tmix_kk_pre_bf16_v5.cu`) used `atomicAdd(float2*)`,
  which exists only on sm_90; guarded with `__CUDA_ARCH__ >= 900` and two scalar atomics for the A100;
  (3) `rwkv7_head_l2wrap_ce_bf16_v4.{cpp,cu}` hard-coded vocab 65,536 (`constexpr`, used only as row stride /
  strided loop bound) — now `-DHEAD_CE_VOCAB`, set from env `RWKV_HEAD_CE_VOCAB=67072` in `model.py`
  (extension name carries the vocab so the build cache cannot mix them).
  `scripts/check_arch.py` verifies the layout (1,062 tensors, 0 mismatches) and runs one forward:
  init loss 15.6 (text 5.7, audio 16.7 — new head rows start near-uniform), 15.7 GiB peak at ctx 16k, no grad.
- RWKV-LM trainer facts: `epoch_steps × real_bsz` must be 40,320 (auto-derived), one "epoch" = 660 M tokens;
  `MyDataset` samples random ctx windows from the flat stream (magic_prime 51,563); `--my_exit_tokens`
  sets the cosine-LR horizon and writes `rwkv-final.pth` when reached; `train_stage 3` resumes from the
  newest `out/stage1/rwkv-*.pth`.
- LR 1e-4 → 1e-5 cosine, 500 warmup steps, batch ≈ 0.5 M tokens, weight decay 0.1.
- Two-phase schedule as in YuE: (a) 1 epoch mixed text-lyrics documents + audio documents;
  (b) anneal on the highest-quality subset (upvote_count ≥ 5, duration 60–300 s).
- Token budget: 94k + 50k songs × ~8k cb0 tokens ≈ 1.2 B audio tokens per epoch. 2–3 epochs.
- Checkpoint every 1k steps; eval loss on held-out cb0 + a 20-song generation sweep.

## 4. Rendering and evaluation (next chat)

**Superseded in part by §7 (2026-09-14):** the YuE1 stage-2 path below is the baseline for the cb0 run only; the target renderer is YuE2.

- Stage-2: `m-a-p/YuE-s2-1B-general` upsamples cb0 → 8 codebooks; X-Codec decode + Vocos
  vocal/instrumental decoders (`models/xcodec_mini_infer/decoders`). The YuE1 `infer.py`
  post-processing is at commit `c7ec7532` of the YuE repo (saved copies in `/tmp/yue1/`,
  copy into `docs/yue1_reference/`).
- Metrics: HeartTranscriptor WER on lyrics (lyric following), CLAP score vs caption,
  SongEval / SongBench-style MOS proxies, plus the same sweep on HeartMuLa-oss-3B as the
  open-model reference.
- Long-context probe: RWKV state is fixed; check cb0 loss vs position over 16k to see whether
  song structure (chorus repetition) is remembered without attention.

## 5. Later

- Permissively licensed replacement for YuE2-Vae (candidates to license-check: DAC, SNAC, X-Codec-2; Stable Audio Open's VAE is non-commercial) once §7 works and a public release is wanted. YuE2 inference-code license is unverified (LICENSE / THIRD_PARTY_NOTICES.md not saved locally).
- Section alignment via HeartTranscriptor timestamps → YuE segment format → better lyric control.
- Own stage-2: bidirectional or small-transformer NAR over X-Codec 8 codebooks, or RWKV
  depth-decoder à la HeartMuLa.
- Serve stage-1 inside WKVM (fixed-state streaming generation).

## 7. YuE2-level generation (started 2026-09-14)

**Goal:** RWKV-7 stage-1 whose output renders through YuE2's acoustic stack (flow-matching NAR → YuE2-Vae, 48 kHz stereo), i.e. YuE2 audio quality with an attention-free, fixed-state stage-1. Licensing deferred (§0).

**The obstacle:** YuE2's stage-1 emits 32,768-entry semantic tokens at 25 Hz (one per VAE latent frame, 48000/1920). The audio→token encoder that produced them for training was never released; the shipped pipeline is one-way (text → tokens → latents → audio). So the corpus cannot be tokenized into YuE2's space directly.

**What *is* released and usable (from `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`, `m-a-p/YuE2-Vae-legacy`):**
- AR backbone: text (+ optional ABC score) → semantic tokens. Lets us *generate unlimited paired (token, latent, audio) data*.
- NAR (same backbone, `nar_*` weights): tokens → 64-dim latents, conditioned through the AR KV cache of `prefix + tokens` (`docs/yue2_reference/yue2_infer_pkg/nar.py`). Needs a YuE2-format text prefix (`protocol.token_prefixes`).
- YuE2-Vae: `encode()` audio → latents **is implemented and loads with `decoder_only=False`** (`modeling_vae.py:491`); `decode()` latents → 48 kHz stereo. 1920× downsampling → 25 Hz latents, exactly token-aligned.
- Local: reference code only (`docs/yue2_reference/`, `/tmp/yue2_infer`). **Weights not downloaded yet.** Source: hf-mirror with the cas-bridge pin (§0), or ModelScope (`/tmp/yue2_config_ms.json` came from there, so a mirror exists; verify the safetensors are complete: 3B AR/NAR bf16 ≈ 7 GB + two VAEs).

### 7.1 Track B (primary): distill an inverse tokenizer, train RWKV in YuE2 token space

B1. **Bring up YuE2.** Download weights; run `yue2_infer` on one A100 (README: 24 GB, FP32 VAE). Save one song's `semantic.npy`, latents, audio. Confirm `synthesize()` accepts externally supplied token lists (it should: `SemanticResult` is a plain dataclass and `pipeline.py:285` only checks the prefix). Measure `YuE2VAE.encode(decode(z))` vs `z` per-frame MSE: this is the domain gap the inverse tokenizer must absorb.
B2. **Paired corpus.** Prompt YuE2 with suno-94k captions + lyrics (94k prompts, same distribution as the training corpus), `cot="off"` unless quality needs ABC. Per song store: tokens (int16, ~4.5k per 3 min), NAR latents (fp16 [T,64]), and **re-encoded latents** `vae.encode(vae.decode(latents))` (fp16). Audio not kept. Cost: 3.6-min song in 71 s on a 4090 → assume ~50 songs/GPU-h on A100. Target 10k songs ≈ 200 GPU-h ≈ 2 days on 4 GPUs; start with 2k for a first classifier. Disk ≈ 2.3 MB/song → 23 GB for 10k.
B3. **Inverse tokenizer.** *(Amended 2026-09-14, see §8.2: use MERT-v2-FullSong features as input, a third-party head + paired corpus already exist.)* Original text: Input re-encoded latents [T,64] @ 25 Hz → token id (32,768-way CE). Bidirectional model with local context (chunk transformer or RWKV, 50–100 M params; ±2 s receptive field suffices since tokens are frame-aligned). Train on B2 pairs, hold out 5 %. Metrics, most important first: (1) **round-trip on real suno audio**: audio → encode → predicted tokens → YuE2 NAR → decode; judge by CLAP vs caption, HeartTranscriptor WER vs lyrics, listening; (2) token top-1 / top-5 on held-out generated songs; (3) latent MSE of NAR(predicted tokens) vs NAR(true tokens). **Gate:** round-trip on real audio clearly intelligible and on-style → B4; otherwise Track A.
B4. **Tokenize the corpus into YuE2 space.** Raw audio was deleted after X-Codec tokenization, so re-stream suno-94k (+humair025): download → 48 kHz stereo → `YuE2VAE.encode` → store latents fp16 (~1.1 MB/song, ~100 GB for both corpora; keep them, Track A needs them too) → inverse tokenizer → tokens. Reuse the §2.4 shard pipeline and the cas-bridge pin; ~85 shards × ~13 min if encode keeps up.
B5. **Stage-1 RWKV on YuE2 tokens.** Vocab: world 65536 + 32,768 codec + specials (~98.6k); extend embedding/head with `scripts/resize_vocab.py`. Sequence format as §2.3 with a `<yue2codec>` marker; ~4.5k audio tokens per song, so ctx 8192 holds a whole song. Init from the **current cb0 run's final checkpoint** (lyrics/style → structure transfers; codec rows re-initialised). Two-phase schedule as §3.
B6. **Render.** RWKV tokens → YuE2 prefix from the same lyrics/style (`protocol.token_prefixes`, cot off) → `synthesize()` → `YuE2VAE.decode` → 48 kHz stereo. Eval as §4, with YuE2 itself and HeartMuLa as references.

Risks: (a) inverse-tokenizer accuracy on real audio (generated→real gap; mitigated by re-encoding through the VAE and by suno audio being synthetic too); (b) NAR sensitivity to token errors → B3 metric (3), consider sampling from the classifier; (c) the NAR is conditioned on the text prefix as well as tokens, so RWKV outputs must be paired with a well-formed YuE2 prefix; (d) GPU contention with the running stage-1 (§7.3).

### 7.2 Track A (fallback / parallel): own NAR from X-Codec cb0 → YuE2-Vae latents

Keep the current cb0 stage-1. Train a flow-matching NAR (design from `nar.py`: bidirectional transformer, time embedding, midpoint solver, ~300 M params) conditioned on X-Codec tokens (cb0 only, or cb0–7 via YuE1 s2 for an easier target) plus text, predicting YuE2-Vae latents; targets are the B4 latents. Decode with YuE2-Vae. No inverse tokenizer and no stage-1 retraining, but a from-scratch renderer on ~140k songs; expect lower quality than YuE2's NAR. Rate mismatch 50 Hz tokens vs 25 Hz latents: pool token pairs or upsample latents ×2 in the conditioning.

### 7.3 Schedule and GPU plan

- Stage-1 cb0 run 2 keeps its 4 GPUs until `rwkv-final` (≈ 21:30 on 2026-09-15; GPUs at 25–27 GB each, YuE2 needs ~24 GB, so they cannot share). It is the Track A stage-1, the Track B init, and the pure-RWKV baseline. Stopping it early is the user's call.
- Now (CPU/network only): download YuE2 weights; write `tools/yue2_gen_pairs.py` (B2), `tools/yue2_vae_encode.py` (B4), `scripts/train_inverse_tokenizer.py` (B3); CPU dry-run of YuE2 for shapes.
- After stage-1 finishes: B1 (1 GPU, ~1 h) → B2 first 2k songs (2 GPUs) alongside the §4 baseline eval on the other 2 → B3 → gate.
- B4 re-streaming starts as soon as one GPU is free (VAE encode only), independent of the gate; its latents feed both tracks.

## 8. Generation 2: beyond YuE2 and Suno v6 (designed 2026-09-14)

**User goal (2026-09-14):** build on YuE2 with the RWKV backbone, and end up *more advanced* than YuE2 and Suno v6.

### 8.1 Where the bar is (measured facts, not claims)

WildSongBench, 192 prompts, from YuE2's own README (their benchmark, their metrics; SongBench Avg 0–10, PER = phoneme error rate):

| System | Musicality | SongBench Avg | MuLan | Q3O | PER |
|---|---:|---:|---:|---:|---:|
| Suno v5 | 5.99 | 6.87 | 0.543 | 4.59 | 8.1 % |
| Suno v5.5 | 5.81 | 6.72 | 0.509 | 4.59 | 6.0 % |
| Suno v6 / v6-wild (2026-09-09 release, per YuE2 project page) | – | 6.56 / 6.42 | – | – | – |
| Mureka 9 | 6.05 | 6.94 | 0.439 | 4.64 | 11.7 % |
| YuE2, 1 of 2 candidates | 5.91 | 6.73 | 0.507 | 4.68 | 8.4 % |
| YuE2 best-of-8 | 6.27 | 6.96 | 0.505 | 4.70 | 9.8 % |
| HeartMuLa / ACE-Step 1.5 / MiniMax Music 3 (best open baselines) | 5.50 / 5.16 / 5.35 | 6.25 / 6.01 / 6.28 | | | 10.7 / 7.5 / 6.3 % |

Reading: (1) YuE2's single-candidate quality is already at Suno v5 level and above Suno v6 on this benchmark; the remaining gaps are **lyric adherence** (Suno 6–8 % PER vs YuE2 8–10 %) and **style/caption match** (MuLan 0.54 vs 0.51). (2) Suno v6's advances are product features, not benchmark quality: 8-minute songs, plain-language section edits, mashups from several sources, image-to-song, voices/custom models, and licensed training data (Warner/BMG/Believe). (3) YuE2 is one-way: no audio input except through SheetSage2 score transcription (covers only), no continuation, no extension, no section edit of an existing recording. Its context is 24,576 positions → ≈ 8 min of audio, same cap as Suno v6.

So "more advanced" is achievable on three axes, and one axis is *not* realistic on this box:
- **Not realistic:** out-training Suno on raw acoustic realism from scratch. Our audio corpus is ~5.4k h of Suno renders (+ 2.9k h humair025, + 1.6k h Jamendo); Suno/YuE2 trained on orders of magnitude more. We therefore *inherit* YuE2's acoustic stack (semantic tokens → flow-matching NAR → VAE, 48 kHz stereo) and improve it where it is cheap to improve (§8.3 R2).
- **Axis 1, quality where YuE2 is weakest:** lyric adherence and caption match, via (a) real Suno audio as training distribution (Suno's structure, hooks and mix, in YuE2's token space), (b) a lyric-cursor auxiliary objective, (c) preference optimisation that bakes YuE2's best-of-8 selection into a single sample. Target: PER ≤ 6 %, MuLan ≥ 0.54, SongBench ≥ 6.9 single-sample, with the *same renderer* as YuE2 so the comparison isolates stage-1.
- **Axis 2, capabilities neither has, that a fixed-state RNN makes natural:** unbounded song length with constant memory (no 8-min cap), true streaming generation, **audio input** (continue / extend / cover / remix an existing recording through the inverse tokenizer, §8.2), section edits by snapshotting the RWKV state at section boundaries and regenerating one section in place (WKVM state save/restore), live parameter changes mid-song.
- **Axis 3, Suno v6-style product features:** plain-language section edit, mashup (two audio inputs → tokens → one state), 8+ minute songs, symbolic (ABC) planning inherited from YuE2 for editable melody/chords.

### 8.2 What changed since §7 was written: the missing encoder exists

`Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` (HF, CC BY-NC 4.0, updated 2026-09-13, scripts saved to `docs/yue2_reference/mothersuperior_v4/`):
- **`tokenizer_head_joint_v4.pt`** — MERT-v2-FullSong **layer-20** features (25 Hz, 1024-d, per-track instance-normalised) → 32,768 YuE2 codes. 8-layer transformer, d=512, 512-frame (20 s) windows. Held-out exact match on YuE2's own songs 16.1 % top-1; they report NAR round-trips sound ≈ 95 % right by ear ("near-miss codes render almost identically").
- **`nar_lora_joint_v4.pt`** — rank-32 LoRA on the NAR branch (`nar_self_attn.{q,k,v,o}`, `nar_mlp.*`, 28 layers) + full `vae2llm`/`llm2vae`, trained jointly with the head on *real* audio with the flow-matching loss as teacher (`joint.py`). This is exactly the "NAR sensitivity to token errors / generated→real gap" risk in §7.1, already addressed.
- **`Mothersuperior/yue2-minted-corpus`** (HF dataset, 166 GB): 4,720 YuE2-generated songs with `semantic.npy` (true tokens), `latent.npy` [T,64], `audio.flac` 48 kHz, `score.abc`, prompt tokens, style + lyrics. Tokens + latents alone are 5.7 GB. This is §7.1 B2 done for us (4.7k songs; we planned 2k–10k).
- Also confirmed: YuE2's real tokenizer is a **causal branch of MERT-v2** (project page: "a separate causal branch becomes YuE2's 25-Hz semantic tokenizer"); the public MERT-v2 checkpoints are the bidirectional siblings. So MERT-v2-FullSong features are the right input for an inverse tokenizer — much closer to the true tokenizer than the VAE re-encode input proposed in §7.1 B3. **§7.1 B3 is amended: input = MERT-v2-FullSong hidden states (layer 20, optionally several layers), not VAE latents.**

Downloaded 2026-09-14 to `models/yue2/`: `YuE2-Vae`, `YuE2-Vae-legacy` (0.53 GB each), `MERT-v2-FullSong` (2.5 GB); `YuE2-3B` (7.26 GB) in progress. hf-mirror works only with `HF_HUB_DISABLE_XET=1` (the Xet CAS server returns 401 through the mirror). Still to fetch: the two Mothersuperior `.pt` files and the minted corpus tokens/latents (skip the flac except a 200-song validation subset).

### 8.3 Architecture (generation 2)

```
                text prompt (style, lyrics, optional ABC score)            audio input (optional)
                                    │                                           │
                                    │                       MERT-v2-FullSong L20 → inverse tokenizer (§8.2)
                                    ▼                                           ▼
     ┌──────────────── RWKV-7 3B stage-1, fixed state ─────────────────┐   YuE2 semantic tokens (25 Hz)
     │  vocab = world 65,536 + 32,768 codec + specials → 98,816       │◄──── prefix / continuation / edit context
     │  aux head: lyric cursor (which lyric token is being sung)       │
     │  state snapshots at section boundaries (WKVM)                   │
     └───────────────────────────────┬─────────────────────────────────┘
                                     ▼  semantic tokens, unbounded length, streamed in 20–60 s chunks
      YuE2 NAR (flow matching, 32 → 8–16 midpoint steps) + our LoRA fine-tuned on real Suno latents
                                     ▼  [T,64] latents @ 25 Hz
      YuE2-Vae decoder (48 kHz stereo)  — later: permissive replacement (§5)
```

Components and what each buys over YuE2 / Suno v6:

- **R1 Inverse tokenizer (ours, better than v4).** Input: MERT-v2-FullSong layers {12, 16, 20, 24} (learned layer weights), 25 Hz. Model: bidirectional RWKV-7 or chunk transformer, ~100 M params, 40 s windows, soft-CE over minted labels + **NAR flow-loss teacher** on real audio (straight-through tokens, as `joint.py`). Train data: minted corpus (4.7k) + our own YuE2 minting from suno-94k captions (5k, `cot="off"`, 2 GPUs × 1 day) + real Suno audio for the teacher term. Gate: top-1 > 16 % on minted val, and round-trip CLAP/PER on 50 real Suno songs within 10 % of the originals. This gives us *audio input*, which YuE2 lacks.
- **R2 Renderer adaptation (the acoustic-quality lever).** LoRA (rank 32–64) on YuE2's NAR branch + `vae2llm`/`llm2vae`, trained on **real Suno latents** (`YuE2VAE.encode` of suno-94k/humair025) with tokens from R1. Suno renders are mastered, stereo-wide commercial-grade mixes; YuE2's NAR learned from whatever it was trained on and scores below Suno v5 on MuLan/AllMusicCaps. Measure with SongEval on identical token sequences: base NAR vs LoRA NAR. Also distil the 32-step midpoint solver to 8 steps (consistency / rectified-flow distillation) for streaming latency.
- **R3 Stage-1 RWKV-7 3B in YuE2 token space (replaces the cb0 model as the product).** Data: 145k real songs (suno-94k + humair025) tokenized by R1 + 4.7k minted + our minted 5k + Jamendo instrumentals; ≈ 4.5k audio tokens per 3-min song → ctx 8,192 holds text + song; ~700 M audio tokens/epoch. Init from cb0 `rwkv-final` (structure and lyric conditioning transfer; codec rows re-initialised). Format extends §2.3: `[Genre]…[Lyrics]… <EOD> <SOA><yue2codec> tokens <EOA>`, plus, once timestamps exist, YuE-style interleaved sections. **Lyric cursor:** HeartTranscriptor word timestamps on the vocal stem → per-frame index of the lyric token being sung → small auxiliary head (weight 0.05–0.1); Mothersuperior found this necessary to stop lyric drift when fine-tuning YuE2's AR, and PER is the metric where Suno leads.
- **R4 Preference optimisation (bake in best-of-8).** GRPO/DPO on stage-1 with rewards computed on 30–45 s rendered excerpts (R2 renderer at 8 steps): SongEval musicality, caption similarity (MuLan/CLAP-style, MERT-v2 embedding vs caption-embedding probe), PER via HeartTranscriptor, plus a KL to the SFT model. YuE2's best-of-8 lifts SongBench 6.73 → 6.96; a policy that gets that on sample 1 is our clearest single-number win. The user already has a GRPO pipeline (RNN-StateTuning) to adapt.
- **R5 Audio-conditioned operations (Suno v6 features, YuE2 lacks them).** Continue/extend: tokens of the input → state → generate. Cover/remix: input tokens as ICL prompt + new style tags. **Section edit:** train with an `<edit>` format (prefix tokens, `<section:chorus 2>` spec, new instruction, suffix tokens visible via a second pass) so one section is regenerated with both sides fixed; at inference restore the state snapshot from the section start (WKVM) instead of re-prefilling. Mashup: two inputs → interleaved ICL prompt.
- **R6 Length and streaming.** No positional limit; NAR already runs in chunks. Target demo: a 12-minute song and a live stream at < 1× realtime on one A100 (RWKV 3B ≈ 40 tok/s needed; 25 tok/s is realtime).
- **R7 Symbolic planning (optional, keep YuE2's editability).** SheetSage2 (released, on MERT-v2-FullSong) transcribes suno songs → ABC; add `[Score] abc` between tags and lyrics for 20–30 % of documents so the model can be steered by melody/chords at inference, like YuE2 `cot="melody"`.

### 8.4 Phases, GPU budget, gates

| Phase | Work | GPUs | Gate |
|---|---|---|---|
| G0 (now, CPU + 1 GPU when free) | Bring up YuE2 in its own venv (torch 2.10 pinned; keep the RWKV env untouched). Fetch Mothersuperior head + LoRA + minted tokens/latents. Round-trip 20 suno val songs: audio → MERT → head v4 → NAR(+LoRA) → VAE. Listen; CLAP + PER vs originals. | 1 × A100, ~4 h | Intelligible, on-style round-trips → continue. Else fall back to §7.2 Track A |
| G1 | Corpus in YuE2 space: re-stream suno-94k + humair025 (§2.4 pipeline, cas-bridge pin) → per song: VAE latents fp16 [T,64] (~0.6 MB), MERT L{12,16,20,24} features for a 6k-song subset only (10 MB/song), tokens from head v4 now, re-tokenized by R1 later from stored latents? — no: tokens come from MERT features, so **store MERT L20 fp16 for all songs** (~5 MB/song → ~700 GB) *or* tokenize inline and re-stream again for R1. Decision: tokenize inline with the best head available and keep MERT features only for the 6k subset; if R1 beats v4 clearly, re-stream once more (20 h network). | 2 × A100, ~1 day (network-bound, ~13 min/shard) | 145k songs tokenized, latents stored (~95 GB) |
| G2 | R1 inverse tokenizer + R2 NAR LoRA (joint training as `joint.py`), our minting of 5k songs from suno captions for extra labels. | 2 × A100, 2 days | R1 top-1 > 16 %, R2 SongEval ≥ base NAR on 100 held-out token sequences |
| G3 | R3 stage-1 v2: vocab 98,816, ctx 8,192, init from cb0 `rwkv-final`, lyric-cursor head, 3 epochs. At 12.6 k tok/s (ctx 8k is faster): ≈ 2 days. Peak LR 3e-5 (run 1/2 NaN history), step saves every 1k. | 4 × A100, 2–3 days | val loss + 20-song sweep rendered through R2; PER/CLAP vs YuE2 on the same prompts |
| G4 | R4 GRPO/DPO, 2k prompts, 4 samples each, 30 s excerpts. | 4 × A100, 1–2 days | SongBench/PER single-sample ≥ YuE2 best-of-8 on a 50-prompt WSB subset |
| G5 | R5/R6 features: continuation, cover, section edit format (fine-tune 0.5 epoch with edit documents), 12-min demo, WKVM serving with state snapshots. | 2 × A100, 3 days | demos + latency table |
| G6 | Evaluation on all 192 WildSongBench prompts with SongEval + PER (HeartTranscriptor) against YuE2 (same renderer) and Suno v6 samples from the public demo set; write-up. | 1 × A100 | report |

Ordering constraint: the cb0 stage-1 run 2 owns all 4 GPUs until ~2026-09-15 21:30. G0 needs one GPU (YuE2 peak 11–14 GiB; our GPUs have 13–15 GB free next to the 25–27 GB trainer) — **G0 can start alongside training** if the trainer's headroom holds; watch for OOM. G1 downloads are network-bound and can start now with VAE/MERT encode on the shared GPU.

### 8.5 Risks (honest)

- **Inverse-tokenizer ceiling.** 16 % exact match sounds low; whether "near-miss codes render almost identically" holds for *Suno* audio (not YuE2's own) is exactly the G0 gate. If real-audio round-trips are muddy, R2 (NAR LoRA on real latents) is the fix, and it is already shown to work at small scale.
- **Distribution mismatch in the NAR conditioning.** The NAR attends to the text prefix and to the semantic tokens; RWKV output plus a YuE2-format prefix from the same lyrics/style must look like YuE2 output. Minted data in training (grammar regulariser, as Mothersuperior does 50/50) protects this.
- **License.** Everything downstream of YuE2 weights (tokens, LoRA, minted data, MERT-v2) is CC BY-NC 4.0; suno-94k is research-only. Fine for experiments; a public release still needs §5's permissive renderer and a licence review of the corpus. The RWKV stage-1 weights themselves are trained on tokens derived through NC models — treat them as NC until reviewed.
- **Stage-1 stability.** Run 2 hit a second non-finite loss at step 6,057 (LR 5.4e-5, well below run 1's 9.4e-5), supervisor resumed from `step-6000.pth` within one minute. Two NaNs at different LRs point at a kernel/data edge case rather than LR alone; before G3, test the head-CE kernel at vocab 98,816 and add per-step grad-norm logging so the trigger can be found.
- **Compute.** Total plan ≈ 10 GPU-days on 4 A100s after the cb0 run; nothing here needs more than 40 GB per GPU (YuE2 3B in bf16 + LoRA training fits 24 GB; RWKV 3B at ctx 8k with ZeRO-2 fits as today).

## 9. Status log

- **2026-09-12** Shard 0 of suno-94k tokenized: 1,135 clips, 0 failures, 64.4 h audio, 12.8 min
  on one A100 (302× realtime), 177 MiB npz. Stats: median song 203 s (10.2k cb0 tokens), p95 327 s
  (16.4k tokens → exceeds a 16k context: truncate or drop >5 min songs for stage-1), lyrics
  median 1,072 chars, script mix 85 % Latin / 6 % Cyrillic / 6 % CJK, 5,054 distinct caption tags
  in one shard, 10 % of songs share exact lyrics with another song (dedupe by lyric hash), 1,091
  distinct creators (no creator dominates). Full run launched: shards 1–42 on GPU 0, 43–84 on GPU 1
  (`logs/prep_suno94k_gpu{0,1}.log`); expected ≈ 9–10 h. Tokens ≈ 15 GB total, tars deleted as they finish.
- **2026-09-12 22:15** Progress check: 19/85 shards done (0–9, 43–52), 0 clip failures, 13.2 min tokenize
  per shard, but the wall-clock cycle was 21–23 min and rising because the worker downloads each tar
  serially and the HF CDN edge degraded to <1 MB/s (see §0 caveat). Pinned the fast edge in `/etc/hosts`
  (≈100 MB/s per download again) and launched `scripts/prefetch_shards.py`. Remaining 66 shards at the
  13-min tokenize cycle → done ≈ 05:30 on 2026-09-13. Disk: 467 GB free, steady-state raw audio <25 GB;
  user will mount additional disk before the humair025 / MTG-Jamendo phases. Leftover from the shard-0
  test run: `data/raw/suno94k/suno-various-94k-00000.tar` (5.4 GB), safe to delete.
- **2026-09-13 09:20** Overnight run finished 80/85 shards with 0 clip failures (14 GB of npz). Shard 84 is
  the half-size last shard (570 clips; index lists 1,140 files = mp3+json pairs), not a truncation. Shards
  28, 29, 71, 72, 73 were skipped: a ~1 min CDN blip around 03:15 made the worker's own `download()`
  exhaust its 3 retries (10/20/30 s backoff) before the prefetcher had those tars. Relaunched
  `run_all_shards.sh` (resumable; now appends to logs instead of truncating — the overnight log text was
  lost to the old `>`), GPUs 0/1 back at 100 %; expected complete ≈ 10:05. Prefetcher still running and
  will exit on its own. Follow-ups: raise worker download retries / backoff, then `dataset_report.py`.
- **2026-09-13 14:00** suno-94k complete: 85/85 shards, 94,174 songs, 0 failures, 5,423 h, 976 M cb0
  tokens, 15 GB npz, all arrays validated. Full-corpus stats: duration p50 205 s / p95 334 s, 8,819 songs
  > 300 s, 817 < 30 s; lyrics 87 % Latin / 5.5 % Cyrillic / 5.6 % CJK; ~11.7k instrumental-only; 53k
  creators (top share 0.5 %); Suno versions auk 37 % / fenix 34 % / crow 16 %. Stage-1 manifest:
  84,500 kept (dropped 8,819 too long, 817 too short, 38 over ctx), **train 83,714 songs / 4,500 h /
  845 M tokens (810 M cb0 + 35 M text)**, val 775 songs / 7.8 M tokens (11 removed for lyric overlap
  with train). Doc length p50 10,343 / p95 14,633; text p50 395 tokens. binidx written and spot-checked
  (decoded text == manifest text, cb0 == npz, control tokens in place). `magic_prime` train = 51,563,
  val = 467. Download retries in `prepare_suno94k.py` raised 3 → 10. Shard-0 test tar deleted.
  **Next (phase 3):** resize the g1k checkpoint to vocab 67,072 (`scripts/resize_vocab.py`, to write),
  then a smoke run of RWKV-LM `train.py` on 4 GPUs: `--data_type binidx --vocab_size 67072 --ctx_len 16384
  --magic_prime 51563`, note that `epoch_steps × real_bsz` must equal 40,320 (RWKV-LM assert), so with
  4 GPUs × micro_bsz 1 use `--epoch_steps 10080`. The extra corpora (humair025, MTG-Jamendo) wait for
  the additional disk the user is mounting.
- **2026-09-13 16:15 Stage-1 training launched** (`scripts/train_stage1.sh 0,1,2,3`, log
  `logs/train_stage1_20260913_160907.log`, per-step loss in `out/stage1/step_log.txt`: columns
  step, loss, lr, kt/s, Gtokens, time). Getting there took: pytorch-lightning 1.9.5 + deepspeed 0.16.9
  install, `resize_vocab.py` (emb/head → 67,072 rows), and four local patches to RWKV-LM train_temp
  (LoRA-dim env overrides, sm_80 atomics in 3 kernels, compile-time vocab for the chunked head-CE kernel,
  step logging in `trainer.py`; originals saved as `*.orig`, details in §3). Two false starts: the first
  launch died on the head-CE kernel's hard-coded vocab 65,536; the second ran but logged nothing per step.
  Measured: 4 × A100, micro_bsz 1, ctx 16,384, ZeRO-2 + grad_cp + head_chunk 65,536 → **25.6–27 GB per GPU,
  12.6 k tokens/s total (5.2 s per 65,536-token step)**. That is ≈ 18.6 h per pass over the 845 M-token
  train set and ≈ 37 h to the 1.69 B-token cosine horizon (`rwkv-final.pth`); first checkpoint
  `rwkv-0.pth` after 10,080 steps ≈ 14.5 h. Loss: 16.4 at step 1 → 11.8 at step 10 (new head rows learning
  the audio unigram), no NaN. Speed levers if wanted later: micro_bsz 2 (memory headroom ~13 GB), the
  `@rwkv3` kernel variant, or dropping grad_cp on the last layers. To watch: `tail out/stage1/step_log.txt`;
  to resume after a crash: rerun `train_stage1.sh` (train_stage 3 picks the newest `out/stage1/rwkv-*.pth`).
  **Next:** an eval script over `data/binidx/suno94k_stage1_val` (cb0 loss by position, text vs audio) to
  run on `rwkv-0.pth`, then a 20-song generation sweep through YuE1 stage-2 + X-Codec decode (§4).
- **2026-09-14 08:30 Run 1 diverged; run 2 launched.** Loss fell 16.4 → 6.1 (step 60) → 4.5 (first 500)
  → 3.30 (steps 3500–4640, still improving), then **NaN at step 4645** (23:00) with no spike beforehand
  (max loss after step 1000 was 3.89) and nothing unusual in the 36 windows around it (checked token ids,
  runs, text/audio mix). LR was 9.4e-5, just past the 1e-4 peak. bf16 ZeRO-2 has no overflow skip, so one
  inf gradient turned 1,059/1,062 tensors NaN; the run then burned 9 h producing NaN (moved to
  `out/stage1/nan-rwkv-0.pth`, no usable checkpoint — the only save was the 10,080-step epoch save).
  Fixes for run 2: peak LR **6e-5 → 6e-6**; `trainer.py` saves `step-N.pth` every 1,000 steps (≈ 1.4 h,
  keeps 2) and kills the run on a non-finite loss; `scripts/train_stage1.sh` is now a supervisor that resumes
  from the newest `step-N.pth` with `RWKV_STEP_OFFSET=N` (LR schedule, warmup and data windows continue
  from N; `dataset.py` patched), uses `train_stage 0` so `train.py` does not override `--load_model`, and
  stops only when `rwkv-final.pth` (cosine horizon) exists. The trainer's own one-pass save was renamed
  `rwkv-onepass.pth`. Events (saves, NaN kills, restarts) go to `out/stage1/events.txt`. Run 2 restarts from
  `rwkv-init.pth` (7 h of good training lost). ETA at 12.6 k tok/s: one pass ≈ 18.6 h, horizon ≈ 37 h → ~21:30 on 2026-09-15.
- **Disk plan (2026-09-14):** 450 GB free. Project footprint now 47 GB (tokens 15, binidx 3.2, models 16,
  out 12). Growth if the plan is followed with streaming (raw audio deleted after tokenizing): stage-1
  checkpoints ≤ 6 GB each, keep ~5 → 30 GB; humair025 tokens ≈ 7 GB; MTG-Jamendo tokens ≈ 8 GB; YuE1 stage-2
  + decoders ≈ 3 GB; generated audio negligible → **≈ 100 GB total, fits without the new disk.** Keeping raw
  audio would need 454 + 213 + 117 ≈ 785 GB → that is what the extra disk is for, if wanted.
- **2026-09-14** Reviewed YuE2 for reuse (`docs/yue2_reference/`): no audio→semantic-token encoder is released (inference-only pipeline), the NAR only renders YuE2's own 32,768-entry semantic tokens, and all weights are CC BY-NC 4.0. Decision at the time: stay on X-Codec + YuE1 stage-2 (Apache-2.0); YuE2 evaluation-only. Added License row to §0 and renderer note to §5.
- **2026-09-14 (later)** User decision: **target YuE2-level quality, licensing deferred, experiments first.** Added §7 (Track B inverse-tokenizer distillation, Track A own NAR → YuE2-Vae, GPU schedule), rewrote §0 Audio-tokens / Renderer / License rows, marked §4 YuE1 path as baseline. Stage-1 cb0 run 2 left running (step 5160, loss 3.29 at 16:06). YuE2 weights still to download. Status log renumbered to §8.
- **2026-09-14 18:10 Generation-2 design (§8).** User goal restated: build on YuE2 with the RWKV backbone and surpass YuE2 and Suno v6. Researched: Suno v6 family released 2026-09-09 (v6 / v6-wild / v6-mini; 8-min songs, plain-language section edits, mashups, image-to-song, licensed data; scores *below* Suno v5 and YuE2 on WildSongBench per YuE2's page). Found the missing audio→YuE2-token encoder as a third-party release (Mothersuperior v4 head on MERT-v2-FullSong L20, 16.1 % top-1, plus NAR LoRA for real audio and a 4,720-song paired minted corpus); confirmed YuE2's own tokenizer is a causal MERT-v2 branch. Wrote §8 (three axes: lyric/caption adherence + RL, audio-input & state-based editing & unbounded length, Suno-v6 product features; phases G0–G6 with gates). Downloaded YuE2-Vae, YuE2-Vae-legacy, MERT-v2-FullSong to `models/yue2/`; YuE2-3B (7.26 GB) downloading (`HF_HUB_DISABLE_XET=1` required). Stage-1 cb0 run 2: second NaN at step 6,057 (LR 5.4e-5), auto-resumed from `step-6000.pth`, loss 3.03–3.08 at step 6,330. **Next:** G0 — YuE2 venv (torch 2.10), fetch Mothersuperior head/LoRA + minted tokens, round-trip 20 suno val songs.
- **2026-09-14 20:50 G0 round trips work (6 real Suno songs, shard 0).** Setup: `venvs/yue2` (Python 3.12, torch 2.10.0+cu128 aarch64 from download.pytorch.org via aria2c, transformers 4.57.6, `tools/YuE` = yue2-infer 0.1.6 cloned through gh-proxy.com); weights in `models/yue2/` (YuE2-3B, YuE2-Vae, YuE2-Vae-legacy, MERT-v2-FullSong, `mothersuperior_v4/` head + NAR LoRA); shard-0 tar re-downloaded to `data/raw/suno94k/` (kept, 5.4 GB) for real audio. Scripts: `tools/yue2_roundtrip.py` (audio → MERT L20 → v4 head → NAR(±LoRA) → VAE; also writes the VAE-only round trip), `tools/yue2_roundtrip_eval.py` (HeartTranscriptor WER vs lyrics and vs the original's transcript; MERT-v2 recording-embedding cosine), `tools/yue2_mint.py` (B2 minting from suno prompts). Runs fit next to the trainer: ~8.5 GiB peak, 6 songs in 158 s on a shared A100 (NAR 32 steps ≈ 15 s/song). **Results (mean of 6):** WER vs lyrics — original 0.27, VAE-only 0.35, token round trip 0.48 (NAR+LoRA) / 0.44 (stock NAR); WER vs the original's own transcript 0.42 / 0.44; MERT cosine to the original 0.97 for both round trips (VAE-only 0.99, unrelated song 0.64); token repeat rate 0.115; per-frame latent MSE ≈ 1.3 at latent variance ≈ 0.95 (flow matching resamples, so this is not a useful metric). Reading: style/identity survive the 32k-token bottleneck; lyric intelligibility roughly halves with the v4 head (16 % top-1), and the v4 NAR LoRA gives no WER gain on Suno audio. **Gate: on-style yes, intelligible partially → proceed with R1 (better inverse tokenizer) as the first lever; R2 to be re-judged on our own LoRA.** Files: `out/yue2_roundtrip/<id>/{A_original,B_vae_only,C_roundtrip_lora_s32,C_roundtrip_base_s32}.flac` for listening. Running: `yue2_mint.py --n 5` on GPU 0 (true-token ceiling test: render minted songs from true vs predicted tokens), 20-song round trip + eval on GPU 1 (`logs/yue2_roundtrip_n20.log`). Minted-corpus per-track files (`data/yue2_minted/tracks/`) trickle in at 5–80 KB/s (new CDN host `us.aws.cdn.hf.co`, all edges slow); the regularizer pack (all 4,732 token sequences + prompts, 100 MB) is complete.
- **2026-09-14 21:30 G0 confirmed on 20 songs; the inverse tokenizer is the bottleneck, not the renderer.** 20 real Suno songs (shard 0): WER vs lyrics 0.22 original / 0.26 VAE-only / 0.48 (v4 head + v4 NAR LoRA) / 0.46 (v4 head + stock NAR); WER vs the original's transcript 0.17 / 0.44 / 0.44; MERT cosine 0.99 / 0.97 / 0.97 (unrelated song 0.64). **Ceiling test on 5 YuE2-minted songs** (`tools/yue2_mint.py`, cot=off, suno shard-1 prompts, ~1.3× realtime on a shared A100): rendering from the *true* tokens gives WER-vs-original 0.20 and cosine 0.99; from v4-predicted tokens 0.36 and 0.92 (one rap song collapsed: head top-1 0.8 % vs 13–18 % on the others). So NAR+VAE reproduce a song from its tokens nearly losslessly for ASR purposes and the v4 head loses about half the lyric intelligibility → **R1 first.** R1 pipeline built and running: `tools/yue2_extract_mert.py` (MERT-v2-FullSong layers 12/16/20/23 at 25 Hz; corpus tracks are re-synthesised from `latent.npy` with YuE2-Vae, which equals their flac, so the 160 GB audio is not needed), `scripts/train_inverse_tokenizer.py` (112 M params, 4-layer softmax mix → 12-layer bidirectional transformer, 40 s windows, soft-CE over 8 codec-embedding neighbours from `data/yue2_minted/codec_nbr_*.npy`), first run `out/inverse_tok/v1` on GPU 2 (590 corpus tracks + 5 own, 6k steps) — `logs/train_inverse_tok_v1.log`; compare with `yue2_roundtrip.py --head-ckpt out/inverse_tok/v1/best.pt` on the same 20 songs. Overnight: `yue2_mint.py --n 400` on GPU 0 (`data/yue2_minted_own/`, ~2.5 min/song → ~17 h), minted-corpus trickle download, extraction to rerun on new tracks. **Cost:** the cb0 trainer runs at 5–7 k tok/s instead of 12.5 k while GPUs 0–2 are shared (ZeRO waits for the slowest card); its horizon slides accordingly (step 8,300 at 21:09, loss ≈ 3.1–3.3).
- **2026-09-15 11:45 R1 v1 negative, v2 queued; real-audio prep running.** v1 (590 minted tracks, 112 M params, 6k steps): held-out top-1 peaked at **7.1 %** at step 1,750 (top-5 20 %), then overfit (train loss 6.5 → 2.7 while val fell to 6.1 %); on the 20 real songs it is *worse* than the v4 head (WER vs lyrics 0.52 vs 0.48, MERT cosine 0.96 vs 0.97). Expected: v4 trained on 4,765 tracks; exact-match accuracy needs data, not depth. All 4,720 corpus tracks (latent + semantic) finished downloading overnight, plus 264 own minted songs (`data/yue2_minted_own/`, 259 new). MERT features for all of them extracting on GPUs 2/3 (~2,200 tracks each, ~1 h). **v2** queued after that on GPU 2: same model, ~5k tracks, 20k steps, dropout 0.2, 1-s time masking, 10 % feature dropout (`logs/train_inverse_tok_v2.log`). Also started `tools/yue2_prep_real.py --shard 0` on GPU 0: 873 real Suno songs → MERT L12/16/20/23 + true VAE latents + cot=off prefix (`data/yue2_real/shard-00000/`), the inputs for the NAR-teacher flow loss (joint head + NAR-LoRA training, next script). Stage-1 cb0 trainer: step 15,510, loss ≈ 2.95–3.1, back at 12.5 k tok/s when GPUs were free this morning.
