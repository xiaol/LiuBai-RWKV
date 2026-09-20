# RWKV-7 Song LM — Plan

Goal: a **pure RWKV-7 (attention-free) lyrics+style → song model**, following the YuE
recipe (text → semantic audio tokens → acoustic rendering), trained on the 4×A100-40GB box.
Project root: `/root/x/rwkv-music`. Written 2026-09-12. Chronological record: **`docs/STATUS_LOG.md`**.

## Where we are (2026-09-20) — read this first

**Live pipeline (all under `venvs/yue2` unless noted):** real Suno audio → MERT-v2 L12/16/20/23 → **R1-a inverse tokenizer**
(`out/joint/v2/head_best.pt`, real-audio joint teacher) → YuE2 semantic tokens (25 Hz, 32,768) → **RWKV-7 3B stage-1** (rwkv_py312 env,
vocab 98,816, ctx 8,192) → YuE2 NAR + R1-a LoRA + YuE2-Vae → 48 kHz stereo. Corpus in this space: 93,058 songs (`data/yue2_corpus/suno94k`),
opus audio kept for 40k of them, lyric↔time alignment for those 40k (`data/lyric_align/fa`, MMS forced alignment, 600× realtime).

| Piece | Best artefact | Gate result | Status |
|---|---|---|---|
| R1 inverse tokenizer | `out/joint/v2/{head,lora}_best.pt` (R1-a) | unseen real songs: WER vs original transcript 0.33 / MERT cos 0.981 (v4 head 0.50 / 0.971) | **accepted 2026-09-16**; v3 (22k real songs, online MERT) did not beat it |
| G3 stage-1, whole-song format | `out/stage1_yue2/rwkv-final.pth` | audio WER vs lyrics 1.37, cos 0.80; teacher-forced: ignores the lyrics (shuffled − matched = 0.10 nats) | **negative** 2026-09-18 |
| R3.1 stage-1, section format | `out/stage1_sec/rwkv-final.pth` | teacher-forced binding appeared (permuted − matched 0.0005 → 0.106); audio WER 1.05, lyric-word overlap 10 % (G3 7 %, original 80 %), cos 0.79 | **better, still no lyric adherence** 2026-09-20 |
| R2 NAR LoRA, R4 GRPO, R5 audio ops, R6 length, R7 ABC | – | – | not started |

**Open decision (R3.2, see last entry of `docs/STATUS_LOG.md`):** line-level interleaving, higher LR / more epochs, section-only data; and,
if the transcript overlap stays under ~30 %, an enforced objective (ASR/CTC reward on rendered audio or the joint teacher's flow loss).
Also cheap and not yet done: align the other 53k songs (their audio was not kept; re-streaming is network-bound, ≈ 13 min/shard).

**What in this document is stale (kept for the record, do not execute):** §1–§3 describe the X-Codec cb0 pipeline (frozen 2026-09-15,
`out/stage1/step-20000.pth`); §2.3's section format is now implemented in YuE2 token space with **SOS = 65541**
(`scripts/yue2_layout.py`, `scripts/build_yue2_sec_binidx.py`); §4's YuE1 renderer is dropped; §5's "section alignment via
HeartTranscriptor" is done another way (MMS forced alignment, `tools/lyric_align_fa.py`; the Whisper aligner is 50× slower);
§7.1 B2/B4 are done (G1); §8.3 R3's "lyric cursor" is not built; §8.4's G3 row has a negative result; §8.6's schedule is history.
Disk is at 97 % (≈ 28 GB free); backups under `/models/rwkv-music/`.

## 0. Decisions already made (and why)

| Decision | Choice | Reason |
|---|---|---|
| Backbone | RWKV-7, no attention layers | fits WKVM state-native engine; fixed-size state; user preference |
| Init | continue-pretrain **rwkv-g1k-3b-temp-5441.pth** (in-progress g1k data run, L32-D2560, world vocab 65536; from `BlinkDL/temp-latest-training-models`) — user's choice 2026-09-12. `rwkv7-g1j-2.9b-20260831` is downloaded as the released fallback | competitive from scratch is impossible on 4 GPUs. Caveat: the temp checkpoint is mid-training (step 5441) and will be superseded; record its filename in every run config |
| Audio tokens | **Product token space = YuE2 semantic tokens** (decision 2026-09-15, §8.6). The X-Codec cb0 stage-1 run is **frozen at step 20,000** (`out/stage1/step-20000.pth`, kept as the R3 init candidate); cb0 tokens (codebooks 0–7, `data/tokens/suno94k`) stay on disk as a baseline. **Target:** YuE2 semantic tokens (25 Hz, 32,768 entries, ids `151853..184620` in YuE2's vocab), reached by *distilling an inverse tokenizer* (§7 Track B), because YuE2 ships no audio→token encoder | X-Codec is the only open song tokenizer with a released encoder. YuE2 / HeartMuLa / MiniMax ship generators only: their AR models emit semantic tokens from text, no audio→token encoder, so their token spaces cannot be used as a training target without §7. YuE2's NAR only renders YuE2 tokens; X-Codec and YuE2 token spaces are unrelated (vocab, rate, meaning), no adapter exists. YuE2-Vae does include an encoder, but it yields continuous 64-dim latents, not tokens |
| Renderer (stage-2) | **target = YuE2 quality (48 kHz stereo)**: YuE2 NAR + YuE2-Vae via Track B in §7, or own NAR → YuE2-Vae via Track A. YuE1 `YuE-s2-1B-general` + X-Codec/Vocos was the 16 kHz baseline path for the cb0 run; **dropped 2026-09-15** with the cb0 run (§8.6) | user decision 2026-09-14: YuE2-level output is the goal, experiments first; 2026-09-15: focus on the SOTA RWKV model, R1 inverse tokenizer is the critical path |
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
- **R3 Stage-1 RWKV-7 3B in YuE2 token space (replaces the cb0 model as the product).** Data: 145k real songs (suno-94k + humair025) tokenized by R1 + 4.7k minted + our minted 5k + Jamendo instrumentals; ≈ 4.5k audio tokens per 3-min song → ctx 8,192 holds text + song; ~700 M audio tokens/epoch. Init from cb0 `step-20000.pth` (run frozen 2026-09-15, §8.6; structure and lyric conditioning transfer; codec rows re-initialised) — or from `rwkv-init.pth` if a 500-step probe from both inits shows no transfer benefit. Format extends §2.3: `[Genre]…[Lyrics]… <EOD> <SOA><yue2codec> tokens <EOA>`, plus, once timestamps exist, YuE-style interleaved sections. **Lyric cursor:** HeartTranscriptor word timestamps on the vocal stem → per-frame index of the lyric token being sung → small auxiliary head (weight 0.05–0.1); Mothersuperior found this necessary to stop lyric drift when fine-tuning YuE2's AR, and PER is the metric where Suno leads.
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

Ordering constraint (superseded 2026-09-15): the cb0 stage-1 run 2 was stopped at step 20,350 (§8.6); **all 4 GPUs belong to §8 work now.** G0 is done (gate passed, §9 2026-09-14 21:30). Current allocation is in §8.6.

### 8.5 Risks (honest)

- **Inverse-tokenizer ceiling.** 16 % exact match sounds low; whether "near-miss codes render almost identically" holds for *Suno* audio (not YuE2's own) is exactly the G0 gate. If real-audio round-trips are muddy, R2 (NAR LoRA on real latents) is the fix, and it is already shown to work at small scale.
- **Distribution mismatch in the NAR conditioning.** The NAR attends to the text prefix and to the semantic tokens; RWKV output plus a YuE2-format prefix from the same lyrics/style must look like YuE2 output. Minted data in training (grammar regulariser, as Mothersuperior does 50/50) protects this.
- **License.** Everything downstream of YuE2 weights (tokens, LoRA, minted data, MERT-v2) is CC BY-NC 4.0; suno-94k is research-only. Fine for experiments; a public release still needs §5's permissive renderer and a licence review of the corpus. The RWKV stage-1 weights themselves are trained on tokens derived through NC models — treat them as NC until reviewed.
- **Stage-1 stability.** Run 2 hit a second non-finite loss at step 6,057 (LR 5.4e-5, well below run 1's 9.4e-5), supervisor resumed from `step-6000.pth` within one minute. Two NaNs at different LRs point at a kernel/data edge case rather than LR alone; before G3, test the head-CE kernel at vocab 98,816 and add per-step grad-norm logging so the trigger can be found.
- **Compute.** Total plan ≈ 10 GPU-days on 4 A100s after the cb0 run; nothing here needs more than 40 GB per GPU (YuE2 3B in bf16 + LoRA training fits 24 GB; RWKV 3B at ctx 8k with ZeRO-2 fits as today).

### 8.6 Re-focus (2026-09-15): the product is R3, the critical path is R1, the cb0 run is frozen

**User decision (2026-09-15):** we are on the YuE2 pipeline; the goal is the SOTA RWKV music model, so
focus on the inverse tokenizer (R1) and stop spending GPUs on the X-Codec cb0 stage-1 run.

**Why the cb0 run stops here (stopped 21:26 at step 20,350, last save `step-20000.pth`, 1.33 B tokens, loss ≈ 3.0):**
- Its product (cb0 tokens → YuE1 stage-2, 16 kHz) is the §4 baseline, not the model we ship. R3 uses it only
  as an init and re-initialises the codec rows, so what transfers is the text→song-structure conditioning in
  the body; `step-20000.pth` at LR 1.2e-5 on the cosine tail carries that as well as `rwkv-final` would have.
  The remaining 0.36 B tokens at LR ≤ 1.2e-5 would have cost ≥ 12 h × 4 GPUs for a few hundredths of a nat.
- Sharing GPUs cost both sides: the trainer ran at 5–9 k tok/s instead of 12.6 k, and R1 jobs ran at 30–50 %.
- Kept: `step-20000.pth` (R3 init candidate), `step-19000.pth`, `rwkv-0.pth` (10,080-step epoch save),
  `rwkv-init.pth` (resized g1k). Deleted `nan-rwkv-0.pth`. `rwkv-onepass.pth` can go too (duplicate of rwkv-0).
  Resume, if ever wanted: `scripts/train_stage1.sh` picks the newest `step-N.pth`.
- Three non-finite-loss kills at LR 1e-4 / 5.4e-5 / 2.0e-5 (steps 4,645 / 6,057 / 17,323): the trigger is not
  the LR. Before G3: per-step grad-norm logging, a bf16 overflow guard (skip step on non-finite grad), and the
  head-CE kernel test at vocab 98,816 (§8.5).

**Is the data enough for R1? Measured so far:**

| head | train tracks | held-out top-1 (minted) | 20 real songs: WER vs lyrics / MERT cos |
|---|---:|---:|---|
| Mothersuperior v4 | 4,765 | 16.1 % | 0.48 / 0.97 |
| R1 v1 | 595 | 7.1 % (overfit after step 1,750) | 0.52 / 0.96 |
| R1 v2 (`out/inverse_tok/v2/best.pt`) | 4,984 (4,720 corpus + 264 own) | **16.4 %** at step 20k, flat from 17k | **0.70 / 0.93** (`logs/yue2_roundtrip_r1v2_n20.log`) — worse than v1 |

- **Minted exact match does not transfer to real audio.** v2 matches v4 on minted val (16.4 vs 16.1 %) yet on
  real Suno songs it is far worse than v4 (WER 0.70 vs 0.48, cosine 0.93 vs 0.97) and worse than v1 (0.52 /
  0.96); its real flow loss at joint step 0 is 1.216 vs 1.148 for v1. Reading: 20k steps on minted-only
  audio (YuE2 renders) with dropout/time-masking fit the *minted MERT feature distribution*; real Suno audio
  is out of that distribution. v4 generalises because it was trained jointly with the real-audio flow loss.
  **Consequence: the real-audio joint teacher is mandatory for R1, and minted-val top-1 is only a grammar
  sanity check, not a selection metric. Select heads by the 20-song real round trip and the real held-out
  flow loss.** Regularised minted-only pretraining may still hurt: compare joint runs from the v2 and v1 inits
  (`out/joint/v2`, `out/joint/v1b`).
- Exact match scales with the number of *distinct* token sequences: 8.4× the tracks (v1 → v2) gave 2.3× the
  top-1, and with the same ~5k tracks as v4, v2 lands at v4's number. More minted songs raise it; on a
  dedicated A100 minting runs at **≈ 50 s per song (≈ 1,700 songs per GPU-day)**, not the 2.5 min measured
  beside the trainer, so 10k songs ≈ 1.5 days on 4 GPUs — affordable, but by the point above it is the
  secondary lever (CE anchor / grammar), not the one that fixes real audio.
- Exact match is a proxy. What the product needs is (1) tokens from *real* Suno audio that the NAR renders
  faithfully (round-trip WER/PER + MERT cosine, later SongEval) and (2) R3 trained on those tokens producing
  songs the NAR renders well. Objective (1) is optimised directly by the **joint teacher**
  (`scripts/train_joint_teacher.py`: head → straight-through codec embeddings → frozen YuE2 AR+NAR (+LoRA) →
  flow-matching loss against the song's true VAE latents). Real Suno audio is unlimited (94k songs); minted
  soft-CE stays as the grammar anchor.
- The real constraint is **disk, not data:** 146 GB free; stored MERT L12/16/20/23 features cost 38 MB per song
  (33 GB per ~850-song shard; `data/yue2_minted/tracks` alone is 171 GB), so at most 2–3 more real shards fit.
  Fix: **online MERT** in the joint trainer — decode the stored latent with YuE2-Vae (or read the tar audio) and
  run MERT-v2-FullSong on the fly; then only latents (1.8 MB/song) + prefix live on disk and all 94k songs are
  usable. Normalisation: per-track stats from a full-track MERT pass per song per step (≈ 0.5 s on an A100) or
  global stats (`logs/mert_stats.log`); retrain v2 with global stats to check the top-1 cost before switching.
- Cheap minted augmentation (no AR): re-render the 4,720 corpus token sequences with new NAR seeds (≈ 15 s/song)
  → same tokens, new audio; attacks v1-style overfitting but adds no token diversity. Second priority.

**Running since 2026-09-15 21:30 (4 GPUs, no trainer):**
- GPU 0: joint teacher from the v2 head, real shards 0–1 (1,697 songs), 6k steps → `out/joint/v2`, `logs/train_joint_v2head.log`.
- GPU 1: v2 head round trip on the 20 real songs + eval → done 21:50 (`logs/yue2_roundtrip_r1v2_n20.log`, variant `lora_r1_v2_s32`, result in the table above); then the control joint run from the v1 head → `out/joint/v1b`, `logs/train_joint_v1bhead.log`.
- GPUs 2–3: `yue2_mint.py --n 300` from suno shards 2 and 3 (seed bases 2000/3000; 246 + 262 songs after the length filter) → `data/yue2_minted_own/`, ≈ 50 s/song → done ~01:30; MERT extraction (`tools/yue2_extract_mert.py`) afterwards.

**Joint-teacher results (2026-09-16 00:40; 6k steps, real shards 0–1, minted anchor; each head rendered with its own NAR LoRA, 32 steps):**

| head | shard 0, 20 songs (seen in joint training): WER vs orig transcript / vs lyrics / cos | **shard 2, 20 unseen real songs:** WER vs orig / vs lyrics / cos | 10 fresh minted: WER vs orig / cos |
|---|---|---|---|
| originals (ASR floor) | – / 0.22 / – | – / 0.47 / – | – |
| true tokens (ceiling) | – | – | 0.35 / 0.996 |
| v4 head + v4 LoRA | 0.44 / 0.48 / 0.968 | 0.50 / 0.60 / 0.971 | 0.38 / 0.994 |
| **joint v2** (`out/joint/v2/{head,lora}_best.pt`) | **0.29 / 0.34 / 0.980** | **0.33 / 0.52 / 0.981** | **0.38 / 0.994** |
| joint v1b (`out/joint/v1b`) | 0.39 / 0.44 / 0.971 | 0.45 / 0.60 / 0.974 | 0.47 / 0.988 |
| v2 minted-only | 0.67 / 0.70 / 0.933 | – | – |

Logs: `logs/yue2_roundtrip_eval_joint_n20.log`, `logs/yue2_roundtrip_eval_s2.log`, `logs/yue2_roundtrip_eval_ceil10.log`; audio under `out/yue2_roundtrip{,_s2,_ceil10}/`. Shard-2 originals transcribe badly (WER 0.47 vs lyrics, several non-English / screamed songs), so compare *relative to the original's transcript*: joint v2 loses 0.33 there where v4 loses 0.50 — a third less lyric damage, and on unseen audio. On minted songs joint v2 ties v4 and sits 0.03 above the true-token ceiling. **Gate passed → R1-a = joint v2.** Two lessons: (1) minted pretraining *then* real-audio teacher is the right order (v2 init beats v1 init by 0.12 WER although v2 alone was the worst head); (2) 1.7k real songs and 6k steps already beat v4's 4.7k-song joint run, so more real audio and longer joint training are the obvious next lever, not more minting.

**Online MERT: what works and what does not (2026-09-16 morning, `scripts/online_mert.py`):**
- *VAE.decode(latent) → MERT does **not** reproduce real-audio features.* On 4 real songs, frame-level cosine
  between stored (real audio) and decoded-latent features is 0.59–0.65 at layers 12/16, 0.70 at L20, 0.80 at L23
  (1-s-smoothed: 0.88–0.96, so the slow content agrees, the fine frame detail does not); the R1-a head's tokens agree
  on only **9–14 % of frames**. The VAE round trip is a different acoustic domain for MERT. Latents-only storage is
  therefore not enough for the real-audio teacher; the head must see MERT of the *real* audio.
- *Opus at ~80 kbps keeps the features.* Recomputing MERT from the original mp3 reproduces 99.2 % of the tokens;
  from 24 kHz mono opus at libsndfile compression_level 0.7 (~80 kbps, **≈ 2.3 MB per song**) 79–80 %; at 0.25
  (~130 kbps, 4–6 MB) 96 %; at 0.9 (~20 kbps, 0.5 MB) 50 %. The 80 % level is a mild augmentation, not a domain
  shift, and 36 shards × ~2.6 GB fit the disk → **G1 now stores `shard-NNNNN.audio/<id>.opus` for shards 1–18 and
  43–60** (`prepare_suno94k_yue2.py --audio-shards`), i.e. ≈ 40k real songs for the teacher, 40× the stored-feature set.
- `scripts/train_joint_online.py` = the joint teacher with a producer thread that computes MERT on the GPU from
  opus (or, with `--require-opus 0`, from decoded latents), 2 windows per decoded song, G1 shards re-scanned every
  250 steps while G1 is still running, shards 0–2 excluded (test material). Eval unchanged (8 stored-feature
  held-out real tracks + minted val) so numbers stay comparable with `out/joint/v2`.
- Disk: stored MERT features for corpus tracks 1,501–4,720 deleted (111 GB freed; latents/semantic kept,
  regenerable with `tools/yue2_extract_mert.py`); the v3 run uses the remaining 1,500 for the minted CE anchor.

**Gates (concrete, as set before the runs):** R1 head "R1-a" is accepted when it beats v4 on the 20 real songs (WER vs lyrics < 0.44 with
cosine ≥ 0.97) *and* on the 5 minted songs of the true-token ceiling test (WER 0.36 → toward 0.20). Then freeze
it, start G1 with it (tokens + latents for suno-94k; latents are head-independent, tokens can be redone), then G3.

**Near-term schedule:**
- 09-15 night: the four jobs above.
- 09-16: online-MERT joint trainer; latents-only prep of real shards 2–4; evaluate the joint-v2 head on the 20 songs and the ceiling test; pick R1-a or iterate (more real data via online MERT first, minting second).
- 09-17 → 18: G1 with R1-a on 2 GPUs (network-bound, ~13 min/shard); R2 LoRA vs base NAR on 100 held-out token sequences on the other 2.
- 09-19 → 21: G3 (R3 stage-1 v2, 4 GPUs, 2–3 days) from `step-20000.pth` or `rwkv-init.pth` (500-step probe decides); then G4–G6 as in §8.4.

## 9. Status log

Moved to **`docs/STATUS_LOG.md`** (2026-09-20). Append new entries there, newest last; keep this file for design, decisions and the current-state block at the top.
