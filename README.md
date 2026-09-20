# LiuBai-RWKV — an RWKV-7 song language model

Text (style caption + lyrics) → full song with vocals, 48 kHz stereo. The autoregressive model is a **3 B-parameter RWKV-7**
(linear attention, constant-size state, no KV cache) trained from scratch on ~93 k Suno songs in **YuE2's semantic token space**;
the tokens are rendered to audio by YuE2's non-autoregressive stage and VAE. Everything here is a work in progress and the numbers
below are the honest ones.

## 🎧 Listen — milestone 2026-09-20 (R3.1)

**[▶ Play all samples in the browser](https://htmlpreview.github.io/?https://github.com/xiaol/LiuBai-RWKV/blob/main/samples/player.html)**
(the same page is `samples/player.html`; GitHub strips audio players from README files, so the links below open the MP3 files).

20 songs from prompts the model never saw (style caption + lyrics of real Suno songs from suno-94k shard 2). Tokens sampled by the
R3.1 checkpoint (T 1.0, top-p 0.95, section-interleaved decoding), rendered with YuE2 NAR + our R1-a LoRA + YuE2-Vae, MP3 ≈ 128 kbps.
Sorted best first by the lyric WER (word error rate of an ASR transcript against the prompt lyrics, lower is better; the original
Suno recordings score 0.47 on the same ASR). Prompt text and generated section boundaries are in the matching `.json`.

| ▶ MP3 | title | style (from the prompt) | length | lyric WER |
|---|---|---|---|---|
| [062dcf1b](samples/r31_s2/062dcf1b.mp3) | Get to get (together) | disco | 3:41 | 0.90 |
| [0885311e](samples/r31_s2/0885311e.mp3) | db82xUr | high-energy industrial metal, 120 bpm, minor key | 3:16 | 0.90 |
| [0649ecd8](samples/r31_s2/0649ecd8.mp3) | We Here (just three) | bass-heavy tech house, punchy drums, dark bouncy bassline | 2:47 | 0.91 |
| [0699f253](samples/r31_s2/0699f253.mp3) | Don't Try to Understand Me | punk | 1:34 | 0.93 |
| [07be3653](samples/r31_s2/07be3653.mp3) | Ay, Qué Calor (Mashup) | breakbeat | 4:00 | 0.94 |
| [086aa970](samples/r31_s2/086aa970.mp3) | Selencio Mortal | drill rap, male voice, bounce drop, epic | 1:00 | 0.94 |
| [07608f1b](samples/r31_s2/07608f1b.mp3) | Speaking from the Heart (Part 2) | groovy bossa nova deep-house R&B chillwave, breathy vocals | 4:00 | 0.95 |
| [0827b65b](samples/r31_s2/0827b65b.mp3) | All About Peace | gritty male vocal | 1:00 | 0.97 |
| [08960a73](samples/r31_s2/08960a73.mp3) | Motown du comptable | motown | 4:00 | 0.97 |
| [07c60f75](samples/r31_s2/07c60f75.mp3) | Ready for the fall | male vocalist, breathy verses, powerful chorus, sparse acoustic guitar | 4:00 | 0.98 |
| [08dd9d6c](samples/r31_s2/08dd9d6c.mp3) | ngi ai nguh ia me kotrai | jazz | 1:00 | 0.99 |
| [064e0e69](samples/r31_s2/064e0e69.mp3) | (untitled) | choir | 4:00 | 0.90 (repetitive) |
| [06b8e143](samples/r31_s2/06b8e143.mp3) | Baila comigo (Laranja) | male vocals, reggaeton | 3:22 | 1.00 |
| [08288cb0](samples/r31_s2/08288cb0.mp3) | De Profundo (Tenor version) | gregorian chant, latin gospel, cathedral organ | 2:03 | 1.05 |
| [06c40b48](samples/r31_s2/06c40b48.mp3) | Your action reflects your love | electric guitar | 2:51 | 1.06 |
| [06fc0b11](samples/r31_s2/06fc0b11.mp3) | noche de verano | cello | 2:45 | 1.09 |
| [07dd3b1c](samples/r31_s2/07dd3b1c.mp3) | Curly Hair | motown | 1:55 | 1.17 |
| [0833d7ac](samples/r31_s2/0833d7ac.mp3) | Carrying Things up the Mountain | mandolin | 2:40 | 1.21 |
| [08e4f4ab](samples/r31_s2/08e4f4ab.mp3) | Não Gosto de Frutas | ambient | 2:31 | 1.35 |
| [09092e80](samples/r31_s2/09092e80.mp3) | Old Drama | mandolin | 1:58 | 1.79 |

**Same prompt, previous model (G3, whole-song format) for comparison.** R3.1 wins 15 of 20 prompts and has no collapsed songs;
G3's worst four (WER 3.40 / 2.46 / 2.02 / 1.86) are all in this list.

| prompt | style | R3.1 (this milestone) | G3 (2026-09-18) |
|---|---|---|---|
| Don't Try to Understand Me | punk | [▶ 0.93](samples/r31_s2/0699f253.mp3) | [▶ 1.42](samples/g3_s2/0699f253.mp3) |
| Curly Hair | motown | [▶ 1.17](samples/r31_s2/07dd3b1c.mp3) | [▶ 3.40](samples/g3_s2/07dd3b1c.mp3) |
| ngi ai nguh ia me kotrai | jazz | [▶ 0.99](samples/r31_s2/08dd9d6c.mp3) | [▶ 1.86](samples/g3_s2/08dd9d6c.mp3) |
| Carrying Things up the Mountain | mandolin | [▶ 1.21](samples/r31_s2/0833d7ac.mp3) | [▶ 2.02](samples/g3_s2/0833d7ac.mp3) |
| Não Gosto de Frutas | ambient | [▶ 1.35](samples/r31_s2/08e4f4ab.mp3) | [▶ 1.79](samples/g3_s2/08e4f4ab.mp3) |
| Old Drama | mandolin | [▶ 1.79](samples/r31_s2/09092e80.mp3) | [▶ 2.46](samples/g3_s2/09092e80.mp3) |
| Baila comigo (Laranja) | reggaeton | [▶ 1.00](samples/r31_s2/06b8e143.mp3) | [▶ 0.95](samples/g3_s2/06b8e143.mp3) |
| Your action reflects your love | electric guitar | [▶ 1.06](samples/r31_s2/06c40b48.mp3) | [▶ 0.96](samples/g3_s2/06c40b48.mp3) |

**What you will hear, honestly:** the genre, instrumentation and language follow the caption, the vocals are fluent, and songs have
verse/chorus structure. The *sung words are mostly not the prompt lyrics* yet: only 10 % of the transcript's content words occur in
the lyrics (G3 7 %, real Suno songs 80 %). Lyric adherence is the open problem; see [Status](#status-and-next-step).
Full per-song table with token-repeat rates and the original songs' scores: [`samples/README.md`](samples/README.md).

## How it works

```
style caption + lyrics ──► RWKV-7 3B (ctx 8,192, vocab 98,816)
                              │  YuE2 semantic tokens, 25 Hz, 32,768-way
                              ▼
                   YuE2 NAR (+ our R1-a LoRA) ──► YuE2-Vae ──► 48 kHz stereo
```

- **Token space.** YuE2 never released its audio encoder, so we trained one: **R1-a**, an *inverse tokenizer* from MERT-v2 features of
  real audio to YuE2 semantic tokens (`scripts/train_joint_teacher.py`, `tools/yue2_*.py`). It is trained with a real-audio *joint
  teacher*: head → straight-through codec embeddings → frozen YuE2 AR+NAR (+LoRA) → flow-matching loss against the song's true VAE
  latents, with a minted-song cross-entropy anchor. On 20 unseen real songs its round trip loses 0.33 WER against the original's
  transcript (MERT cosine 0.981); the best public head (Mothersuperior v4) loses 0.50.
- **Corpus.** suno-94k (94,174 Suno songs, 5,423 h) tokenized with R1-a into YuE2 tokens: 93,058 songs. Lyric↔time alignment for the
  40 k songs whose audio was kept, by MMS forced alignment at ≈ 600× realtime (`tools/lyric_align_fa.py`); the rest is being
  re-streamed now.
- **Sequence format (R3.1).** `header EOD (SOS <lyric section> SOA <codes> EOA)* 0`: each lyric stanza is followed by its own audio
  tokens, so the text→audio distance is seconds instead of a whole song (`scripts/yue2_layout.py`, `scripts/build_yue2_sec_binidx.py`).
  Generation mirrors it (`scripts/rwkv_generate.py --sections 1`).
- **Training.** RWKV-LM v7 `train_temp` with local patches (`tools/patches/`), 4× A100, bf16, DeepSpeed; G3 = 1.28 B tokens from
  scratch in whole-song format, R3.1 = +0.47 B tokens (2 epochs) continued in section format (`scripts/train_stage1_sec.sh`).
- **Evaluation.** ASR (HeartTranscriptor) WER of the rendered audio against the prompt lyrics and against the original's transcript,
  MERT cosine to the original for style, plus a teacher-forced *binding* test: code cross-entropy with the matched lyrics vs the same
  song with permuted lyrics (`scripts/rwkv_eval_loss_sec.py`).

## Results so far

| piece | artefact | gate result | status |
|---|---|---|---|
| R1 inverse tokenizer | `out/joint/v2/{head,lora}_best.pt` (R1-a) | unseen real songs: WER vs original transcript 0.33, MERT cos 0.981 (v4 head: 0.50 / 0.971) | accepted 2026-09-16 |
| G3 stage-1, whole-song format | `out/stage1_yue2/rwkv-final.pth` | audio WER vs lyrics 1.37, cos 0.80; teacher-forced: ignores the lyrics (permuted − matched = 0.10 nats over the whole song, 0.0005 per section) | negative, 2026-09-18 |
| **R3.1 stage-1, section format** | `out/stage1_sec/rwkv-final.pth` | binding appeared (permuted − matched 0.0005 → 0.106 nats); audio WER 1.05, lyric-word overlap 10 %, cos 0.79 | **better, still no lyric adherence**, 2026-09-20 |
| R3.2 line-level format + 80 k aligned songs | `out/stage1_line/` | probe running | in progress |

**Checkpoints are on Hugging Face: [xiaol/LiuBai-RWKV](https://huggingface.co/xiaol/LiuBai-RWKV)** (R3.1 and G3 RWKV-7 3B, R1-a head + LoRA,
training logs, samples, model card; CC BY-NC 4.0, 13 GB). Its layout mirrors `out/`, so `hf download xiaol/LiuBai-RWKV --local-dir out`
puts the files where the scripts below expect them. Data is not distributed; this repo holds the code, the plan, the log and the rendered samples.

## Status and next step

The R3.1 train loss dropped only when the second pass over the 26 k aligned songs began and the held-out loss did not follow, so the
limit is **aligned data, not steps**. R3.2 (running): (1) re-stream and align the 54 k songs whose audio was not kept → ~80 k aligned
songs; (2) **line-level** interleaving, one SOS block per lyric line (1.8–8.6 s of audio each) instead of per stanza; (3) LR 3e-5,
section-only data. Gate: the teacher-forced permuted-minus-matched gap must clearly beat R3.1's 0.106 before the full run. If the
transcript overlap still stays under ~30 %, the binding will be enforced with an objective on rendered audio (ASR/CTC reward or the
joint teacher's flow loss) rather than more LM tokens. Details and every dated entry: [`docs/STATUS_LOG.md`](docs/STATUS_LOG.md);
design and decisions: [`PLAN.md`](PLAN.md) (its top block is the current state; §1–§5 describe the earlier X-Codec pipeline, kept for the record).

## Repository layout

```
PLAN.md                    design, decisions, current-state block (read the top first)
docs/STATUS_LOG.md         dated experiment log, newest last
docs/yue1_reference/       upstream YuE1 inference code kept for reference
docs/yue2_reference/       YuE2 model/inference code + Mothersuperior v4 inverse-tokenizer reference
samples/                   rendered MP3s + prompts (R3.1: r31_s2/, G3: g3_s2/) and the browser player
scripts/
  prepare_suno94k_yue2.py  stream suno-94k, MERT → R1-a tokens (+ opus audio for the teacher)
  build_yue2_sec_binidx.py section / line interleaved binidx for RWKV-LM;  yue2_layout.py = token layout
  train_stage1*.sh         supervised RWKV-LM runs (stage1 = X-Codec, _yue2 = G3, _sec = R3.1, _line = R3.2)
  rwkv_generate.py         sample songs from a checkpoint (whole-song or --sections 1 [--unit line])
  rwkv_eval_loss_sec.py    teacher-forced matched vs permuted-lyrics binding test
  train_joint_teacher.py   R1 head + LoRA with the real-audio joint teacher;  train_joint_online.py = online-MERT variant
  r31_render_eval.sh, r32_align_chain.sh   the pipelines behind the tables above
tools/
  yue2_render_tokens.py    tokens → YuE2 NAR (+LoRA) → VAE → audio
  yue2_roundtrip*.py       ASR WER + MERT-cosine evaluation
  yue2_mint.py, yue2_extract_mert.py, yue2_prep_real.py   data for the inverse tokenizer
  lyric_align_fa.py        MMS forced alignment of lyrics to audio;  fetch_opus_shards.py = audio re-streaming
  patches/                 pinned RWKV-LM / YuE commits and the local RWKV-LM patch
```

## Reproducing the samples

```bash
# one-off: vendored upstream code (see tools/patches/README.md for the exact commits)
git clone https://github.com/BlinkDL/RWKV-LM.git tools/RWKV-LM && git -C tools/RWKV-LM checkout 9a75f9f
git -C tools/RWKV-LM apply ../../tools/patches/rwkv-lm-9a75f9f-train_temp.patch
git clone https://github.com/multimodal-art-projection/YuE.git tools/YuE && git -C tools/YuE checkout 0edaf2f

# weights: hf download xiaol/LiuBai-RWKV --local-dir out --exclude "samples/*"
# generate 20 shard-2 prompts with the R3.1 checkpoint (section decoding), render, score
python scripts/rwkv_generate.py --ckpt out/stage1_sec/rwkv-final.pth --sections 1 --shard 2 --n 20 --out out/gen/r31_s2_sec
venvs/yue2/bin/python tools/yue2_render_tokens.py --gen out/gen/r31_s2_sec --tag rwkv_r31 --eval-root out/yue2_roundtrip_s2
venvs/yue2/bin/python tools/yue2_roundtrip_eval.py --root out/yue2_roundtrip_s2 --variants rwkv_r31_s32,rwkv_final_s32,lora_r1_jv2_s32
```

Two Python environments are used: `rwkv_py312` (RWKV-LM training / sampling, CUDA kernels) and `venvs/yue2` (YuE2, MERT, ASR).
Model weights needed: YuE2 AR/NAR/VAE, MERT-v2-FullSong, HeartTranscriptor, plus our R1-a head/LoRA and RWKV checkpoints from [xiaol/LiuBai-RWKV](https://huggingface.co/xiaol/LiuBai-RWKV).

## Credits

RWKV-7 by BlinkDL ([RWKV-LM](https://github.com/BlinkDL/RWKV-LM)); [YuE](https://github.com/multimodal-art-projection/YuE) / YuE2 by
M-A-P for the token space, NAR and VAE; Mothersuperior v4 as the reference inverse tokenizer; MERT-v2 (M-A-P) features; MMS forced
alignment (Meta); HeartTranscriptor for ASR; the suno-94k dataset for prompts and training audio. The original Suno recordings used as
prompts are not redistributed here; only our generated audio is.
