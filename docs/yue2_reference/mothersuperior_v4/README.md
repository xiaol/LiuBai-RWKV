---
license: cc-by-nc-4.0
base_model:
- m-a-p/YuE2-3B
- m-a-p/MERT-v2-FullSong
tags:
- audio
- music
- yue2
- tokenizer
- lora
---
# yue2-mothersuperior-realaudio-tokenizer-v4

Real-audio tooling for [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B): the **audio → semantic-token encoder** YuE2 doesn't ship, plus a
**NAR-branch LoRA** so the decoder renders real-production latents. Together they let you tokenize your own recordings, LoRA-tune YuE2's AR on
an artist, and generate new songs or covers.

## Files
| file | what |
|---|---|
| `tokenizer_head_joint_v4.pt` | MERT-v2-FullSong layer-20 features (per-track instance-normalised, 25 Hz) → 32,768 YuE2 semantic codes. 8-layer transformer, d=512, 512-frame windows. Held-out exact match on YuE2's own songs: 16.1% top-1 (near-miss codes render almost identically; ear tests of NAR round-trips sit around 95%). |
| `nar_lora_joint_v4.pt` | rank-32 LoRA on `nar_self_attn.{q,k,v,o}_proj` + `nar_mlp.{gate,up,down}_proj` (28 layers) + full `vae2llm`/`llm2vae`. Trained jointly with the head on real audio. |
| `scripts/` | the training loop and inference scripts (below). |

Trained on 4,765 YuE2 self-generated songs, then adapted to real audio. If your material sounds off, rerun `joint.py` on your own audio (step 3 below).

## Requirements
Python 3.12 venv with [`yue2-infer`](https://github.com/multimodal-art-projection/YuE) (commit 92a73cc7), torch 2.10 + cu128, torchaudio 2.10,
transformers, soundfile, scipy, safetensors, demucs; `HF_HOME` with `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`, `m-a-p/MERT-v2-FullSong`.
A 24 GB GPU is enough for every stage (14–18 GB measured with gradient checkpointing).

**You also need the minted regularizer pack**: `regularizer/minted_regularizer_pack.pt` from
[Mothersuperior/yue2-minted-corpus](https://huggingface.co/datasets/Mothersuperior/yue2-minted-corpus) (~100 MB, 4,732 YuE2-generated songs as
`{name, src, style, lyrics, codec}`). The AR trainer draws 50% of its songs from it so a small artist set cannot collapse YuE2's token grammar; the
`minted_val` items are the held-out check whose loss should stay flat. The full corpus (audio + tokens + latents) is in the same dataset if you
want to retrain the head.

**Paths are hard-coded to our pod layout** (`/workspace/tok/full`, `/workspace/real/...`, `/workspace/yue2-corpus/tracks`, `/workspace/real/ar/dataset.pt`).
Recreate that layout or edit the constants at the top of each script.

## Train an artist LoRA (folder of songs → LoRA)
Per song you need `<name>.flac`, `<name>.lyrics.txt` (**full** lyrics with `[verse]/[chorus]/[bridge]/...` tags — truncated lyrics ruin structure),
and `<name>.txt` = a style caption starting with your trigger phrase (e.g. `xyzq, in the style of xyzq. <description of the sound>`).

1. `python prep_real.py` — MERT features, VAE latents and the prompt prefix per song.
2. `python cursor_prep.py` — Demucs vocal stem → MMS forced alignment of the lyrics → lyric-cursor targets (automatic).
3. *(optional, recommended for a new artist/era)* `HOLD_TRACK=<one song name> python joint.py joint_mine 3000 1 1 tokenizer_head_joint_v4.pt nar_lora_joint_v4.pt`
   — adapts head + NAR to your audio. Otherwise use the v3 files as-is.
4. `python ar_prep.py <head.pt>` — tokenizes your songs and merges the regularizer pack into `dataset.pt` (`ar_prep.py` expects the pack's records;
   point it at the downloaded file).
5. `SCHED_STEPS=3000 CK_FROM=600 CK_EVERY=200 python ar_lora_cursor.py my_lora 1600 64 0.5 none 1e-4 0.08`
   — rank-64 AR LoRA, 50/50 artist vs minted, lyric-cursor weight 0.08, checkpoints at 600/800/1000/1200/1400/1600. **Do not train longer**: past
   ~1,500 steps the model memorises the songs.
6. `LADDER_STYLE_TRACK=<song> LADDER_LYRICS=<lyrics.txt> bash ladder.sh my_lora nar_lora_joint_v4.pt` — renders one fixed prompt from every checkpoint (optional `FINALS=<file>` with lines `tag style_track lyrics seed`). Pick by ear (ours: step 800).

## Decoder
Consider decoding with [Mothersuperior/YuE2-Vae-merge-0.666](https://huggingface.co/Mothersuperior/YuE2-Vae-merge-0.666): a weight merge of
YuE2-Vae (0.666) and YuE2-Vae-legacy (0.334). The two releases share one encoder and only differ in the decoder; the merge sits between the clean
default decoder and the more musical legacy decoder. Drop-in: `YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="Mothersuperior/YuE2-Vae-merge-0.666")`,
or point `ar_generate.py` at it.

## Inference
```bash
# new song: style caption with your trigger + lyrics, score-free
python ar_generate.py my_lora/step-800.pt nar_lora_joint_v4.pt out_tag <style_track> lyrics.txt 12
# cover: transcribe any recording with SheetSage2 (--melody-only), then
ABC_FILE=score.abc COT=melody STRIP_TEMPO_KEY=1 python ar_generate.py my_lora/step-800.pt nar_lora_joint_v4.pt cover_tag <style_track> lyrics.txt 21
# LoRA strength: AR_SCALE=0.77 ...   stock model control: pass `none` for either LoRA
```
`ar_generate.py` folds both LoRAs into the base weights and runs YuE2's own pipeline, so the stock sampler, CFG and VAE apply unchanged.

## Other scripts
`extract_full.py` + `train_v2.py` (retrain the head on the minted corpus), `nar_lora.py` (NAR LoRA alone), `teacher_train.py` (head fine-tune with the NAR as
teacher), `ar_lora.py` (AR LoRA without the cursor), `build_reg_pack.py` (rebuild the regularizer pack from a corpus).

Weights derive from YuE2-3B (CC BY-NC 4.0): non-commercial use only. Scripts are provided as-is.
