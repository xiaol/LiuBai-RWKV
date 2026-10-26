#!/bin/bash
# Cursor LM on VAE v4 with the 2026-10-26 fixes: bounded-lookahead latents (VAE.encode_local: MERT on 3 s windows, 2 blocks per
# window, <= ~1 s lookahead), cleaned lyrics + deduped val (tools/clean_lyrics.py), flow loss on the MERT dims x3 (--sem-w 3).
# 20k-step gate run, then the 96-crop causal-prompt audio gate (same crops as out/clm_eval/*_causal96). usage: run_cursor_v4b.sh GPU
cd /root/x/rwkv-music; G=${1:-2}; export CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=venvs/yue2/bin/python
$PY vae/train_cursor_lm.py --mode cursor --out out/clm/cursor_v4b --vae out/vae/v4/G_s60000.pt --sem-w 3 --steps 20000 \
  --manifest data/real_corpus/manifest/pilot_clm_train_clean.jsonl --val-manifest data/real_corpus/manifest/slim_val_clean.jsonl > logs/clm_cursor_v4b.log 2>&1
echo "$(date) training rc=$?"
$PY vae/eval_cursor_lm.py --ckpt out/clm/cursor_v4b/M_ema_s20000.pt --n 96 --causal-prompt --out out/clm_eval/cursor_v4b_20k_causal96 > logs/gate1_cursor_v4b_20k_causal96.log 2>&1
echo "$(date) gate rc=$?"
