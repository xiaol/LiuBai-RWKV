#!/bin/bash
# Lyric-CFG audio gate on cursor_v4b (v4, bounded-lookahead latents): wait for the 20k EMA checkpoint, then own / swap
# Whisper CER on the same 96 causal-prompt crops as out/clm_eval/cursor_v4_40k_c96_cfg{3,5}. usage: gate1_cursor_v4b_cfg.sh GPU CFG
cd /root/x/rwkv-music; G=$1; C=$2; CK=out/clm/cursor_v4b/M_ema_s20000.pt
while [ ! -f $CK ]; do sleep 120; done; sleep 60   # let torch.save finish
echo "$(date) $CK found, cfg $C on GPU $G"
CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venvs/yue2/bin/python vae/eval_cursor_lm.py \
  --ckpt $CK --vae out/vae/v4/G_s60000.pt --n 96 --causal-prompt --cfg $C --conds own,swap \
  --out out/clm_eval/cursor_v4b_20k_c96_cfg$C
echo "$(date) done rc=$?"
