#!/bin/bash
# Gate-1 audio test on the v4-latent cursor LM: wait for the 20k EMA checkpoint, then own / swap / none Whisper CER
# (same 24 held-out crops as out/clm_eval/cursor1_25k). usage: gate1_cursor_v4.sh GPU STEP
cd /root/x/rwkv-music; S=${2:-20000}; CK=out/clm/cursor_v4/M_ema_s$S.pt
while [ ! -f $CK ]; do sleep 120; done; sleep 60   # let torch.save finish
echo "$(date) $CK found, evaluating"
CUDA_VISIBLE_DEVICES=${1:-3} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venvs/yue2/bin/python vae/eval_cursor_lm.py \
  --ckpt $CK --vae out/vae/v4/G_s60000.pt --out out/clm_eval/cursor_v4_${S%000}k
echo "$(date) done rc=$?"
