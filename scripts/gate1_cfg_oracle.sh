#!/bin/bash
# Lyric CFG + oracle-cursor test on the v2 cursor LM (same 96 causal-prompt crops as out/clm_eval/cursor1_25k_causal96).
# usage: gate1_cfg_oracle.sh GPU CFG CONDS   (waits until the GPU has < 1 GiB in use)
cd /root/x/rwkv-music; G=$1; C=$2; CONDS=${3:-own,swap,oracle}
while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1024 ]; do sleep 300; done
echo "$(date) GPU $G free, cfg $C conds $CONDS"
CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venvs/yue2/bin/python vae/eval_cursor_lm.py \
  --ckpt out/clm/cursor1/s25k.pt --vae out/vae/v2/G_s60000.pt --n 96 --causal-prompt --cfg $C --conds $CONDS \
  --out out/clm_eval/cursor1_25k_c96_cfg$C
echo "$(date) done rc=$?"
