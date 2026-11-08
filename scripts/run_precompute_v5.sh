#!/bin/bash
# Whole-song v5 latents + letter timings for the 3B LM (tools/precompute_latents.py), pilot_clm_train_clean (333k songs, ~21k h).
# usage: run_precompute_v5.sh GPU RANK [RANK ...]   (world 6: two ranks per GPU; resumable)
cd /root/x/rwkv-music; G=$1; shift
for R in "$@"; do
  CUDA_VISIBLE_DEVICES=$G nohup venvs/yue2/bin/python tools/precompute_latents.py --rank $R --world 6 --workers 12 \
    > logs/precompute_v5_r$R.log 2>&1 &
done
