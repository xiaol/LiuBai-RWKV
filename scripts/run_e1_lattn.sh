#!/bin/bash
# E1 arm "gated + local attention" (vae/gated_nar.py --lattn): same config as run_e1.sh (cursor_v4b setup, v4 latents, 20k steps),
# then the same 96-crop causal CFG 3 / 5 gate plus CFG 1 (binding without CFG). Compare paired vs e1_gated_s{0,1} and cursor_v4b.
# usage: run_e1_lattn.sh GPU SEED [TAG]   (TAG: run-name suffix, e.g. _wkv)
cd /root/x/rwkv-music; G=$1; S=$2; N=e1_lattn_s$S$3
export CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=venvs/yue2/bin/python
$PY vae/train_cursor_lm.py --mode cursor --head gated --lattn --seed $S --out out/clm/$N --vae out/vae/v4/G_s60000.pt --sem-w 3 --steps 20000 \
  --manifest data/real_corpus/manifest/pilot_clm_train_clean.jsonl --val-manifest data/real_corpus/manifest/slim_val_clean.jsonl > logs/clm_$N.log 2>&1
echo "$(date) training rc=$?"
for C in 1 3 5; do
  $PY vae/eval_cursor_lm.py --ckpt out/clm/$N/M_ema_s20000.pt --vae out/vae/v4/G_s60000.pt --n 96 --causal-prompt --cfg $C --conds own,swap \
    --out out/clm_eval/${N}_20k_c96_cfg$C > logs/gate1_${N}_cfg$C.log 2>&1
done
echo "$(date) gate done"
