#!/bin/bash
# E1 (docs/VAE_PLAN.md): flow head = MLP (CursorLM) vs gated read-only NAR rows (vae/gated_nar.py), same config as cursor_v4b
# (v4 latents via encode_local, cleaned lyrics, --sem-w 3, 20k steps), then the 96-crop causal-prompt CFG 3 / 5 gate (own, swap)
# on the same crops as out/clm_eval/cursor_v4b_20k_c96_cfg{3,5}. usage: run_e1.sh GPU HEAD SEED [TAG]  (TAG = run-name suffix, e.g. _wkv: live-WKV re-run after the 2026-10-31 init fix)
cd /root/x/rwkv-music; G=$1; H=$2; S=$3; N=e1_${H}_s$S$4
export CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=venvs/yue2/bin/python
$PY vae/train_cursor_lm.py --mode cursor --head $H --seed $S --out out/clm/$N --vae out/vae/v4/G_s60000.pt --sem-w 3 --steps 20000 \
  --manifest data/real_corpus/manifest/pilot_clm_train_clean.jsonl --val-manifest data/real_corpus/manifest/slim_val_clean.jsonl > logs/clm_$N.log 2>&1
echo "$(date) training rc=$?"
for C in 3 5; do
  $PY vae/eval_cursor_lm.py --ckpt out/clm/$N/M_ema_s20000.pt --vae out/vae/v4/G_s60000.pt --n 96 --causal-prompt --cfg $C --conds own,swap \
    --out out/clm_eval/${N}_20k_c96_cfg$C > logs/gate1_${N}_cfg$C.log 2>&1 &
done; wait
echo "$(date) gate done"
