#!/usr/bin/env bash
# Lyric-cursor CTC A/B (docs/ARCHITECTURE_NOTES.md section 9): wait until no other train.py holds the GPUs, then train
# three cursor-loss variants and a repeat of the stock baseline (236M, line format, one GPU each) and score them all
# with scripts/arch_ab_eval.py. Compare against out/arch_ab/rwkv_line (first baseline) and rwkv_line_seed2 (seed noise).
# usage: setsid nohup bash scripts/queue_aux_cursor_ab.sh > logs/queue_aux_cursor_ab.log 2>&1 &
cd "$(dirname "$0")/.."
source /root/envs/env_rwkv 2>/dev/null || true
while pgrep -f "train_temp/train.py|^python train.py" > /dev/null; do sleep 300; done
echo "$(date) GPUs free; launching"
AUX_W=0.1 AUX_LAYERS=11 bash scripts/train_arch_ab.sh rwkv line 0
AUX_W=0.1 AUX_LAYERS=5,11 bash scripts/train_arch_ab.sh rwkv line 1
AUX_W=0.02 AUX_LAYERS=11 bash scripts/train_arch_ab.sh rwkv line 2
TAG=seed2 bash scripts/train_arch_ab.sh rwkv line 3
score() {  # variant gpu
  local d=out/arch_ab/$1
  while [ ! -f "$d/rwkv-final.pth" ]; do sleep 60; done
  while pgrep -f "proj_dir $PWD/$d " > /dev/null; do sleep 20; done
  CUDA_VISIBLE_DEVICES=$2 python scripts/arch_ab_eval.py --ckpt "$d/rwkv-final.pth" --mixer rwkv --fmt line --n 100 > "logs/arch_ab_eval_$1.log" 2>&1
  echo "$(date) $1 scored rc=$?"
}
score rwkv_line_aux0.1L11 0 & score rwkv_line_aux0.1L5-11 1 & score rwkv_line_aux0.02L11 2 & score rwkv_line_seed2 3 &
wait
echo "$(date) ALL DONE"
