#!/usr/bin/env bash
# Wait for the four line-format A/B runs (scripts/train_arch_ab.sh) and score each on its own GPU with scripts/arch_ab_eval.py.
# usage: setsid nohup bash scripts/run_arch_ab_eval.sh > logs/arch_ab_eval_driver.log 2>&1 &
cd "$(dirname "$0")/.."
source /root/envs/env_rwkv 2>/dev/null || true
N=${N:-100}
run() {  # variant gpu mixer slot_heads slot_m
  local v=$1 g=$2 mix=$3 k=$4 m=$5 d=out/arch_ab/$1
  while [ ! -f "$d/rwkv-final.pth" ]; do sleep 60; done
  while ps -eo args | grep -v grep | grep -q "proj_dir $PWD/$d "; do sleep 20; done
  echo "$(date) $v final; scoring"
  CUDA_VISIBLE_DEVICES=$g python scripts/arch_ab_eval.py --ckpt "$d/rwkv-final.pth" --mixer "$mix" --fmt line --n "$N" \
    --slot-heads "$k" --slot-m "$m" > "logs/arch_ab_eval_$v.log" 2>&1
  echo "$(date) $v scored rc=$?"
}
run rwkv_line 0 rwkv 0 1 &
run rwkv_line_slot4x1 1 rwkv 4 1 &
run rwkv_line_slot4x4 2 rwkv 4 4 &
run attn_line 3 attn 0 1 &
wait
echo "$(date) ALL DONE"
