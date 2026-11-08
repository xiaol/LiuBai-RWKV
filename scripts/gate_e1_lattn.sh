#!/bin/bash
# Gate for out/clm/e1_lattn_s0 (replaces the serial gate loop of run_e1_lattn.sh): waits for the training pid to exit, then runs
# CFG 1 / 3 / 5 (own, swap, 96 causal crops) in parallel. CFG 1 (the deciding number) on GPU 3; CFG 3 / 5 on a free GPU among
# 0-2 (< 2 GB used) if there is one, else also on GPU 3. usage: gate_e1_lattn.sh TRAIN_PID
cd /root/x/rwkv-music; N=e1_lattn_s0; PY=venvs/yue2/bin/python
while kill -0 $1 2>/dev/null; do sleep 60; done
[ -f out/clm/$N/M_ema_s20000.pt ] || { echo "$(date) no M_ema_s20000.pt, abort"; exit 1; }
free_gpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 < 3 && $2 < 2000 {print $1}'; }
FREE=($(free_gpus)); echo "$(date) training done, free GPUs among 0-2: ${FREE[*]:-none}"
run() {  # GPU CFG
  CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY vae/eval_cursor_lm.py --ckpt out/clm/$N/M_ema_s20000.pt \
    --vae out/vae/v4/G_s60000.pt --n 96 --causal-prompt --cfg $2 --conds own,swap --out out/clm_eval/${N}_20k_c96_cfg$2 > logs/gate1_${N}_cfg$2.log 2>&1
  echo "$(date) cfg $2 done on GPU $1 rc=$?"
}
run 3 1 &
run ${FREE[0]:-3} 3 &
run ${FREE[1]:-3} 5 &
wait; echo "$(date) gate done"
