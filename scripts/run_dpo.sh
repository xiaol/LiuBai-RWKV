#!/bin/bash
# DPO trial (vae/dpo_cursor_lm.py, docs/VAE_PLAN.md): pairs from e1_gated_s0_wkv on 2 GPUs, then two arms in parallel
# (A: on-policy CFG-1 candidates only, B: all candidates incl. CFG 3), then the 96-crop gate at CFG 1 / 3 (own, swap) per arm.
# usage: run_dpo.sh GPU_A GPU_B [N_CROPS]
cd /root/x/rwkv-music; GA=$1; GB=$2; N=${3:-1200}; PY=venvs/yue2/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=out/clm/e1_gated_s0_wkv/M_ema_s20000.pt; P=out/dpo/p1; VAE=out/vae/v4/G_s60000.pt
CUDA_VISIBLE_DEVICES=$GA $PY vae/dpo_cursor_lm.py pairs --ckpt $CK --out $P --n $N --rank 0 --world 2 --crops-per-batch 128 > logs/dpo_pairs_r0.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GB $PY vae/dpo_cursor_lm.py pairs --ckpt $CK --out $P --n $N --rank 1 --world 2 --crops-per-batch 128 > logs/dpo_pairs_r1.log 2>&1 &
wait; echo "$(date) pairs done"
CUDA_VISIBLE_DEVICES=$GA $PY vae/dpo_cursor_lm.py train --ckpt $CK --pairs $P --out out/dpo/a_onpol --use-cfg 1 > logs/dpo_a_onpol.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GB $PY vae/dpo_cursor_lm.py train --ckpt $CK --pairs $P --out out/dpo/b_mixed --use-cfg all > logs/dpo_b_mixed.log 2>&1 &
wait; echo "$(date) training done"
for arm in a_onpol:$GA b_mixed:$GB; do
  A=${arm%%:*}; G=${arm##*:}
  ( for C in 1 3; do
      CUDA_VISIBLE_DEVICES=$G $PY vae/eval_cursor_lm.py --ckpt out/dpo/$A/M_s1500.pt --vae $VAE --n 96 --causal-prompt --cfg $C --conds own,swap \
        --out out/clm_eval/dpo_${A}_1500_c96_cfg$C > logs/gate_dpo_${A}_cfg$C.log 2>&1
    done ) &
done; wait; echo "$(date) gate done"
