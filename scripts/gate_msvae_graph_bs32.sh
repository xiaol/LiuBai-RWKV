#!/bin/bash
# same draws as gate_msvae_full1_20k_cfg3 (gen-bs 32), CUDA-graph path: isolates the code path from the noise draws
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "[g]ate_msvae_steps.sh"; do sleep 30; done
CUDA_VISIBLE_DEVICES=${GPU:-3} venvs/yue2/bin/python vae/eval_cursor_lm.py --ckpt out/liubai_h/msvae_full1/M_ema_s20000.pt \
  --out out/clm_eval/msvae_full1_20k_c96_cfg3_graph_bs32 --manifest data/real_corpus/manifest/slim_val.jsonl --n 96 --conds own,swap \
  --causal-prompt --cfg 3 --gen-bs 32 > logs/gate_msvae_full1_20k_cfg3_graph_bs32.log 2>&1
