#!/bin/bash
# flow-step sweep of the 20k msvae_full1 gate (CUDA-graph fast path, gen-bs 96), CFG 3, same 96 crops as gate_msvae_full1_20k
cd "$(dirname "$0")/.."
for S in 24 12 8; do
  CUDA_VISIBLE_DEVICES=${GPU:-3} venvs/yue2/bin/python vae/eval_cursor_lm.py --ckpt out/liubai_h/msvae_full1/M_ema_s20000.pt \
    --out out/clm_eval/msvae_full1_20k_c96_cfg3_st$S --manifest data/real_corpus/manifest/slim_val.jsonl --n 96 --conds own,swap \
    --causal-prompt --cfg 3 --gen-bs 96 --steps $S > logs/gate_msvae_full1_20k_cfg3_st$S.log 2>&1
done
