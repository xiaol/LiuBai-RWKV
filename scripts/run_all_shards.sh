#!/usr/bin/env bash
# Tokenize all 85 suno-94k shards on two GPUs. Resumable (existing npz shards are skipped); logs are appended.
# Usage: bash scripts/run_all_shards.sh [GPU_A] [GPU_B]
set -u
cd "$(dirname "$0")/.."
A=${1:-0}; B=${2:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup python3 scripts/prepare_suno94k.py --shards 1-42  --gpu "$A" --workers 12 >> logs/prep_suno94k_gpu${A}.log 2>&1 < /dev/null &
setsid nohup python3 scripts/prepare_suno94k.py --shards 43-84 --gpu "$B" --workers 12 >> logs/prep_suno94k_gpu${B}.log 2>&1 < /dev/null &
echo "launched: shards 1-42 on GPU $A, 43-84 on GPU $B; logs in logs/prep_suno94k_gpu*.log"
echo "monitor:  grep -h '^\[shard' logs/prep_suno94k_gpu*.log ; ls data/tokens/suno94k | wc -l"
