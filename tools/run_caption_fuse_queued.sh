#!/usr/bin/env bash
# Wait for two GPUs with >= 39 GB free each, then run caption stage C (Qwen3-30B-A3B bf16, TP 2); retry, because other jobs on
# the shared box grab memory between the check and vLLM's allocation. usage: <manifest> <facts> <omni> <out>
cd /root/x/rwkv-music
for attempt in $(seq 1 50); do
  while true; do
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 39000 {print $1}' | head -2 | paste -sd,)
    [[ "$G" == *,* ]] && break; sleep 60
  done
  echo "attempt $attempt: GPUs $G free at $(date)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G /root/envs/vllm13-venv/bin/python tools/caption_fuse.py --manifest "$1" --facts "$2" --omni "$3" --out "$4" --tp 2 && exit 0
  sleep 120
done
