#!/usr/bin/env bash
# Wait for one GPU with >= 39 GB free, then run caption stage A on it. usage: run_caption_omni_queued.sh <manifest> <out>
cd /root/x/rwkv-music
G=$(bash /root/x/state-bridge/scripts/wait_gpu.sh 39000); echo "GPU $G free at $(date)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G exec /root/envs/vllm13-venv/bin/python tools/caption_omni.py --manifest "$1" --out "$2"
