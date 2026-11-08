#!/bin/bash
# Waits until the lat_v5 precompute (GPUs 0-2) has exited, then runs the live-WKV local-attention E1 arm on GPU 0
# (run_e1_lattn.sh 0 0 _wkv: training 20k steps + CFG 1/3/5 gate). Cancel: pkill -f queue_e1_lattn_wkv.sh
cd /root/x/rwkv-music
while pgrep -f tools/precompute_latents.py > /dev/null; do sleep 60; done
sleep 120; echo "$(date) precompute gone, starting e1_lattn_s0_wkv on GPU 0"
bash scripts/run_e1_lattn.sh 0 0 _wkv
