#!/bin/bash
# Whole-corpus MSVAE precompute (tools/precompute_msvae_full.py): lyric phase (timed songs, 1.22M / ~77k h) then audio phase
# (instrumental / untimed / unknown, ~1.7M), 4 GPUs, one rank per GPU. Resumable: re-run the script after a crash or reboot.
# Stop: touch logs/precompute_msvae_full.stop (checked between phases) or kill the ranks.
cd /root/x/rwkv-music; mkdir -p logs; GPUS=${GPUS:-0,1,2,3}; IFS=, read -ra G <<< "$GPUS"; W=${#G[@]}
for phase in lyric audio; do
  [ -f logs/precompute_msvae_full.stop ] && { echo "$(date) stop file"; exit 0; }
  echo "$(date) phase $phase start"
  for ((r = 0; r < W; r++)); do
    CUDA_VISIBLE_DEVICES=${G[$r]} nice -n 5 venvs/yue2/bin/python tools/precompute_msvae_full.py --phase $phase --rank $r --world $W \
      >> logs/precompute_msvae_full.$phase.rank$r.log 2>&1 &
    sleep 20                                                         # stagger the index scans
  done
  wait; echo "$(date) phase $phase done"
done
