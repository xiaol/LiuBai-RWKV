#!/usr/bin/env bash
# Real-corpus prep worker <k> of <W> on GPU <g>, following the live manifest. usage: run_prep_real.sh <gpu> <k> <W> [head]
cd /root/x/rwkv-music
HEAD=${4:-out/joint/v2/head_best.pt}
while true; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    venvs/yue2/bin/python scripts/prepare_real_yue2.py --manifest data/real_corpus/manifest/wangyiyun.jsonl --head "$HEAD" \
    --worker $2 --n-workers $3 --follow
  echo "worker exited $? at $(date), restarting in 60 s"; sleep 60
done
