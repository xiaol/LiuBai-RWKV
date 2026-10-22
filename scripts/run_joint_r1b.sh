#!/usr/bin/env bash
# R1-b: joint-teacher fine-tune of R1-a on real NetEase songs (prepare_real_yue2.py shards, original flac -> online MERT), GPU 3.
# Waits until 2 training shards exist; shard 0 is the held-out test set (gate: tools/yue2_roundtrip.py --manifest on 20 of its songs).
cd /root/x/rwkv-music
D=/models/rwkv-music/real_corpus/wangyiyun
until [ "$(ls $D/shard-0000[1-9].jsonl 2>/dev/null | wc -l)" -ge 2 ]; do sleep 60; done
echo "start $(date)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  venvs/yue2/bin/python scripts/train_joint_online.py --out out/joint/r1b --head-init out/joint/v2/head_best.pt --lora-init out/joint/v2/lora_best.pt \
  --latent-shards "$D/shard-*.latents.npz" --exclude-shards 0 --max-sec 420 --online-p 0.7 --wins-per-song 2 --rescan-every 500 \
  --steps 12000 --lr-head 3e-5 --lr-lora 1e-5 --eval-every 500 --hold-jsonl $D/shard-00000.jsonl
