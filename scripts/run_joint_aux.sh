#!/usr/bin/env bash
# Tokenizer + NAR-LoRA joint training with the audio-domain loss (Mothersuperior joint_v6 port) on the NetEase real corpus.
# usage: run_joint_aux.sh <gpu> <out> <head-arch r1|v4> <head-init> <lora-init> [aux_w=4.0] [steps=6000]
cd /root/x/rwkv-music; D=/models/rwkv-music/real_corpus/wangyiyun
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  venvs/yue2/bin/python scripts/train_joint_online.py --out $2 --head-arch $3 --head-init $4 --lora-init $5 \
  --latent-shards "$D/shard-*.latents.npz" --exclude-shards 0 --max-sec 420 --online-p 0.8 --win 512 --rescan-every 2000 \
  --steps ${7:-6000} --eval-every 500 --save-every 1000 --aux-w ${6:-4.0} --hold 8 --hold-jsonl $D/shard-00000.jsonl \
  --lr-head 3e-5 --lr-lora 1e-5
