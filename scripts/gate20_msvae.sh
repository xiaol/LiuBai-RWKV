#!/bin/bash
# MSVAE (ASLP-lab release) round trip on the 20 gate songs, scored next to v5 / YuE2-Vae (B).  usage: gate20_msvae.sh GPU_A GPU_B
cd /root/x/rwkv-music; export HF_HUB_OFFLINE=1; PY=/root/envs/msvae-venv/bin/python
CUDA_VISIBLE_DEVICES=${1:-3} $PY tools/msvae_roundtrip.py --tag msvae &
CUDA_VISIBLE_DEVICES=${2:-2} $PY tools/msvae_roundtrip.py --tag msvae_n03 --noise 0.3 &
wait
CUDA_VISIBLE_DEVICES=${1:-3} venvs/yue2/bin/python tools/yue2_roundtrip_eval.py --root out/yue2_roundtrip_gate20 \
  --manifest data/real_corpus/manifest/wangyiyun.jsonl --language auto --variants msvae,msvae_n03,vaev5_80k
