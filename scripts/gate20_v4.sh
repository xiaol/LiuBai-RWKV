#!/bin/bash
# VAE v4 clean round trip on the 20 gate songs, Whisper-scored next to v2 / v3.  usage: gate20_v4.sh GPU STEP
cd /root/x/rwkv-music; PY=venvs/yue2/bin/python; export CUDA_VISIBLE_DEVICES=${1:-1}; S=${2:-5000}
$PY vae/eval_roundtrip.py --ckpt out/vae/v4/G_s$S.pt --tag vaev4_${S%000}k
$PY tools/yue2_roundtrip_eval.py --root out/yue2_roundtrip_gate20 --manifest data/real_corpus/manifest/wangyiyun.jsonl --language auto \
  --variants vaev2,vaev3_30k,vaev4_${S%000}k
