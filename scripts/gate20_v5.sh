#!/bin/bash
# VAE v5 (48 kHz) round trip on the 20 gate songs, Whisper-scored next to v2 / v4.  usage: gate20_v5.sh GPU STEP
cd /root/x/rwkv-music; PY=venvs/yue2/bin/python; export CUDA_VISIBLE_DEVICES=${1:-0}; S=${2:-80000}
$PY vae/eval_roundtrip.py --ckpt out/vae/v5/G_s$S.pt --tag vaev5_${S%000}k &
CUDA_VISIBLE_DEVICES=$(( ${1:-0} + 1 )) $PY vae/eval_roundtrip.py --ckpt out/vae/v5/G_s$S.pt --tag vaev5_${S%000}k_n03 --noise 0.3 &
wait
$PY tools/yue2_roundtrip_eval.py --root out/yue2_roundtrip_gate20 --manifest data/real_corpus/manifest/wangyiyun.jsonl --language auto \
  --variants vaev2,vaev2_n03,vaev4_60k,vaev5_${S%000}k,vaev5_${S%000}k_n03
