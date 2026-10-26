#!/bin/bash
# VAE v3 round trip on the 20 gate songs (clean + 30 % relative latent noise), Whisper-scored next to v2.
cd /root/x/rwkv-music; PY=venvs/yue2/bin/python; export CUDA_VISIBLE_DEVICES=${1:-0}
for s in 20000 30000; do
  $PY vae/eval_roundtrip.py --ckpt out/vae/v3/G_s$s.pt --tag vaev3_${s%000}k
  $PY vae/eval_roundtrip.py --ckpt out/vae/v3/G_s$s.pt --tag vaev3_${s%000}k_n03 --noise 0.3
done
$PY tools/yue2_roundtrip_eval.py --root out/yue2_roundtrip_gate20 --manifest data/real_corpus/manifest/wangyiyun.jsonl --language auto \
  --variants vaev2,vaev2_n03,vaev3_20k,vaev3_20k_n03,vaev3_30k,vaev3_30k_n03
