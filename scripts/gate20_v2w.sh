#!/bin/bash
# Gate 0: Whisper gate20 for the oracle word-timing renderers (out/render/v2w_lyr, v2w_lyr_ptok05) at 40k steps:
# own / swap / none lyrics. Renders on GPU $1 (v2w_lyr) and $2 (ptok05) in parallel, then scores on $1.
cd /root/x/rwkv-music; PY=venvs/yue2/bin/python; G1=${1:-0}; G2=${2:-3}
run() { for L in own swap none; do CUDA_VISIBLE_DEVICES=$1 $PY vae/eval_lyric_render_roundtrip.py --vae out/vae/v2/G_s60000.pt \
          --render out/render/$2/last.pt --tag $3_$L --lyrics $L; done; }
run $G1 v2w_lyr v2w40k > logs/rt_v2w40k.log 2>&1 &
run $G2 v2w_lyr_ptok05 v2wp40k > logs/rt_v2wp40k.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=$G1 $PY tools/yue2_roundtrip_eval.py --root out/yue2_roundtrip_gate20 --manifest data/real_corpus/manifest/wangyiyun.jsonl \
  --language auto --variants v2w40k_own,v2w40k_swap,v2w40k_none,v2wp40k_own,v2wp40k_swap,v2wp40k_none,v2n_lyr0_50k,vaev2
