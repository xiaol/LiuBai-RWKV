#!/usr/bin/env bash
# R3.1 audio gate: render the section-mode generations in out/gen/r31_s2_sec through YuE2 NAR + R1-a LoRA + VAE and score them
# against the real shard-2 songs together with the G3 (rwkv_final_s32) and R1-a round-trip (lora_r1_jv2_s32) variants.
# usage: setsid nohup bash scripts/r31_render_eval.sh > logs/r31_render_eval_driver.log 2>&1 &
cd "$(dirname "$0")/.."
GEN=${GEN:-out/gen/r31_s2_sec}; TAG=${TAG:-rwkv_r31}; GPU=${GPU:-1}
while [ "$(find "$GEN" -name semantic.npy | wc -l)" -lt 20 ]; do sleep 60; done
while ps -eo args | grep -v grep | grep -q "python scripts/rwkv_generate.py --ckpt out/stage1_sec"; do sleep 30; done
echo "$(date) all 20 generated; rendering"
CUDA_VISIBLE_DEVICES=$GPU venvs/yue2/bin/python tools/yue2_render_tokens.py --gen "$GEN" --tag "$TAG" --eval-root out/yue2_roundtrip_s2 > "logs/render_${TAG}_s2.log" 2>&1
echo "$(date) rendered; scoring"
CUDA_VISIBLE_DEVICES=$GPU venvs/yue2/bin/python tools/yue2_roundtrip_eval.py --root out/yue2_roundtrip_s2 --variants "${TAG}_s32,rwkv_final_s32,lora_r1_jv2_s32" > "logs/eval_${TAG}_s2.log" 2>&1
echo "$(date) DONE"
