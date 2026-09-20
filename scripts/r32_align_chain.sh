#!/usr/bin/env bash
# R3.2 data chain: wait for the opus re-stream (tools/fetch_opus_shards.py) and for the line-level probe to release the GPUs,
# then forced-align the re-streamed shards on 4 GPUs. Leaves the R3.2 corpus build + launch to a human decision (unit choice).
# usage: setsid nohup bash scripts/r32_align_chain.sh > logs/r32_align_chain.log 2>&1 &
cd "$(dirname "$0")/.."
SHARDS=19-42,61-84
while ! grep -q "FETCH DONE" logs/fetch_opus_shards.log 2>/dev/null; do sleep 120; done
echo "$(date) fetch done: $(grep -c '^\[shard' logs/fetch_opus_shards.log) shards"
while [ ! -f out/stage1_line/rwkv-final.pth ] && ps -eo args | grep -v grep | grep -q "train.py --load_model"; do sleep 120; done
echo "$(date) GPUs free; aligning $SHARDS"
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g venvs/yue2/bin/python tools/lyric_align_fa.py --shards $SHARDS --part $g --nparts 4 --workers 6 --log-every 200 > logs/lyric_align_fa_r32_p$g.log 2>&1 & done
wait
echo "$(date) ALIGN CHAIN DONE"; grep -h "ALIGN DONE" logs/lyric_align_fa_r32_p*.log
