#!/usr/bin/env bash
# Singing ASR encoder (vae/train_sing_asr.py), DDP over the GPUs in $1 (default 0,3), out dir $2 (default out/asr/s1).
# Ranks started directly with env:// on 127.0.0.1 (torchrun's rendezvous needs DNS this host lacks); resumes from last.pt.
cd /root/x/rwkv-music; GPUS=${1:-0,3}; OUT=${2:-out/asr/s1}; shift 2 2>/dev/null
IFS=, read -ra G <<< "$GPUS"; N=${#G[@]}
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29561 WORLD_SIZE=$N NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
       CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
mkdir -p "$OUT"
for attempt in $(seq 1 20); do
  pids=()
  for r in $(seq 0 $((N - 1))); do
    RANK=$r LOCAL_RANK=$r venvs/yue2/bin/python -u vae/train_sing_asr.py --out "$OUT" "$@" > logs/sing_asr_$(basename $OUT).rank$r.log 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ $rc = 0 ] && exit 0
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done
  echo "attempt $attempt failed $(date), retry in 120 s"; sleep 120; MASTER_PORT=$((MASTER_PORT + 1))
done
