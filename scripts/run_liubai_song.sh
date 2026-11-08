#!/bin/bash
# LiuBai-H whole-song training (vae/train_liubai_song.py), FSDP2 over the GPUs in $GPUS, one process per GPU with env:// init
# (torchrun's rendezvous cannot resolve this hostname). Resumes from <out>/dcp.
# usage: GPUS=0,1,2,3 scripts/run_liubai_h.sh OUT [train_liubai_h.py args...]
cd /root/x/rwkv-music; OUT=$1; shift; GPUS=${GPUS:-0,1,2,3}; IFS=, read -ra G <<< "$GPUS"; N=${#G[@]}
mkdir -p logs; export MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29621} WORLD_SIZE=$N NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME=$(basename $OUT)
for ((r = 0; r < N; r++)); do
  RANK=$r LOCAL_RANK=$r venvs/yue2/bin/python vae/train_liubai_song.py --out $OUT "$@" > logs/liubai_song_${NAME}.rank$r.log 2>&1 &
done
wait
echo "$(date) done"
