#!/usr/bin/env bash
# VAE v1 main run: 4-GPU DDP, YuE2-Vae-sized conv (ch 64, mults 1,2,4,8,16,32), lyric CTC on the encoder, 24 kHz / 12.5 Hz.
# Workers are started directly (no torchrun: its rendezvous store resolves this host's name, which has no DNS entry here) with
# env:// init on 127.0.0.1. Resumes from out/vae/v1/last.pt after a crash; a reboot needs a manual relaunch.
cd /root/x/rwkv-music
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29533 WORLD_SIZE=4 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
       CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
for attempt in $(seq 1 20); do
  pids=()
  for r in 0 1 2 3; do
    RANK=$r LOCAL_RANK=$r venvs/yue2/bin/python vae/train.py --out out/vae/v1 --ch 64 --mults 1,2,4,8,16,32 \
      --ctc-w 1.0 --lr 1.2e-4 --clip 50 --steps 200000 --bs 8 --gan-start 5000 --eval-every 5000 --workers 8 > logs/vae_v1.rank$r.log 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ $rc = 0 ] && exit 0
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done
  echo "attempt $attempt failed $(date), retry in 120 s"; sleep 120; MASTER_PORT=$((MASTER_PORT + 1))
done
