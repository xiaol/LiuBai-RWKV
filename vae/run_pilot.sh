#!/usr/bin/env bash
# VAE pilot variant on one GPU, auto-resume from out/vae/<name>/last.pt. usage: run_pilot.sh <gpu> <name> [train.py args...]
cd /root/x/rwkv-music; G=$1; N=$2; shift 2
for attempt in 1 2 3 4 5; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    venvs/yue2/bin/python vae/train.py --out out/vae/$N --steps 40000 --bs 8 "$@" && exit 0
  echo "attempt $attempt failed $(date), retry in 60 s"; sleep 60
done
