#!/usr/bin/env bash
# VAE v6 = v5 with live WKV (docs/VAE_PLAN.md, init bug 2026-10-31): every RWKV time-mix in v1-v5 had o = g2 = 0, a fixed point,
# so the enc / dec BiRWKV blocks, the CTC reader and the prior never used their state. Warm start from v5 (G, D, ctc, prior,
# heads); train.py --init revives the dead gates (vae/rwkv.py revive_wkv: o stays 0, so step 0 decodes exactly as v5) and the
# WKV trains from there. Same recipe as v5 minus the v4 anchor and the stage freeze: the latent is free to move (cursor LMs and
# the lat_v5 precompute are retrained on v6 anyway, they are all affected by the same bug). GAN from step 0 (D is warm).
# Data: every song on disk = the clean index (data/real_corpus/clean/audio_train.jsonl, all sources, 2.89M songs / 181k h, its
# sampling weights w; v5 had 2.06M / 134k h), ~20-25 GB RAM per rank. Val stays v5_val_clean (numbers comparable with v5; a few
# of its 760 songs may sit in the clean train split). The index is read once per launch (the corpus refresh rewrites it).
# Waits until the lat_v5 precompute is gone (GPUs 0-2). 60k steps at ~1.7 s/step ~ 28 h. usage: run_v6.sh [BS per GPU] [STEPS]
cd /root/x/rwkv-music; BS=${1:-4}; STEPS=${2:-60000}; M=data/real_corpus/manifest
while pgrep -f tools/precompute_latents.py > /dev/null; do sleep 60; done
echo "$(date) GPUs free, starting v6"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29611 WORLD_SIZE=4 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
       CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
for attempt in $(seq 1 20); do
  pids=()
  for r in 0 1 2 3; do
    RANK=$r LOCAL_RANK=$r venvs/yue2/bin/python vae/train.py --out out/vae/v6 --init out/vae/v5/last.pt \
      --sr 48000 --strides 2,2,2,4,4,5,6 --ch 64 --mults 1,1,2,4,8,16,32 --enc local --gan-start 0 \
      --ctc-w 1.0 --kl-w 1e-2 --noise-max 0.35 --mms-w 0.3 --align-w 2.0 --chroma-w 1.0 --sem 32 --lr 1e-4 --clip 50 --steps $STEPS --bs $BS \
      --eval-every 2500 --workers 8 --manifest data/real_corpus/clean/audio_train.jsonl --val-manifest $M/v5_val_clean.jsonl > logs/vae_v6.rank$r.log 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ $rc = 0 ] && exit 0
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done
  echo "attempt $attempt failed $(date), retry in 120 s"; sleep 120; MASTER_PORT=$((MASTER_PORT + 1))
done
