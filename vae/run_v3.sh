#!/usr/bin/env bash
# VAE v3: v2 + MSVAE-style partitioned semantic latent (docs/VAE_PLAN.md, 2026-10-23), init v2, 4-GPU DDP.
# dims 0-31 cosine-aligned to MERT-v2 L12 (w 2), dims 32-63 to chroma (w 1); v2 losses (KL 1e-2, latent noise, MMS w 0.3) kept.
# Ranks started directly with env:// on 127.0.0.1 (torchrun's rendezvous needs a resolvable hostname, this box has none).
cd /root/x/rwkv-music
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29571 WORLD_SIZE=4 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
       CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
for attempt in $(seq 1 20); do
  pids=()
  for r in 0 1 2 3; do
    RANK=$r LOCAL_RANK=$r venvs/yue2/bin/python vae/train.py --out out/vae/v3 --init out/vae/v2/last.pt --ch 64 --mults 1,2,4,8,16,32 \
      --ctc-w 1.0 --kl-w 1e-2 --noise-max 0.35 --mms-w 0.3 --align-w 2.0 --chroma-w 1.0 --lr 6e-5 --clip 50 --steps 60000 --bs 8 --gan-start 0 --eval-every 2500 \
      --workers 8 --manifest data/real_corpus/manifest/slim_train.jsonl --val-manifest data/real_corpus/manifest/slim_val.jsonl > logs/vae_v3.rank$r.log 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ $rc = 0 ] && exit 0
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done
  echo "attempt $attempt failed $(date), retry in 120 s"; sleep 120; MASTER_PORT=$((MASTER_PORT + 1))
done
