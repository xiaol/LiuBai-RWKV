#!/usr/bin/env bash
# VAE v2: robust latent for generators (docs/VAE_PLAN.md "Phase A result and the v2 retrain"), init v1, 4-GPU DDP.
# KL 1e-2, latent noise s2 ~ U(0, 0.35) on 80 % of decodes (+30 % time blur), MMS_FA CTC on the decoded audio (w 0.3).
# Ranks started directly with env:// on 127.0.0.1 (torchrun's rendezvous needs a resolvable hostname, this box has none).
cd /root/x/rwkv-music
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29561 WORLD_SIZE=4 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
       CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
for attempt in $(seq 1 20); do
  pids=()
  for r in 0 1 2 3; do
    RANK=$r LOCAL_RANK=$r venvs/yue2/bin/python vae/train.py --out out/vae/v2 --init out/vae/v1/last.pt --ch 64 --mults 1,2,4,8,16,32 \
      --ctc-w 1.0 --kl-w 1e-2 --noise-max 0.35 --mms-w 0.3 --lr 6e-5 --clip 50 --steps 60000 --bs 8 --gan-start 0 --eval-every 2500 \
      --workers 8 > logs/vae_v2.rank$r.log 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ $rc = 0 ] && exit 0
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done
  echo "attempt $attempt failed $(date), retry in 120 s"; sleep 120; MASTER_PORT=$((MASTER_PORT + 1))
done
