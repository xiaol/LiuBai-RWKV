#!/usr/bin/env bash
# VAE v5 = v4 at 48 kHz (docs/VAE_PLAN.md, 2026-10-26), 4-GPU DDP. Same 12.5 Hz x 64 latent: one extra outer stride-2 stage
# (strides 2,2,2,4,4,5,6 = 3840), init from v4 (inner stages shifted by one, first conv / output conv reused, new stage fresh,
# D scales by n_fft + a fresh 4096 one), inner stages frozen for the first 3k steps, GAN from 2k.
# Latent anchored to frozen v4's mu on the 24 kHz-resampled crop (normalised MSE, w 1, constant) so cursor LMs trained on v4
# latents stay usable with the 48 kHz decoder. Semantic half from bounded-lookahead MERT (--enc local, = cursor_v4b), so the
# decoder trains on the latents the LM produces. Data: full slim train (2.06M songs) with cleaned lyrics,
# unweighted (tools/bandwidth_scan.py on 2k random songs: 89 % have content to >= 19 kHz, so no bandwidth weighting needed).
# usage: run_v5.sh [BS per GPU] [STEPS]
cd /root/x/rwkv-music; BS=${1:-4}; STEPS=${2:-80000}; M=data/real_corpus/manifest
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29591 WORLD_SIZE=4 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
       CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
for attempt in $(seq 1 20); do
  pids=()
  for r in 0 1 2 3; do
    RANK=$r LOCAL_RANK=$r venvs/yue2/bin/python vae/train.py --out out/vae/v5 --init out/vae/v4/last.pt --teacher out/vae/v4/G_s60000.pt \
      --sr 48000 --strides 2,2,2,4,4,5,6 --ch 64 --mults 1,1,2,4,8,16,32 --enc local --anchor-w 1.0 --freeze-inner 3000 --gan-start 2000 \
      --ctc-w 1.0 --kl-w 1e-2 --noise-max 0.35 --mms-w 0.3 --align-w 2.0 --chroma-w 1.0 --sem 32 --lr 1e-4 --clip 50 --steps $STEPS --bs $BS \
      --eval-every 2500 --workers 8 --manifest $M/v5_train_clean.jsonl --val-manifest $M/v5_val_clean.jsonl > logs/vae_v5.rank$r.log 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ $rc = 0 ] && exit 0
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done
  echo "attempt $attempt failed $(date), retry in 120 s"; sleep 120; MASTER_PORT=$((MASTER_PORT + 1))
done
