#!/bin/bash
# wait for msvae_full1's step-10k save + export, stop it, then resume from 10k (no sample gate, user 2026-11-05) with the
# MSVAE linguistic half (dims 32-63) weighted x3 in the flow loss (= prod1's --sem-w 3 on v5's MERT half)
cd /root/x/rwkv-music; R=out/liubai_h/msvae_full1
until grep -q "^step 10050 " $R/log.txt; do sleep 60; done
echo "$(date) 10k saved: $(ls -la $R/M_ema_s10000.pt)"
pids=$(pgrep -f "train_liubai_h.py --out $R"); kill $pids; sleep 60
if pgrep -f "train_liubai_h.py --out $R" > /dev/null; then echo "trainer still alive"; exit 1; fi
for r in 0 1 2 3; do mv logs/liubai_h_msvae_full1.rank$r.log logs/liubai_h_msvae_full1.rank$r.lr15e-5_5000to10k.log; done
echo "$(date) resuming weighted"
GPUS=0,1,2,3 bash scripts/run_liubai_h.sh $R --lora 0 --blk 8 --sem-w 3 --sem-off 32 --steps 20000 --lr-delta 1.5e-4 \
  --data /models/rwkv-music/data/lat_msvae/pilot_clm_train_clean --val-data /models/rwkv-music/data/lat_msvae/slim_val_clean --export-every 5000
