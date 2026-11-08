#!/bin/bash
# s1 -> YRC overlay restart (user 2026-11-06 "yes, do it"): wait for the step-5000 save, stop s1, tiny-test the trainer with
# the overlay on GPU 0, then resume s1 from 5000 (same args, --yrc defaults to the YRC index).
cd /root/x/rwkv-music; R=out/liubai_song/s1
until grep -q "^step 5050 " $R/log.txt; do sleep 60; done
echo "$(date) 5000 saved"
pids=$(pgrep -f "train_liubai_song.py --out $R"); kill $pids; sleep 60
if pgrep -f "train_liubai_song.py --out $R" > /dev/null; then echo "trainer still alive"; exit 1; fi
rm -rf /tmp/msvae_lm/song_tiny2
MASTER_ADDR=127.0.0.1 MASTER_PORT=29662 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  venvs/yue2/bin/python vae/train_liubai_song.py --out /tmp/msvae_lm/song_tiny2 --tiny --steps 50 --eval-every 50 --save-every 1000 \
  --export-every 1000 --n-val 4 --workers 2 > logs/song_tiny_yrc.log 2>&1
if ! grep -q "^EVAL step 50" logs/song_tiny_yrc.log; then echo "$(date) tiny test FAILED, s1 not restarted"; tail -5 logs/song_tiny_yrc.log; exit 1; fi
echo "$(date) tiny test ok: $(grep '^step 50' logs/song_tiny_yrc.log)"
for r in 0 1 2 3; do mv logs/liubai_song_s1.rank$r.log logs/liubai_song_s1.rank$r.mms_0to5000.log; done
echo "$(date) resuming s1 with YRC"
GPUS=0,1,2,3 bash scripts/run_liubai_song.sh $R --steps 200000
