#!/bin/bash
# s1 -> RWKV g1 chat-template prompt (user 2026-11-07): tiny train + generate test with the new vae/song_prompt.py, then
# resume s1 from its last dcp (step 5000) on 4 GPUs. s1 must already be stopped.
cd /root/x/rwkv-music
if pgrep -f "train_liubai_song.py --out out/liubai_song/s1" > /dev/null; then echo "s1 still running"; exit 1; fi
rm -rf /tmp/msvae_lm/song_tiny3 /tmp/msvae_lm/gen_tiny3
MASTER_ADDR=127.0.0.1 MASTER_PORT=29663 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  venvs/yue2/bin/python vae/train_liubai_song.py --out /tmp/msvae_lm/song_tiny3 --tiny --steps 50 --eval-every 50 --save-every 50 \
  --export-every 50 --n-val 4 --workers 2 > logs/song_tiny_chat.log 2>&1
grep -q "^EVAL step 50" logs/song_tiny_chat.log || { echo "$(date) tiny train FAILED"; tail -5 logs/song_tiny_chat.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 venvs/yue2/bin/python tools/gen_song.py --ckpt /tmp/msvae_lm/song_tiny3/M_ema_s50.pt --out /tmp/msvae_lm/gen_tiny3 \
  --dev 1 --max-sec 10 --steps 2 --request "帮我写一首歌。" > logs/song_tiny_chat_gen.log 2>&1
grep -q "^done" logs/song_tiny_chat_gen.log || { echo "$(date) tiny gen FAILED"; tail -5 logs/song_tiny_chat_gen.log; exit 1; }
echo "$(date) tiny ok: $(grep '^EVAL step 50' logs/song_tiny_chat.log)"; cat /tmp/msvae_lm/gen_tiny3/000_*.txt; echo
for r in 0 1 2 3; do mv logs/liubai_song_s1.rank$r.log logs/liubai_song_s1.rank$r.oldprompt_2500to5200.log 2>/dev/null; done
echo "$(date) resuming s1 with the chat template"
GPUS=0,1,2,3 bash scripts/run_liubai_song.sh out/liubai_song/s1 --steps 200000
