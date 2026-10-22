#!/usr/bin/env bash
# Follow the live NetEase download dir: every 20 min add newly completed songs (Info.json present) to the real-corpus manifest.
cd /root/x/rwkv-music
while true; do
  python3 tools/ingest_lrc_corpus.py --src /models/musicdl/wangyiyun --out data/real_corpus/manifest/wangyiyun.jsonl --append 2>>logs/ingest_wangyiyun.err
  echo "$(date '+%F %T') total $(wc -l < data/real_corpus/manifest/wangyiyun.jsonl)"; sleep 1200
done
