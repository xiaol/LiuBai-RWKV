#!/bin/bash
# Re-ingest every /models/musicdl source with the 2026-10-27 ingest (lyric_kind, untimed lyrics, m4a, [offset:], source id
# prefixes) into data/real_corpus/manifest/v2/<source>.jsonl. Leaves the v1 manifests (and v5's v5_*_clean.jsonl) untouched.
cd /root/x/rwkv-music; export PYTHONPATH=venvs/clean_extra; P=${1:-48}
for s in youtube kugou soda jamendo wangyiyun; do
  echo "$(date '+%F %T') start $s"
  nice -n 19 python3 tools/ingest_lrc_corpus.py --src /models/musicdl/$s --out data/real_corpus/manifest/v2/$s.jsonl --procs $P \
    2> logs/clean/ingest_v2_$s.err
  echo "$(date '+%F %T') done $s rows $(wc -l < data/real_corpus/manifest/v2/$s.jsonl)"
done
