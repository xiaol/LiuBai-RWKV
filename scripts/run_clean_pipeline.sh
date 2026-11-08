#!/bin/bash
# CPU data cleanup, 2026-10-27 (all nice 19, no GPU): waits for scripts/run_ingest_v2.sh, then
#   1. tools/audio_scan.py over every v2 manifest row (bandwidth cliff, clipping, mono, silence, unreadable) -> data/real_corpus/clean/scan/
#   2. tools/build_clean_index.py -> data/real_corpus/clean/{index,lyric_*,audio_*}.jsonl + stats.json
cd /root/x/rwkv-music; export PYTHONPATH=venvs/clean_extra; P=${1:-48}; C=data/real_corpus/clean; mkdir -p $C/scan
while pgrep -f run_ingest_v2.sh > /dev/null; do sleep 120; done
echo "$(date '+%F %T') ingest done: $(wc -l data/real_corpus/manifest/v2/*.jsonl | tail -1)"
for f in data/real_corpus/manifest/v2/*.jsonl; do
  s=$(basename $f .jsonl); echo "$(date '+%F %T') scan $s"
  nice -n 19 python3 tools/audio_scan.py $f $C/scan/$s.jsonl --procs $P --resume || echo "scan $s rc=$?"
done
echo "$(date '+%F %T') index"
nice -n 19 python3 tools/build_clean_index.py --out $C --procs 32 $(for f in $C/scan/*.jsonl; do echo --scan $f; done)
echo "$(date '+%F %T') done rc=$?"
