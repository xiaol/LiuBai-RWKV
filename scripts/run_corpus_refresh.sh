#!/bin/bash
# Incremental CPU corpus refresh (2026-10-29, all nice 19, no GPU), loops while the downloaders keep writing:
#   1. ingest --append: ncm_vip (every <lang>/ dir, new langs picked up each pass), wangyiyun, kugou -> manifest/v2/<src>.jsonl
#   2. tools/audio_scan.py --resume on every v2 manifest -> clean/scan/
#   3. tools/build_clean_index.py into clean_build/, then files moved into clean/ (readers never see a half-written index)
# usage: scripts/run_corpus_refresh.sh [procs 48] [sleep between passes, s 7200]   stop: touch logs/clean/refresh.stop
cd /root/x/rwkv-music; export PYTHONPATH=venvs/clean_extra; P=${1:-48}; SLEEP=${2:-7200}
M=data/real_corpus/manifest/v2; C=data/real_corpus/clean; B=data/real_corpus/clean_build; mkdir -p $C/scan logs/clean
while [ ! -e logs/clean/refresh.stop ]; do
  echo "$(date '+%F %T') pass start"
  nice -n 19 python3 tools/ingest_lrc_corpus.py --append --procs $P --out $M/ncm_vip.jsonl \
    $(for d in /models/musicdl/ncm_vip/*/; do echo --src ${d%/}; done) 2>> logs/clean/ingest_v2_ncm_vip.err
  for s in wangyiyun kugou; do
    nice -n 19 python3 tools/ingest_lrc_corpus.py --append --procs $P --src /models/musicdl/$s --out $M/$s.jsonl 2>> logs/clean/ingest_v2_$s.err
  done
  echo "$(date '+%F %T') ingest: $(wc -l $M/*.jsonl | tr '\n' ' ')"
  for f in $M/*.jsonl; do
    nice -n 19 python3 tools/audio_scan.py $f $C/scan/$(basename $f) --procs $P --resume || echo "scan $f rc=$?"
  done
  rm -rf $B; mkdir -p $B
  if nice -n 19 python3 tools/build_clean_index.py --src $M --out $B --procs 32 $(for f in $C/scan/*.jsonl; do echo --scan $f; done); then
    mv $B/* $C/; echo "$(date '+%F %T') index updated: $(python3 -c "import json;s=json.load(open('$C/stats.json'));print({k:s[k] for k in s if k.startswith(('lyric_ok','audio_ok','rows'))})")"
  else echo "$(date '+%F %T') index FAILED, clean/ untouched"; fi
  sleep $SLEEP
done
echo "$(date '+%F %T') stopped (refresh.stop)"
