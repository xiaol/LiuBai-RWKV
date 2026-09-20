#!/usr/bin/env bash
# Move the big project directories (out/, data/, models/) to the NFS disk /models/rwkv-music and replace them with symlinks.
# Per directory: rsync -a (no --delete, existing backups stay) → checksum verification pass → rm -rf original → ln -s.
# Nothing is deleted unless the checksum pass reports zero differing files. Log: logs/move_to_models.log
# usage: setsid nohup bash scripts/move_to_models.sh > logs/move_to_models.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
DST=/models/rwkv-music
for d in out data models; do
  [ -L "$d" ] && { echo "$(date) $d is already a symlink -> $(readlink "$d"); skip"; continue; }
  [ -d "$d" ] || { echo "$(date) $d missing; skip"; continue; }
  echo "$(date) === $d: $(du -sh "$d" | cut -f1) → $DST/$d"
  mkdir -p "$DST/$d"
  rsync -a --info=progress2,stats1 "$d/" "$DST/$d/" 2>&1 | tail -n 12
  rc=${PIPESTATUS[0]}; [ "$rc" -ne 0 ] && { echo "$(date) rsync rc=$rc for $d; NOT deleting"; continue; }
  echo "$(date) verifying $d by checksum"
  diff_n=$(rsync -a -n -c --out-format='%n' "$d/" "$DST/$d/" | grep -v '/$' | tee "logs/move_verify_$d.txt" | wc -l)
  if [ "$diff_n" -eq 0 ]; then
    echo "$(date) $d verified identical; replacing with symlink"
    mv "$d" "$d.moving" && ln -s "$DST/$d" "$d" && rm -rf "$d.moving"
    echo "$(date) $d done: $(ls -ld "$d" | awk '{print $NF}'), root free $(df -h / | awk 'NR==2{print $4}')"
  else
    echo "$(date) $d: $diff_n files differ after rsync (see logs/move_verify_$d.txt); NOT deleting"
  fi
done
echo "$(date) ALL DONE; root: $(df -h / | tail -1)"
