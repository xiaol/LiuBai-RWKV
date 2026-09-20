#!/usr/bin/env bash
# Copy cold project data to the NFS mount /models (PLAN §8.6). COPY ONLY: originals are left in place.
# After the copy is verified, the root-disk originals can be replaced by symlinks by hand:
#   rm -rf <rel> && ln -s /models/rwkv-music/<rel> <rel>
# Usage: bash scripts/copy_to_models.sh <relative path> [...]   (log: logs/copy_to_models.log)
set -u
cd "$(dirname "$0")/.."
DST_ROOT=/models/rwkv-music
for rel in "$@"; do
  src="$PWD/$rel"; dst="$DST_ROOT/$rel"
  [ -e "$src" ] || { echo "[skip] $rel missing"; continue; }
  mkdir -p "$(dirname "$dst")"
  echo "[$(date '+%F %T')] rsync $rel ($(du -sh "$src" | cut -f1))"
  if [ -d "$src" ]; then rsync -a --no-inc-recursive "$src/" "$dst/"; left=$(rsync -a -n -i "$src/" "$dst/" | grep -v '^\.d' | head -n 5)
  else rsync -a "$src" "$dst"; left=$(rsync -a -n -i "$src" "$dst" | head -n 5); fi
  if [ -n "$left" ]; then echo "[fail] verify $rel: $left"; else echo "[$(date '+%F %T')] copied+verified $rel -> $dst"; fi
done
df -h / /models | tail -n 2
