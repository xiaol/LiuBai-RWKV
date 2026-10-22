#!/usr/bin/env bash
# hf-mirror part of the caption stack -> /models/caption (aria2, no hf CLI: its file locks hang on NFS)
cd /root/x/rwkv-music
for r in ASLP-lab/SongFormer OpenMuQ/MuQ-MuLan-large nvidia/music-flamingo-hf; do
  echo "== $r $(date)"; python3 tools/fetch_hf_aria2.py "$r" "/models/caption/$(basename $r)"
done
echo "== hf done $(date)"; du -sh /models/caption/*
