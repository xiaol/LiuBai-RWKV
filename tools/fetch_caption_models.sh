#!/usr/bin/env bash
# Caption-stack weights for the real-song corpus -> /models/caption (NFS). Qwen3-Omni from ModelScope, rest via hf-mirror.
set -u
cd /models/caption
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0
for r in ASLP-lab/SongFormer OpenMuQ/MuQ-MuLan-large nvidia/music-flamingo-hf; do
  echo "== $r $(date)"; hf download "$r" --local-dir "$(basename $r)" --max-workers 8 || echo "FAILED $r"
done
echo "== Qwen3-Omni-30B-A3B-Captioner $(date)"
python3 - <<'PY'
import json, subprocess, urllib.request, os
repo = "Qwen/Qwen3-Omni-30B-A3B-Captioner"; out = "Qwen3-Omni-30B-A3B-Captioner"; os.makedirs(out, exist_ok=True)
files = json.load(urllib.request.urlopen(f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"))["Data"]["Files"]
for f in files:
    if f["Type"] != "blob": continue
    p = f["Path"]; dst = os.path.join(out, p)
    if os.path.exists(dst) and os.path.getsize(dst) == f["Size"]: continue
    os.makedirs(os.path.dirname(dst) or out, exist_ok=True)
    url = f"https://modelscope.cn/models/{repo}/resolve/master/{p}"
    subprocess.run(["aria2c", "-x8", "-s8", "-c", "--console-log-level=warn", "--summary-interval=0", "-d", os.path.dirname(dst) or out, "-o", os.path.basename(dst), url])
    print(p, f["Size"], flush=True)
PY
echo "== done $(date)"; du -sh /models/caption/*
