#!/usr/bin/env python3
"""Download a HF repo (model or dataset) via hf-mirror.com.
Usage: hf_download.py <repo_id> <dest> [model|dataset] [allow_pattern ...]"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
rtype = sys.argv[3] if len(sys.argv) > 3 else "model"
allow = sys.argv[4:] or None
p = snapshot_download(repo_id=repo, repo_type=rtype, local_dir=dest, allow_patterns=allow, max_workers=8)
print("DONE", p)
