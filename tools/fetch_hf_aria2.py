#!/usr/bin/env python3
"""Lock-free model fetch: list a repo via hf-mirror's API, download every file with aria2c (the hf CLI hangs on NFS file locks).
usage: fetch_hf_aria2.py <repo> <dest_dir> [--include substr,...]"""
import json, os, subprocess, sys, urllib.request

repo, dest = sys.argv[1], sys.argv[2]
inc = sys.argv[sys.argv.index("--include") + 1].split(",") if "--include" in sys.argv else None
EP = "https://hf-mirror.com"
info = json.load(urllib.request.urlopen(urllib.request.Request(f"{EP}/api/models/{repo}?blobs=true", headers={"User-Agent": "curl/8.5"})))
for s in info["siblings"]:
    p = s["rfilename"]; size = s.get("size")
    if inc and not any(k in p for k in inc): continue
    dst = os.path.join(dest, p)
    if os.path.exists(dst) and (size is None or os.path.getsize(dst) == size) and not os.path.exists(dst + ".aria2"): continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for _ in range(5):
        r = subprocess.run(["aria2c", "-x8", "-s8", "-c", "--console-log-level=warn", "--summary-interval=0", "--allow-overwrite=true",
                            "-d", os.path.dirname(dst), "-o", os.path.basename(dst), f"{EP}/{repo}/resolve/main/{p}"])
        if r.returncode == 0: break
    print(p, size, "ok" if r.returncode == 0 else "FAILED", flush=True)
print("done", repo, flush=True)
