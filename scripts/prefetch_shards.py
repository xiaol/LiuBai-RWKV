#!/usr/bin/env python3
"""Prefetch suno-94k tars ahead of the two tokenizer workers so GPUs never wait on the network.

Cooperates with the running scripts/prepare_suno94k.py workers without restarting them:
  * a worker's "current" shard is the lowest shard in its range without an npz; it downloads
    that shard itself (aria2c writes <name>.tar + <name>.tar.aria2). We never touch it.
  * we download the next AHEAD shards of each range to <name>.tar.part and rename to
    <name>.tar once complete. prepare_suno94k.download() treats an existing tar with no
    .aria2 control file as done, so the worker picks it up instantly.
  * bounded disk use: at most AHEAD tars (~5.3 GB each) per range beyond the worker's current one.

Usage: python scripts/prefetch_shards.py [--ranges 1-42,43-84] [--ahead 2]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_suno94k import BASE, RAW, TOK, parse_shards  # noqa: E402


def fname(n):
    return f"suno-various-94k-{n:05d}"


def done(n):
    return (TOK / f"shard-{n:05d}.npz").exists()


def fetch(url, dest, conns=16):
    """Download url to dest via a .part temp file. Returns MB/s or None if skipped/failed."""
    if dest.exists() or Path(str(dest) + ".aria2").exists():
        return None  # already there, or the worker is downloading it itself
    part = dest.with_name(dest.name + ".part")
    t0 = time.time()
    r = subprocess.run(["aria2c", "-q", "-x", str(conns), "-s", str(conns), "--file-allocation=none",
                        "-c", "-d", str(dest.parent), "-o", part.name, url])
    if r.returncode != 0 or not part.exists() or Path(str(part) + ".aria2").exists():
        return None
    # last-moment check: if the worker started its own download meanwhile, do not clobber it
    if dest.exists() or Path(str(dest) + ".aria2").exists():
        part.unlink(missing_ok=True)
        return None
    part.rename(dest)
    return dest.stat().st_size / max(time.time() - t0, 1e-3) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", default="1-42,43-84", help="one worker range per comma-separated item")
    ap.add_argument("--ahead", type=int, default=2)
    ap.add_argument("--poll", type=int, default=30, help="seconds between checks")
    args = ap.parse_args()
    ranges = [parse_shards(r) for r in args.ranges.split(",")]
    RAW.mkdir(parents=True, exist_ok=True)
    while True:
        pending = False
        for shards in ranges:
            todo = [n for n in shards if not done(n)]
            if not todo:
                continue
            pending = True
            for n in todo[1:1 + args.ahead]:  # todo[0] is the worker's current shard: leave it alone
                tar = RAW / f"{fname(n)}.tar"
                if tar.exists():
                    continue
                mbps = fetch(f"{BASE}/{fname(n)}.tar", tar)
                if mbps is not None:
                    fetch(f"{BASE}/{fname(n)}.json", RAW / f"{fname(n)}.json")
                    print(f"[prefetch] shard-{n:05d} tar ready, {tar.stat().st_size/2**30:.2f} GiB at {mbps:.1f} MB/s",
                          flush=True)
                    if mbps < 5:
                        print(f"[prefetch] WARNING slow edge ({mbps:.1f} MB/s); check /etc/hosts pin for "
                              f"cas-bridge.xethub.hf.co (see PLAN.md §0)", flush=True)
        if not pending:
            print("[prefetch] all shards tokenized; exiting", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
