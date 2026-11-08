#!/usr/bin/env python3
"""Whole-corpus MSVAE precompute (user 2026-11-05: "precompute all songs with msvae, prepare all songs, whole-song generation
with prompt and lyrics"). Reads the clean index (tools/build_clean_index.py), writes whole-song latents + letter timings + meta.

rows   : phase "lyric" = lyric_{dev,test,train}.jsonl (timed lines), phase "audio" = audio_{dev,test,train}.jsonl rows that are
         not in the lyric phase (instrumental / no or untimed lyrics). Order = md5(id), so any prefix of shards is a uniform
         random sample (training can start on a partial cache). Rank r of W takes every W-th row.
latent : vae/msvae.py MSVAE posterior mean over the whole song, 25 Hz x 64, fp16 (bf16 autocast), 24 kHz stereo input
         (mono duplicated). Songs are padded to a multiple of 8 frames (0.32 s) like tools/precompute_latents.py.
letters: vae/align_letters.py LetterAligner at 25 Hz over the whole song (MMS emissions in 30 s pieces with 3 s margins), the
         same line table as precompute_latents.song_lines (f0, f1raw, f1capped, i0, i1), so train_liubai_h.make_crop /
         crop_letters work unchanged.
out    : <out>/<phase>/r{rank}_{k:05d}.npz (keys as tools/precompute_latents.py: lat, lat_off, ids, st, id_off, lines, line_off,
         row, dur) + r{rank}_{k:05d}.meta.jsonl, one line per song in npz order: id, split, lang, title, artists, album,
         is_live, flags, source, lyric_kind, sr, lines (cleaned text with t0 / t1 seconds) = everything a text prompt needs.
         "row" = index of the song in this rank's ordered list (meta.jsonl carries the id). Resumable: ids already written by
         this rank are skipped. meta.json in <out>.
usage  : CUDA_VISIBLE_DEVICES=g venvs/yue2/bin/python tools/precompute_msvae_full.py --phase lyric --rank r --world 4
"""
import argparse, gc, hashlib, json, os, sys, time
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import precompute_latents as PL  # noqa: E402
from vae.msvae import MSVAE  # noqa: E402
from vae.align_letters import LetterAligner  # noqa: E402

PL.RATE = MSVAE.rate                                                  # song_lines frames at 25 Hz
CLEAN = ROOT / "data/real_corpus/clean"
META_KEYS = ("title", "artists", "album", "is_live", "flags", "source", "lyric_kind", "sr", "lang", "split")


def read_audio(path):
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True); return x, sr
    except Exception:
        if not path.lower().endswith((".m4a", ".opus", ".aac", ".mp4", ".webm", ".mp3")): raise
    sys.path.insert(0, str(ROOT / "venvs/clean_extra")); import av
    with av.open(path) as c:
        s = c.streams.audio[0]; sr = s.rate; buf = []
        for fr in c.decode(s):
            a = fr.to_ndarray()
            a = a.reshape(-1, s.channels) if fr.format.is_packed else a.T
            if a.dtype.kind == "i": a = a.astype(np.float32) / np.iinfo(a.dtype).max
            buf.append(a.astype(np.float32))
    return np.concatenate(buf), sr


class Songs(torch.utils.data.Dataset):
    def __init__(s, paths, grid): s.paths, s.grid = paths, grid
    def __len__(s): return len(s.paths)
    def __getitem__(s, i):
        try:
            x, sr = read_audio(s.paths[i])
            x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]
            if sr != 24000: g = gcd(sr, 24000); x = resample_poly(x, 24000 // g, sr // g, axis=0).astype(np.float32)
            S = len(x)
            if S < 24000 * 5: return i, None, "shorter than 5 s"
            x = np.pad(x, ((0, (-S) % s.grid), (0, 0)))
            return i, np.ascontiguousarray(x.T), S / 24000
        except Exception as e:
            return i, None, repr(e)[:200]


def rid(l): return l[8:l.index('"', 8)]                                # rows start with {"id": "<id>"


class Rows:
    """this rank's rows of a phase without holding the index in memory (the old loader kept ~20 GB per rank; the host OOM-killed
    s1 2026-11-06). Rank = md5(id) mod world (stable when the clean index is rebuilt: tools/build_clean_index.py runs every 2 h),
    order = md5(id). Rows are kept as (md5, split, offset) and parsed on access from file handles opened here (a rebuilt index
    replaces the files; the open old inodes stay readable)."""
    def __init__(s, phase, splits, rank, world, skip):
        s.fh, keys, seen, excl = {}, [], set(skip), set()
        if phase == "audio":                                           # skip everything the lyric phase covers
            for sp in splits:
                with open(CLEAN / f"lyric_{sp}.jsonl", encoding="utf-8") as f: excl.update(rid(l) for l in f)
        for sp in splits:
            f = s.fh[sp] = open(CLEAN / f"{phase}_{sp}.jsonl", "rb"); off = 0
            for l in f:
                i = rid(l.decode("utf-8", "replace")); h = hashlib.md5(i.encode()).hexdigest()
                if int(h, 16) % world == rank and i not in seen and i not in excl: seen.add(i); keys.append((h, sp, off))
                off += len(l)
        keys.sort(); s.keys = keys
    def __len__(s): return len(s.keys)
    def __getitem__(s, k):
        h, sp, off = s.keys[k]; f = s.fh[sp]; f.seek(off); r = json.loads(f.readline()); r["split"] = sp; r["md5"] = h; return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("lyric", "audio"), required=True)
    ap.add_argument("--splits", default="dev,test,train")
    ap.add_argument("--out", default="/models/rwkv-music/data/lat_msvae_full")
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--shard-size", type=int, default=500); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16, help="MSVAE windows per forward")
    ap.add_argument("--max-dur", type=float, default=720.0)
    ap.add_argument("--limit", type=int, default=0, help="test: stop after this many songs")
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out) / args.phase; out.mkdir(parents=True, exist_ok=True)
    done, k0 = set(), 0                                                    # songs done by ANY rank (old runs split by position)
    for p in sorted(out.glob("r*_*.meta.jsonl")):
        with open(p, encoding="utf-8") as f: done.update(json.loads(l)["id"] for l in f)      # meta lines: id is not the first key
        if p.name.startswith(f"r{args.rank}_"): k0 = max(k0, int(p.name.split("_")[1].split(".")[0]) + 1)
    t0 = time.time(); rows = Rows(args.phase, args.splits.split(","), args.rank, args.world, done)
    print(f"rank {args.rank} phase {args.phase}: {len(rows)} rows to do ({time.time() - t0:.0f}s), {len(done)} done by all ranks", flush=True)
    if args.rank == 0:
        json.dump(dict(vars(args), vae="msvae", sr=MSVAE.sr, hop=MSVAE.hop, rate=MSVAE.rate, grid_frames=8,
                       order="md5(id), rank = md5 mod world", letters="MMS LetterAligner at 25 Hz"), open(Path(args.out) / f"meta_{args.phase}.json", "w"), indent=1)
    mine, paths, want = [], [], set()                                      # duration filter needs the row: one pass, keep offsets only
    for k in range(len(rows)):
        r = rows[k]
        if r.get("duration", 0) <= args.max_dur: mine.append(k); paths.append(r["audio_path"]); want.add(r["id"])
        if args.limit and len(mine) >= args.limit: break
    meta_all = {}
    with open(CLEAN / "index.jsonl", encoding="utf-8") as f:              # title / artists / album / flags ... by id
        for l in f:
            if rid(l) in want: r = json.loads(l); meta_all[r["id"]] = {k: r.get(k) for k in META_KEYS if k in r}
    del want, done; gc.collect()
    print(f"rank {args.rank}: {len(mine)} songs to do, shards from {k0}", flush=True)
    V = MSVAE(dev, bs=args.bs, amp=True); LA = LetterAligner(dev, sr=MSVAE.sr, rate=MSVAE.rate); grid = MSVAE.hop * 8
    dl = torch.utils.data.DataLoader(Songs(paths, grid), batch_size=None, shuffle=False, num_workers=args.workers,
                                     prefetch_factor=4, persistent_workers=False)
    buf, k, t0, n_ok, n_bad, secs = [], k0, time.time(), 0, 0, 0.0

    def flush():
        nonlocal buf, k
        if not buf: return
        cat = lambda key, dt: np.concatenate([b[key] for b in buf]).astype(dt)
        off = lambda key: np.cumsum([0] + [len(b[key]) for b in buf]).astype(np.int64)
        d = dict(lat=cat("lat", np.float16), lat_off=off("lat"), ids=cat("ids", np.uint8), st=cat("st", np.float32), id_off=off("ids"),
                 lines=np.concatenate([b["lines"] for b in buf]).astype(np.int32).reshape(-1, 5), line_off=off("lines"),
                 row=np.array([b["row"] for b in buf], np.int64), dur=np.array([b["dur"] for b in buf], np.float32))
        stem = f"r{args.rank}_{k:05d}"; tmp = out / f"{stem}.tmp.npz"; np.savez(tmp, **d); os.replace(tmp, out / f"{stem}.npz")
        with open(out / f"{stem}.meta.tmp", "w", encoding="utf-8") as f:              # written after the npz: meta = commit marker
            for b in buf: f.write(json.dumps(b["meta"], ensure_ascii=False) + "\n")
        os.replace(out / f"{stem}.meta.tmp", out / f"{stem}.meta.jsonl"); buf, k = [], k + 1

    for i, wav, dur in dl:
        r = rows[mine[i]]; n = int(r["md5"][:15], 16)
        if wav is None: n_bad += 1; print(f"skip {r['id']}: {dur}", flush=True); continue
        try:
            x = torch.as_tensor(wav).to(dev)[None]
            mu = V.encode(x)[0][0]; T = mu.shape[0]
            ln, tab = PL.song_lines(r, T)
            if len(tab):
                x16 = LA.res(x.float().mean(1)); em = PL.emissions(LA, x16)
                ids, st, _ = LA.align(em, [ln], T, x16.shape[1])[0]
                assert len(ids) == tab[-1, 4], (r["id"], len(ids), tab[-1])
            else: ids, st = [], np.zeros(0)
        except Exception as e:
            n_bad += 1; print(f"skip {r['id']}: {repr(e)[:200]}", flush=True); torch.cuda.empty_cache(); continue
        m = dict(meta_all.get(r["id"], {}), id=r["id"], split=r["split"], lang=r.get("lang") or meta_all.get(r["id"], {}).get("lang"),
                 dur=round(dur, 2), lines=r.get("lines") or [])
        buf.append(dict(lat=mu.cpu().numpy(), ids=np.asarray(ids, np.uint8), st=np.asarray(st, np.float32), lines=tab, row=n, dur=dur, meta=m))
        n_ok += 1; secs += dur
        if len(buf) >= args.shard_size: flush()
        if n_ok % 200 == 0:
            el = time.time() - t0
            print(f"{n_ok}/{len(mine)} songs, {secs / 3600:.0f} h, {secs / el:.0f}x realtime, bad {n_bad}, align fallback "
                  f"{LA.stats['fallback']}/{LA.stats['lines']} lines, mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
    flush(); print(f"done: {n_ok} songs, {n_bad} skipped, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
