#!/usr/bin/env python3
"""Precompute whole-song VAE latents + letter timings for the 3B LM (docs/VAE_PLAN.md, 3B port), so its training step reads
latents from disk instead of running VAE + MERT + MMS on the fly (train_cursor_lm.py does that per crop).

latents : VAE.encode with the bounded-lookahead MERT branch (same kw as train_cursor_lm.py: blk 4, keep 2, win 3 s, ahead 0.64 s).
          MERT features are computed once over the whole song (its windows sit on an 8-frame grid from the song start: crops for
          training should start on that grid); the acoustic path runs in --chunk s pieces with --margin s of context on each
          side (its influence decays to ~0 within ~0.6 s) and keeps the centre. mu only (no sampling), fp16, 12.5 Hz x 64.
letters : vae/align_letters.py LetterAligner over the whole song (MMS emissions in 30 s pieces with 3 s margins), one start
          frame per letter; per line (f0, f1 raw, f1 capped, i0, i1) = frames + its slice of the song's letter ids, so a
          crop's (ids, starts, mask) can be rebuilt exactly as LetterAligner builds them (complete lines only).
out     : <out>/r{rank}_{k:05d}.npz shards of --shard-size songs: lat [sumT,64] f16, lat_off, ids u8, st f32, id_off, lines i32
          [sumL,5], line_off, row (manifest line number), dur; meta.json. Resumable (rows already in this rank's shards are skipped).
usage   : precompute_latents.py --rank 0 --world 3 (CUDA_VISIBLE_DEVICES picks the GPU)
"""
import argparse, json, os, sys, time
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.model import load_vae  # noqa: E402
from vae.data import romanise, encode_text  # noqa: E402
from vae.align_letters import LetterAligner  # noqa: E402
from vae.align_words import SPACE  # noqa: E402

RATE = 12.5


class Songs(torch.utils.data.Dataset):
    def __init__(s, rows, sr, grid):
        s.rows, s.sr, s.grid = rows, sr, grid
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        n, r = s.rows[i]
        try:
            x, sr = sf.read(r["audio_path"], dtype="float32", always_2d=True)
            x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]
            if sr != s.sr: g = gcd(sr, s.sr); x = resample_poly(x, s.sr // g, sr // g, axis=0).astype(np.float32)
            S = len(x); x = np.pad(x, ((0, (-S) % s.grid), (0, 0)))                 # whole 8-frame windows
            return n, np.ascontiguousarray(x.T), S / s.sr
        except Exception as e:
            return n, None, repr(e)


def song_lines(r, T):
    """manifest lines -> LetterAligner input [(f0, ids, f1)] (absolute frames) + table rows (f0, f1raw, f1c, i0, i1) of the lines it
    keeps, with i0 / i1 = the slice of its concatenated ids (separator SPACE between lines), mirroring LetterAligner.align"""
    ln = []
    for l in r.get("lines") or []:
        ids = encode_text(romanise(l["text"]))
        f0, f1 = int(round(l["t0"] * RATE)), int(round(l["t1"] * RATE))
        if ids and 0 <= f0 <= f1: ln.append((f0, ids, f1))                          # bad LRC: t1 < t0 / absurd t0 overflowed int32
    cap = [(f0, ids, min(f1, f0 + int(RATE * (2.0 + 0.2 * len(ids)))), f1) for f0, ids, f1 in ln]
    full = sorted((l for l in cap if l[0] >= 0 and l[2] <= T), key=lambda l: l[0])
    tab, pos = [], 0
    for f0, ids, f1c, f1 in full:
        n = sum(len(w) for w in LetterAligner.words(ids)); nw = len(LetterAligner.words(ids))
        if pos: pos += 1                                                             # LetterAligner: "if ids: ids.append(SPACE)"
        tab.append((f0, min(f1, 2**31 - 1), f1c, pos, pos + n + max(nw - 1, 0))); pos += n + max(nw - 1, 0)  # raw f1 unused downstream; bad LRC t1 overflowed int32
    return ln, np.asarray(tab, np.int32).reshape(-1, 5)


@torch.no_grad()
def emissions(LA, x16, chunk=480000, margin=48000):
    """x16 [1,S16] -> [1, S16 // 320, labels] (MMS 50 Hz), 30 s pieces with 3 s context each side"""
    S = x16.shape[1]; n_out = S // 320; out = []
    for a in range(0, S, chunk):
        lo, hi = max(0, a - margin), min(S, a + chunk + margin)
        em = LA.emissions(x16[:, lo:hi]); o = (a - lo) // 320; k = min(chunk, S - a) // 320
        em = em[:, o:o + k]
        if em.shape[1] < k: em = torch.cat([em, em[:, -1:].expand(-1, k - em.shape[1], -1)], 1)
        out.append(em)
    em = torch.cat(out, 1)[:, :n_out]
    return torch.cat([em, em[:, -1:].expand(-1, n_out - em.shape[1], -1)], 1) if em.shape[1] < n_out else em


@torch.no_grad()
def encode_song(V, wav, ekw, chunk_f, margin_f):
    """wav [1,2,S] (S multiple of 8 hops) -> mu [T,64]"""
    hop = V.hop; T = wav.shape[-1] // hop
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = V.feats(wav, T, ekw); mus = []
        for a in range(0, T, chunk_f):
            lo, hi = max(0, a - margin_f), min(T, a + chunk_f + margin_f)
            mu = V.encode(wav[..., lo * hop:hi * hop], feats=feats[:, lo:hi])[0].float()
            assert mu.shape[1] == hi - lo, (mu.shape, lo, hi)
            mus.append(mu[0, a - lo:a - lo + min(chunk_f, T - a)])
    return torch.cat(mus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/real_corpus/manifest/pilot_clm_train_clean.jsonl"))
    ap.add_argument("--vae", default=str(ROOT / "out/vae/v5/G_s80000.pt"))
    ap.add_argument("--out", default="/models/rwkv-music/data/lat_v5/pilot_clm_train_clean")
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--shard-size", type=int, default=500); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-dur", type=float, default=20.0); ap.add_argument("--max-dur", type=float, default=600.0)
    ap.add_argument("--chunk", type=float, default=40.0); ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="test: stop after this many songs")
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    V = load_vae(args.vae, dev); ekw = dict(blk=4, keep=2, win=3.0, ahead=0.64)
    LA = LetterAligner(dev, sr=V.sr); grid = V.hop * 8
    if args.rank == 0:
        json.dump(dict(vars(args), ekw=ekw, sr=V.sr, hop=V.hop, rate=RATE, grid_frames=8), open(out / "meta.json", "w"), indent=1)
    done, k0 = set(), 0
    for p in sorted(out.glob(f"r{args.rank}_*.npz")):
        done.update(np.load(p)["row"].tolist()); k0 = max(k0, int(p.stem.split("_")[1]) + 1)
    rows = []
    for n, l in enumerate(open(args.manifest, encoding="utf-8")):
        if n % args.world != args.rank or n in done: continue
        r = json.loads(l)
        if args.min_dur <= r.get("duration", 0) <= args.max_dur: rows.append((n, r))
    if args.limit: rows = rows[:args.limit]
    print(f"rank {args.rank}: {len(rows)} songs to do, {len(done)} done, shards from {k0}", flush=True)
    dl = torch.utils.data.DataLoader(Songs(rows, V.sr, grid), batch_size=None, shuffle=False, num_workers=args.workers,
                                     prefetch_factor=2, persistent_workers=False)
    byrow = dict(rows); buf, k, t0, n_ok, n_bad, secs = [], k0, time.time(), 0, 0, 0.0
    chunk_f, margin_f = int(args.chunk * RATE) // 8 * 8, int(args.margin * RATE) // 8 * 8

    def flush():
        nonlocal buf, k
        if not buf: return
        cat = lambda key, dt: np.concatenate([b[key] for b in buf]).astype(dt)
        off = lambda key: np.cumsum([0] + [len(b[key]) for b in buf]).astype(np.int64)
        d = dict(lat=cat("lat", np.float16), lat_off=off("lat"), ids=cat("ids", np.uint8), st=cat("st", np.float32), id_off=off("ids"),
                 lines=np.concatenate([b["lines"] for b in buf]).astype(np.int32).reshape(-1, 5), line_off=off("lines"),
                 row=np.array([b["row"] for b in buf], np.int64), dur=np.array([b["dur"] for b in buf], np.float32))
        tmp = out / f"r{args.rank}_{k:05d}.tmp.npz"; np.savez(tmp, **d); os.replace(tmp, out / f"r{args.rank}_{k:05d}.npz")
        buf, k = [], k + 1

    for n, wav, dur in dl:
        if wav is None: n_bad += 1; print(f"skip row {n}: {dur}", flush=True); continue
        x = torch.as_tensor(wav).to(dev)[None]
        mu = encode_song(V, x, ekw, chunk_f, margin_f); T = mu.shape[0]
        ln, tab = song_lines(byrow[n], T)
        if len(tab):
            x16 = LA.res(x.float().mean(1)); em = emissions(LA, x16)
            ids, st, _ = LA.align(em, [ln], T, x16.shape[1])[0]
            assert len(ids) == (tab[-1, 4] if len(tab) else 0), (n, len(ids), tab[-1])
        else: ids, st = [], np.zeros(0)
        buf.append(dict(lat=mu.cpu().numpy(), ids=np.asarray(ids, np.uint8), st=np.asarray(st, np.float32), lines=tab, row=n, dur=dur))
        n_ok += 1; secs += dur
        if len(buf) >= args.shard_size: flush()
        if n_ok % 50 == 0:
            el = time.time() - t0
            print(f"{n_ok}/{len(rows)} songs, {el / n_ok:.2f} s/song, {secs / el:.0f}x realtime, bad {n_bad}, "
                  f"align fallback {LA.stats['fallback']}/{LA.stats['lines']} lines, mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
    flush(); print(f"done: {n_ok} songs, {n_bad} unreadable, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
