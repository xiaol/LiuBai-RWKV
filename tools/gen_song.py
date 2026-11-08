#!/usr/bin/env python3
"""Whole-song generation with a LiuBai-H song checkpoint (vae/train_liubai_song.py): text prompt (metadata + lyrics) ->
MSVAE latents block by block (0.32 s) -> 48 kHz stereo. No audio prompt.

inputs : either one song from flags (--lyrics FILE: .lrc "[mm:ss.xx] text" lines or plain text, one sung line per line;
         --title --artist --album --lang ISO-639-3 --live; --dur seconds, default last timestamp + 20 s or 30 s + 3 s/line)
         or --dev N: the first N held-out dev songs of the precomputed shards (their metadata + timed lyrics, length capped
         at --max-sec), for listening tests. --no-ts drops the timestamps (prompt without times, free cursor timing).
lyrics : romanised letters (vae.data) per line, lines joined by a space = the training cursor string. With timestamps the
         cursor jumps to a line's first letter when its start time is reached and cannot enter the next line early.
sampler: incremental (LiuBaiH.cache_init / step_hidden / step_velocity), Euler --steps, CFG --cfg on the flow velocity
         against the null prompt + NULL cursor (two rows per song); cursor advance sampled from the advance head.
         or --test FILE: a fixed song list (data/song_test16.json: id, shard, j), split over GPUs with --part i --parts n
         (one process per GPU, songs i, i+n, ...); --recon decodes those songs' true MSVAE latents instead (score ceiling).
out    : <out>/<k>_<slug>.flac (48 kHz stereo) + .txt (the prompt) + gen[.part<i>].json (settings, timed lines, advance totals)
speed  : step_velocity runs as a CUDA-graph replay from block 1 on (vae/graph_step.py; --no-graph = eager, ~2x slower)
usage  : venvs/yue2/bin/python tools/gen_song.py --ckpt out/liubai_song/s1/M_ema_s10000.pt --out out/gen/s1_10k --dev 8
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vae.gated_rwkv7 import LiuBaiH  # noqa: E402
from vae.data import romanise, encode_text  # noqa: E402
from vae.cursor_lm import windows, NULL  # noqa: E402
from vae.align_words import SPACE  # noqa: E402
from vae import song_prompt as SP  # noqa: E402
from vae.graph_step import GraphedVelocity  # noqa: E402

RATE, BLK = 25.0, 8
LRC = re.compile(r"^\s*\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$")


def read_lyrics(path):
    """-> [(t0 or None, text)]"""
    out = []
    for l in open(path, encoding="utf-8"):
        m = LRC.match(l)
        if m: out.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
        elif l.strip() and not l.strip().startswith("["): out.append((None, l.strip()))
    return [(t, x) for t, x in out if x]


def letters(lines):
    """-> ids (lines joined by SPACE), start index of each line (None if it has no letters)"""
    ids, starts = [], []
    for _, text in lines:
        li = encode_text(romanise(text))
        if not li: starts.append(None); continue
        if ids: ids.append(SPACE)
        starts.append(len(ids)); ids += li
    return ids, starts


def dev_songs(data, n, max_sec, min_lines=4):
    import glob
    out = []
    for p in sorted(glob.glob(f"{data}/r*_*[0-9].meta.jsonl")):
        for m in map(json.loads, open(p, encoding="utf-8")):
            if m.get("split") == "dev" and len(m.get("lines") or []) >= min_lines:
                dur = min(m["dur"], max_sec)
                out.append(song_entry(m, max_sec))
                if len(out) >= n: return out
    return out


def song_entry(m, max_sec):
    dur = min(m["dur"], max_sec); L = [l for l in m["lines"] if l["t0"] < dur - 2]
    return dict(meta=m, lines=[(l["t0"], l["text"]) for l in L], t1=[min(l.get("t1") or l["t0"] + 5, dur) for l in L], dur=dur, name=m["id"])


def test_songs(path, max_sec):
    out = []
    for e in json.load(open(path, encoding="utf-8")):
        with open(e["shard"] + ".meta.jsonl", encoding="utf-8") as f:
            m = json.loads(next(l for k, l in enumerate(f) if k == e["j"]))
        assert m["id"] == e["id"], (e, m["id"]); s = song_entry(m, max_sec); s.update(shard=e["shard"], j=e["j"]); out.append(s)
    return out


def prompt_ids(prompt, max_tok):
    """as in training (train_liubai_song.prompt_ids): a prompt over max_tok tokens is cut, keeping the turn end"""
    ids = SP.encode(prompt)
    if len(ids) > max_tok: tail = SP.encode("\n\n" + SP.THINK); ids = ids[:max_tok - len(tail)] + tail
    return ids


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--lyrics", default=""); ap.add_argument("--title", default=""); ap.add_argument("--artist", default="")
    ap.add_argument("--album", default=""); ap.add_argument("--lang", default=""); ap.add_argument("--live", action="store_true")
    ap.add_argument("--dur", type=float, default=0.0)
    ap.add_argument("--request", default="Make a song.", help="user request text, any language / wording (chat template User turn)")
    ap.add_argument("--dev", type=int, default=0); ap.add_argument("--data", default="/models/rwkv-music/data/lat_msvae_full/lyric")
    ap.add_argument("--max-sec", type=float, default=300.0); ap.add_argument("--no-ts", action="store_true")
    ap.add_argument("--cfg", type=float, default=3.0); ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--adv-temp", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lead", type=float, default=0.16, help="s before a line's timestamp at which the cursor may enter it")
    ap.add_argument("--test", default="", help="fixed song list json (id, shard, j), e.g. data/song_test16.json")
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--recon", action="store_true", help="with --test/--dev: decode the songs' true latents (no LM)")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args(); dev = "cuda"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    if args.test: songs = test_songs(args.test, args.max_sec)
    elif args.dev: songs = dev_songs(args.data, args.dev, args.max_sec)
    else:
        lines = read_lyrics(args.lyrics) if args.lyrics else []
        tl = [t for t, _ in lines if t is not None]
        dur = args.dur or ((max(tl) + 20) if tl else 30 + 3 * len(lines))
        meta = dict(title=args.title, artists=args.artist, album=args.album, lang=args.lang, is_live=args.live)
        songs = [dict(meta=meta, lines=lines, t1=[None] * len(lines), dur=dur, name=re.sub(r"\W+", "_", args.title or "song").strip("_")[:40] or "song")]
    for k, s in enumerate(songs): s["k"] = k
    songs = songs[args.part::args.parts]; tag = f".part{args.part}" if args.parts > 1 else ""
    for s in songs:
        timed = not args.no_ts and all(t is not None for t, _ in s["lines"]) and bool(s["lines"])
        s["timed"] = timed
        s["prompt"] = SP.build(s["meta"], [(t or 0.0, x) for t, x in s["lines"]], s["dur"], None, p_ts=0.0 if timed else 1.0, req=args.request)
        s["ids"], s["lstart"] = letters(s["lines"]); s["NB"] = int(s["dur"] * RATE) // BLK
    print(f"{len(songs)} songs, longest {max(s['dur'] for s in songs):.0f} s", flush=True)
    if args.recon: return recon(songs, out, tag, args)
    if not args.ckpt: ap.error("--ckpt is required unless --recon")

    sd = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False); ta = sd["args"]
    M = LiuBaiH(**(ta.get("tiny_cfg") or dict(lora=ta.get("lora", 0))), blk=ta.get("blk", 8))
    M.load_state_dict(sd["M"]); M = M.to(dev, torch.bfloat16).eval(); lat_m, lat_s = sd["lat_m"].to(dev), sd["lat_s"].to(dev); step = sd.get("step"); del sd
    vocab = M.emb.num_embeddings
    R = len(songs); NB = max(s["NB"] for s in songs); D = M.D; W = M.wb + M.wf
    gen = torch.Generator(device=dev).manual_seed(args.seed); ts = torch.linspace(0, 1, args.steps + 1, device=dev)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        caches = [M.cache_init(1, dev, torch.tensor([prompt_ids(s["prompt"], ta.get("max_tok", 1536))], device=dev) % vocab) for s in songs]
        nullc = M.cache_init(1, dev, torch.tensor([SP.encode(SP.null_prompt(args.request))], device=dev) % vocab)
        cache = [{k: torch.cat([c[l][k] for c in caches] + [nullc[l][k]] * R) for k in nullc[l]} for l in range(len(nullc))]
    print(f"prompts cached ({time.time() - t0:.0f}s), generating {NB} blocks x {args.steps} steps, cfg {args.cfg}", flush=True)
    z = torch.zeros(R, NB, D, device=dev); cs = np.zeros(R, np.int64); adv = np.zeros(R, np.int64)
    nullw = torch.full((R, W), NULL, dtype=torch.long, device=dev); nulla = torch.full((R,), M.max_adv, dtype=torch.long, device=dev)
    G = None
    for i in range(NB):
        tb = i * BLK / RATE
        for r, s in enumerate(songs):                                      # timestamps: jump to a line once its time has come
            if s["timed"]:
                for k, (t, _) in enumerate(s["lines"]):
                    if s["lstart"][k] is not None and t - args.lead <= tb and cs[r] < s["lstart"][k]: cs[r] = s["lstart"][k]
        Wc = torch.from_numpy(np.stack([windows(s["ids"], cs[r:r + 1], M.wb, M.wf)[0] for r, s in enumerate(songs)])).to(dev)
        WW = torch.cat([Wc, nullw]); zprev = None if i == 0 else torch.cat([z[:, i - 1], z[:, i - 1]])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = M.step_hidden(zprev, WW, cache); lg = M.adv(h[:R]).float()
        a = torch.multinomial(torch.softmax(lg / args.adv_temp, -1), 1, generator=gen)[:, 0].cpu().numpy()
        for r, s in enumerate(songs):
            lim = len(s["ids"]) + 1 - cs[r]
            if s["timed"]:                                                  # do not run into a line before its time
                nxt = [s["lstart"][k] for k, (t, _) in enumerate(s["lines"]) if s["lstart"][k] is not None and t - args.lead > tb + BLK / RATE]
                if nxt: lim = min(lim, max(0, min(nxt) - 1 - cs[r]))
            a[r] = max(0, min(int(a[r]), lim))
        A = torch.from_numpy(np.minimum(a, M.max_adv - 1)).to(dev); AA = torch.cat([A, nulla])
        x = torch.randn(R, D, device=dev, generator=gen)
        graph = not args.no_graph and zprev is not None                       # block 0 has no previous block: eager
        if graph:
            if G is None: G = GraphedVelocity(M, cache, torch.cat([x, x]), ts[0].expand(2 * R).contiguous(), WW, AA, zprev)
            else: G.load(cache, WW, AA, zprev)
        for j in range(args.steps):
            if graph: v = G(torch.cat([x, x]), ts[j].expand(2 * R))
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = M.step_velocity(torch.cat([x, x]), ts[j].expand(2 * R), WW, AA, cache, zprev).float()
            x = x + (ts[j + 1] - ts[j]) * (v[R:] + args.cfg * (v[:R] - v[R:]))
        z[:, i] = x; cs += a; adv += a
        if i % 100 == 0: print(f"block {i}/{NB} {time.time() - t0:.0f}s", flush=True)
    del M, cache, G; torch.cuda.empty_cache()
    for r, s in enumerate(songs): s["adv"] = int(adv[r]); s["letters"] = len(s["ids"])
    write_songs(songs, [z[r, :s["NB"]].reshape(1, -1, 64) * lat_s + lat_m for r, s in enumerate(songs)], out, tag, args, step, t0)


def write_songs(songs, zs, out, tag, args, step, t0):
    from vae.msvae import MSVAE
    V = MSVAE("cuda", amp=True)
    for s, zz in zip(songs, zs):
        y = V.decode(zz)[0].float().cpu().numpy().T
        stem = f"{s['k']:03d}_{s['name']}"; s["file"] = stem + ".flac"
        sf.write(out / f"{stem}.flac", np.clip(y, -1, 1), V.out_sr, subtype="PCM_16"); (out / f"{stem}.txt").write_text(s["prompt"], encoding="utf-8")
    keep = ("k", "name", "file", "dur", "timed", "letters", "adv", "prompt")
    json.dump(dict(args=vars(args), ckpt_step=step, songs=[{**{k: s[k] for k in keep if k in s}, "lang": s["meta"].get("lang"),
               "title": s["meta"].get("title"), "artists": s["meta"].get("artists"),
               "lines": [dict(t0=t, t1=t1, text=x) for (t, x), t1 in zip(s["lines"], s["t1"])]} for s in songs]),
              open(out / f"gen{tag}.json", "w"), indent=1, ensure_ascii=False)
    print(f"done {len(songs)} songs in {time.time() - t0:.0f}s -> {out}", flush=True)


def recon(songs, out, tag, args):
    from vae.train_liubai_h import load_shard, song
    t0 = time.time(); zs = []
    for s in songs:
        lat = song(load_shard(s["shard"] + ".npz"), s["j"])[0][:s["NB"] * BLK]
        zs.append(torch.from_numpy(lat.astype(np.float32))[None].cuda())
    write_songs(songs, zs, out, tag, args, "recon", t0)


if __name__ == "__main__":
    main()
