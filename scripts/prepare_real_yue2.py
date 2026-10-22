#!/usr/bin/env python3
"""Real-song corpus -> YuE2 space (the G1 recipe of prepare_suno94k_yue2.py, input = tools/ingest_lrc_corpus.py manifest).

Shards: manifest rows in file order, SHARD_SIZE songs each (the manifest is append-only, so shard n is always the same songs);
only full shards are processed unless --flush-partial. Worker k of W takes shards with n % W == k.
Per song: original audio (flac/mp3, any rate) -> 48 kHz stereo -> YuE2-Vae latents fp16 [T,64]
                                              -> 24 kHz mono -> MERT-v2 L12/16/20/23 @25 Hz -> inverse tokenizer --head -> int16 [T]
Outputs (OUT=/models/rwkv-music/real_corpus/<tag>): shard-NNNNN.{jsonl,tokens.npz,latents.npz}; the jsonl rows carry the manifest
fields (id, title, duration, lyrics, lines, meta, audio_path) + head + n_frames, so train_joint_online.py reads the original audio
(no opus copy) and build_yue2_sec_binidx.py reads lyrics. Songs > --max-sec are skipped (recorded as error). Resumable per shard.
usage: prepare_real_yue2.py --manifest data/real_corpus/manifest/wangyiyun.jsonl --head out/joint/v2/head_best.pt
         [--tag wangyiyun] [--worker 0 --n-workers 4] [--follow]   (run in venvs/yue2, one GPU per worker, ~11 GiB)
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_inverse_tokenizer import InverseTokenizer, predict as head_predict  # noqa: E402
YUE2 = ROOT / "models/yue2"; LAYERS = [12, 16, 20, 23]; SHARD_SIZE = 1000


def load_audio(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    if a.shape[1] == 1: a = np.repeat(a, 2, 1)
    a = a[:, :2]
    g = gcd(sr, 48000); st48 = resample_poly(a, 48000 // g, sr // g, axis=0).astype(np.float32) if sr != 48000 else a
    g = gcd(sr, 24000); m24 = resample_poly(a.mean(1), 24000 // g, sr // g).astype(np.float32)
    return st48, m24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--head", required=True); ap.add_argument("--tag", default="wangyiyun")
    ap.add_argument("--out-root", default="/models/rwkv-music/real_corpus")
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--io-workers", type=int, default=8); ap.add_argument("--max-sec", type=float, default=720)
    ap.add_argument("--follow", action="store_true", help="keep polling the manifest for new full shards (live download)")
    ap.add_argument("--flush-partial", action="store_true", help="also process the last, incomplete shard")
    ap.add_argument("--shards", default="", help="only these shard numbers (comma list), e.g. for a quick test")
    args = ap.parse_args(); dev = "cuda"
    out = Path(args.out_root) / args.tag; out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModel, AutoFeatureExtractor
    from yue2.modeling_vae import YuE2VAE
    proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
    mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()
    vae = YuE2VAE.from_pretrained(str(YUE2 / "YuE2-Vae"), decoder_only=False, device=dev, local_files_only=True)
    ck = torch.load(args.head, map_location=dev); head = InverseTokenizer(**ck["cfg"]).to(dev); head.load_state_dict(ck["model"]); head.eval()
    assert ck["layers"] == LAYERS; head_name = str(Path(args.head))

    @torch.inference_mode()
    def encode_song(st48, m24):
        CH = 24000 * 30; chunks = [m24[s:s + CH] for s in range(0, len(m24), CH)]; chunks = [c for c in chunks if len(c) >= 24000]
        full = [c for c in chunks if len(c) == CH]; tail = [c for c in chunks if len(c) < CH]; feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(full), 8):                                   # batch 8 x 30 s: bounded memory for long songs
                grp = full[i:i + 8]
                inp = {k: v.to(dev) for k, v in proc(grp, sampling_rate=24000, return_tensors="pt").items()}
                hs = mert(**inp, output_hidden_states=True).hidden_states
                feats.append(torch.stack([hs[l].reshape(-1, 1024) for l in LAYERS]))
            for c in tail:
                inp = {k: v.to(dev) for k, v in proc([c], sampling_rate=24000, return_tensors="pt").items()}
                hs = mert(**inp, output_hidden_states=True).hidden_states
                feats.append(torch.stack([hs[l].reshape(-1, 1024) for l in LAYERS]))
        Hm = torch.cat(feats, 1).float(); T25 = int(round(len(m24) / 24000 * 25))
        M = torch.nn.functional.interpolate(Hm.transpose(1, 2), size=T25, mode="linear", align_corners=False).transpose(1, 2).half().cpu().numpy()
        zs = []
        for s in range(0, len(st48), 48000 * 60):
            seg = st48[s:s + 48000 * 60]
            if len(seg) < 1920: break
            zs.append(vae.encode(torch.tensor(seg.T[None], device=dev))[0].T.half().cpu())
        Z = torch.cat(zs, 0).numpy(); n = min(M.shape[1], len(Z))
        return head_predict(head, M[:, :n], dev).astype(np.int16), Z[:n]

    only = {int(x) for x in args.shards.split(",") if x}
    pool = ThreadPoolExecutor(args.io_workers)
    while True:
        rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
        n_full = len(rows) // SHARD_SIZE; n_sh = n_full + (1 if args.flush_partial and len(rows) % SHARD_SIZE else 0)
        todo = [n for n in range(n_sh) if n % args.n_workers == args.worker and (not only or n in only)
                and not (out / f"shard-{n:05d}.jsonl").exists()]
        for n in todo:
            tag = f"shard-{n:05d}"; songs = rows[n * SHARD_SIZE:(n + 1) * SHARD_SIZE]; t0 = time.time()
            toks, lats, recs = {}, {}, []
            futs = [pool.submit(load_audio, s["audio_path"]) for s in songs[:args.io_workers * 2]]; nxt = len(futs)
            for i, s in enumerate(songs):
                rec = dict(s, shard=n, head=head_name); rec.pop("caption", None)
                try:
                    if s["duration"] > args.max_sec: raise ValueError(f"too long ({s['duration']:.0f} s)")
                    st48, m24 = futs[i].result()
                    t, z = encode_song(st48, m24); toks[s["id"]] = t; lats[s["id"]] = z; rec["n_frames"] = int(len(t))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                futs[i] = None
                if nxt < len(songs): futs.append(pool.submit(load_audio, songs[nxt]["audio_path"])); nxt += 1
                recs.append(rec)
                if len(recs) % 100 == 0:
                    sec = sum(r.get("n_frames", 0) for r in recs) / 25; el = time.time() - t0
                    print(f"  [{tag}] {len(recs)} songs, {sec / 3600:.2f} h, {el / 60:.1f} min, {sec / max(el, 1):.0f}x realtime", flush=True)
            n_fail = sum(1 for r in recs if "error" in r)
            np.savez(out / f"{tag}.tokens.npz", **toks); np.savez(out / f"{tag}.latents.npz", **lats)
            tmp = out / f"{tag}.jsonl.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.rename(out / f"{tag}.jsonl")                                   # jsonl last = shard complete
            sec = sum(r.get("n_frames", 0) for r in recs) / 25; dt = time.time() - t0
            print(f"[{tag}] {len(toks)} ok, {n_fail} failed, {sec / 3600:.1f} h in {dt / 60:.1f} min ({sec / max(dt, 1):.0f}x)", flush=True)
        if not args.follow: break
        time.sleep(300)


if __name__ == "__main__":
    main()
