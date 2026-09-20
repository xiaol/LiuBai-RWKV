#!/usr/bin/env python3
"""G1 (PLAN §8.4): re-stream suno-94k shards into YuE2 space.

Per shard: download tar (aria2c, resumable, same CDN pin as prepare_suno94k.py) -> per song:
  48 kHz stereo -> YuE2-Vae.encode -> latents fp16 [T,64]        (head-independent; R2 targets, teacher term)
  24 kHz mono   -> MERT-v2-FullSong L12/16/20/23 @25 Hz -> inverse tokenizer (--head) -> tokens int16 [T]
Outputs: data/yue2_corpus/suno94k/shard-NNNNN.tokens.npz (id -> int16 [T]), shard-NNNNN.latents.npz (id -> fp16 [T,64]),
         shard-NNNNN.jsonl (meta incl. head name, n_frames), tar deleted unless --keep-tar.
Resumable per shard (existing jsonl = skip). Run inside venvs/yue2 with one GPU (~11 GiB).
usage: prepare_suno94k_yue2.py --shards 1-10 --head out/joint/v1/head_best.pt [--gpu 0] [--workers 8] [--keep-tar]
"""
import argparse, io, json, os, subprocess, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_inverse_tokenizer import InverseTokenizer, predict as head_predict  # noqa: E402
REPO = "webshart/suno-various-94k"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/original"
RAW = ROOT / "data/raw/suno94k"; OUT = ROOT / "data/yue2_corpus/suno94k"; YUE2 = ROOT / "models/yue2"
LAYERS = [12, 16, 20, 23]


def parse_shards(spec):
    out = []
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out


def download(url, dest, retries=10):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        if dest.exists() and not (dest.parent / (dest.name + ".aria2")).exists() and dest.stat().st_size > 1e6: return dest
        r = subprocess.run(["aria2c", "-x8", "-s8", "-c", "--console-log-level=warn", "--summary-interval=0", "-d", str(dest.parent), "-o", dest.name, url])
        if r.returncode == 0 and dest.exists(): return dest
        time.sleep(10 * (i + 1))
    raise RuntimeError(f"download failed: {url}")


def iter_pairs(tar_path):
    with tarfile.open(tar_path, "r") as t:
        pend = {}
        for m in t:
            if not m.isfile(): continue
            cid, ext = os.path.splitext(os.path.basename(m.name)); slot = pend.setdefault(cid, {})
            slot[ext] = t.extractfile(m).read()
            if ".json" in slot and any(e in slot for e in (".mp3", ".wav", ".flac")):
                audio = slot.get(".mp3") or slot.get(".wav") or slot.get(".flac")
                yield cid, json.loads(slot[".json"]), audio; del pend[cid]


def decode(audio_bytes):
    a, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if a.shape[1] == 1: a = np.repeat(a, 2, 1)
    g = gcd(sr, 48000); st48 = resample_poly(a, 48000 // g, sr // g, axis=0).astype(np.float32) if sr != 48000 else a
    g = gcd(sr, 24000); m24 = resample_poly(a.mean(1), 24000 // g, sr // g).astype(np.float32)
    return st48, m24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True); ap.add_argument("--head", required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--keep-tar", action="store_true"); ap.add_argument("--vae", default="YuE2-Vae")
    ap.add_argument("--max-sec", type=float, default=480); ap.add_argument("--keep-shards", default="0,1,2", help="never delete these tars (test material)")
    ap.add_argument("--audio-shards", default="", help="shards whose 24 kHz mono audio is kept as <tag>.audio/<id>.opus for online-MERT training (PLAN §8.6)")
    ap.add_argument("--audio-level", type=float, default=0.7, help="libsndfile opus compression_level: 0.7 ~ 80 kbps, ~1.9 MB per 3-min song, ~80 %% exact-token agreement with the original")
    args = ap.parse_args(); os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu)); dev = "cuda"; OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModel, AutoFeatureExtractor
    from yue2.modeling_vae import YuE2VAE
    proc = AutoFeatureExtractor.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True)
    mert = AutoModel.from_pretrained(str(YUE2 / "MERT-v2-FullSong"), trust_remote_code=True).to(dev).eval()
    vae = YuE2VAE.from_pretrained(str(YUE2 / args.vae), decoder_only=False, device=dev, local_files_only=True)
    ck = torch.load(args.head, map_location=dev); head = InverseTokenizer(**ck["cfg"]).to(dev); head.load_state_dict(ck["model"]); head.eval()
    assert ck["layers"] == LAYERS; head_name = str(Path(args.head).resolve().relative_to(ROOT))
    print(f"head {head_name} (step {ck.get('step')}), shards {args.shards}", flush=True)

    @torch.inference_mode()
    def encode_song(st48, m24):
        CH = 24000 * 30; chunks = [m24[s:s + CH] for s in range(0, len(m24), CH)]; chunks = [c for c in chunks if len(c) >= 24000]
        full = [c for c in chunks if len(c) == CH]; tail = [c for c in chunks if len(c) < CH]; feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for group in ([full] if full else []) + [[c] for c in tail]:
                inp = {k: v.to(dev) for k, v in proc(group, sampling_rate=24000, return_tensors="pt").items()}
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
        toks = head_predict(head, M[:, :n], dev).astype(np.int16)
        return toks, Z[:n]

    audio_shards = set(parse_shards(args.audio_shards)) if args.audio_shards else set()
    def write_opus(path, m24): sf.write(path, m24, 24000, format="OGG", subtype="OPUS", compression_level=args.audio_level)
    for n in parse_shards(args.shards):
        tag = f"shard-{n:05d}"; out_meta = OUT / f"{tag}.jsonl"; adir = OUT / f"{tag}.audio"; audio_futs = []
        if n in audio_shards: adir.mkdir(exist_ok=True)
        if out_meta.exists(): print(f"[{tag}] exists, skipping", flush=True); continue
        fname = f"suno-various-94k-{n:05d}"; tar_path = download(f"{BASE}/{fname}.tar", RAW / f"{fname}.tar")
        t0 = time.time(); toks_by_id = {}; lats_by_id = {}; recs = []; pool = ThreadPoolExecutor(max_workers=args.workers); window = []
        def consume(cid, meta, fut):
            rec = {"id": cid, "shard": n, "head": head_name}
            for k in ("title", "duration", "caption", "lyrics", "creator", "model_name", "major_model_version", "created_at", "play_count", "upvote_count"):
                if k in meta: rec[k] = meta[k]
            try:
                st48, m24 = fut.result()
                if len(st48) / 48000 > args.max_sec: raise ValueError("too long")
                toks, Z = encode_song(st48, m24); toks_by_id[cid] = toks; lats_by_id[cid] = Z; rec["n_frames"] = int(len(toks))
                if n in audio_shards: audio_futs.append(pool.submit(write_opus, adir / f"{cid}.opus", m24)); rec["opus"] = True
            except Exception as e:
                rec["error"] = str(e)[:200]
            recs.append(rec)
            if len(recs) % 100 == 0:
                sec = sum(r.get("n_frames", 0) for r in recs) / 25; el = time.time() - t0
                print(f"  [{tag}] {len(recs)} clips, {sec / 3600:.2f} h audio, {el / 60:.1f} min, {sec / max(el, 1):.0f}x realtime", flush=True)
        for cid, meta, audio in iter_pairs(tar_path):
            window.append((cid, meta, pool.submit(decode, audio)))
            if len(window) < args.workers * 2: continue
            consume(*window.pop(0))
        for item in window: consume(*item)
        n_audio_fail = sum(1 for f in audio_futs if f.exception() is not None)
        pool.shutdown()
        n_fail = sum(1 for r in recs if "error" in r)
        if n_fail > 0.1 * max(len(recs), 1):
            print(f"[{tag}] too many failures ({n_fail}/{len(recs)}); nothing written, shard will be retried", flush=True); continue
        np.savez(OUT / f"{tag}.tokens.npz", **toks_by_id); np.savez(OUT / f"{tag}.latents.npz", **lats_by_id)
        with open(out_meta, "w") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dt = time.time() - t0; sec = sum(r.get("n_frames", 0) for r in recs) / 25
        print(f"[{tag}] {len(toks_by_id)} ok, {n_fail} failed, {sec / 3600:.1f} h, {dt / 60:.1f} min ({sec / max(dt, 1):.0f}x), "
              f"tokens {(OUT / f'{tag}.tokens.npz').stat().st_size / 2**20:.0f} MiB latents {(OUT / f'{tag}.latents.npz').stat().st_size / 2**20:.0f} MiB"
              + (f" opus {len(audio_futs) - n_audio_fail} files ({n_audio_fail} failed) {sum(f.stat().st_size for f in adir.iterdir()) / 2**30:.1f} GiB" if n in audio_shards else ""), flush=True)
        if not args.keep_tar and n not in parse_shards(args.keep_shards): tar_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
