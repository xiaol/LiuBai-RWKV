#!/usr/bin/env python3
"""Per-song effective bandwidth for the 48 kHz VAE data mix (VAE v5).

Reads three 2 s excerpts (25 / 50 / 75 % of the song) at the native rate, Welch power spectrum of the mono mix, 250 Hz bins.
  ref     mean level of 1-4 kHz (dB)
  cut_hz  highest bin whose level is within --drop dB of ref, scanning down from Nyquist (lossy encoders brick-wall at
          ~16 / 19 / 20 kHz, so a lossy file shows a cliff there; the tools/audit_corpus.py -40 dB-below-peak test measured
          only the natural roll-off)
  hb16    level of 16-20 kHz minus ref (dB), hb20 level of 20 kHz-Nyquist minus ref
Output: one JSON line per row {audio_path, sr, cut_hz, hb16, hb20} (rows that fail to read are skipped), then
--mix writes a copy of the manifest with a sampling weight "w" per row (vae/data.Crops honours it).
usage: bandwidth_scan.py scan MANIFEST OUT.jsonl [--n 0] [--procs 64]
       bandwidth_scan.py mix MANIFEST SCAN.jsonl OUT_MANIFEST [--w-full 3] [--full-hz 19000] [--w-mid 1] [--mid-hz 16000] [--w-low 0.5]
"""
import argparse, json, random, sys
from multiprocessing import Pool
import numpy as np, soundfile as sf
from scipy.signal import welch

DROP = 60.0


def scan_one(path):
    try:
        info = sf.info(path); sr = info.samplerate; n = int(2.0 * sr)
        if info.frames < 4 * n: return None
        P = 0
        for q in (0.25, 0.5, 0.75):
            a0 = int(info.frames * q) - n // 2
            x, _ = sf.read(path, start=a0, stop=a0 + n, dtype="float32", always_2d=True)
            f, p = welch(x.mean(1), sr, nperseg=4096); P = P + p
        P = P / 3 + 1e-20
        edges = np.arange(0, sr / 2 + 1, 250.0); idx = np.digitize(f, edges) - 1
        lvl = np.array([10 * np.log10(P[idx == i].mean()) if (idx == i).any() else -200.0 for i in range(len(edges) - 1)])
        ctr = edges[:-1] + 125; ref = lvl[(ctr >= 1000) & (ctr < 4000)].mean()
        ok = np.nonzero(lvl >= ref - DROP)[0]; cut = float(edges[ok[-1] + 1]) if len(ok) else 0.0
        band = lambda lo, hi: float(lvl[(ctr >= lo) & (ctr < hi)].mean() - ref) if ((ctr >= lo) & (ctr < hi)).any() else None
        return dict(audio_path=path, sr=sr, cut_hz=cut, hb16=band(16000, 20000), hb20=band(20000, sr / 2))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["scan", "mix"]); ap.add_argument("manifest"); ap.add_argument("a"); ap.add_argument("b", nargs="?")
    ap.add_argument("--n", type=int, default=0, help="scan a random subset of n rows (0 = all)"); ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--w-full", type=float, default=3.0); ap.add_argument("--full-hz", type=float, default=19000)
    ap.add_argument("--w-mid", type=float, default=1.0); ap.add_argument("--mid-hz", type=float, default=16000); ap.add_argument("--w-low", type=float, default=0.5)
    a = ap.parse_args()
    if a.cmd == "scan":
        paths = [json.loads(l)["audio_path"] for l in open(a.manifest, encoding="utf-8")]
        if a.n: random.Random(0).shuffle(paths); paths = paths[:a.n]
        done = 0
        with Pool(a.procs) as pool, open(a.a, "w") as fo:
            for r in pool.imap_unordered(scan_one, paths, chunksize=16):
                done += 1
                if r: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                if done % 20000 == 0: print(done, "/", len(paths), flush=True)
        return
    bw = {}
    for l in open(a.a, encoding="utf-8"):
        r = json.loads(l); bw[r["audio_path"]] = r["cut_hz"]
    cnt = dict(full=0, mid=0, low=0, unscanned=0)
    with open(a.manifest, encoding="utf-8") as fi, open(a.b, "w") as fo:
        for l in fi:
            r = json.loads(l); c = bw.get(r["audio_path"])
            if c is None: cnt["unscanned"] += 1; continue          # unreadable or not scanned: left out
            k = "full" if c >= a.full_hz else "mid" if c >= a.mid_hz else "low"; cnt[k] += 1
            r["w"] = dict(full=a.w_full, mid=a.w_mid, low=a.w_low)[k]; fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = sum(cnt[k] * dict(full=a.w_full, mid=a.w_mid, low=a.w_low)[k] for k in ("full", "mid", "low"))
    print(cnt, {k: f"{cnt[k] * dict(full=a.w_full, mid=a.w_mid, low=a.w_low)[k] / tot:.1%} of crops" for k in ("full", "mid", "low")})


if __name__ == "__main__":
    main()
