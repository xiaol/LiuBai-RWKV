#!/usr/bin/env python3
"""Per-song audio quality numbers for the clean index (tools/build_clean_index.py), CPU only.

Three 4 s excerpts (25 / 50 / 75 % of the song) at the native rate; libsndfile for flac / mp3 / wav / ogg, PyAV for m4a / opus
(PYTHONPATH=venvs/clean_extra).
  cut_hz, hb16, hb20  bandwidth as tools/bandwidth_scan.py: highest 250 Hz bin within 60 dB of the 1-4 kHz level (a lossy
                      encode shows a cliff at ~16 / 19 / 20 kHz), 16-20 kHz and 20 kHz-Nyquist level relative to 1-4 kHz (dB)
  rms_db, peak        level of the excerpts (dBFS), sample peak
  clip                fraction of samples with |x| >= 0.999
  silence             fraction of 50 ms frames below -60 dBFS
  lr_corr             L/R correlation (>= 0.999: mono in a stereo container)
Output: one JSON line per readable row {audio_path, sr, ...}; unreadable rows -> {audio_path, err}.
usage: audio_scan.py MANIFEST OUT.jsonl [--procs 64] [--resume]
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np, soundfile as sf
from scipy.signal import welch

SEG = 4.0; DROP = 60.0


def excerpts(path):
    """-> (sr, [x [n, ch] float32] at 25 / 50 / 75 %)"""
    try:
        info = sf.info(path); sr = info.samplerate; n = int(SEG * sr)
        if info.frames < 4 * n: raise ValueError("short")
        out = []
        for q in (0.25, 0.5, 0.75):
            a0 = int(info.frames * q) - n // 2
            out.append(sf.read(path, start=a0, stop=a0 + n, dtype="float32", always_2d=True)[0])
        return sr, out
    except ValueError: raise
    except Exception:
        if not path.lower().endswith((".m4a", ".opus", ".aac", ".mp4", ".webm")): raise
    import av
    with av.open(path) as c:
        s = c.streams.audio[0]; sr = s.rate
        dur = float(s.duration * s.time_base) if s.duration else c.duration / 1e6
        if dur < 4 * SEG: raise ValueError("short")
        out = []
        for q in (0.25, 0.5, 0.75):
            t = dur * q - SEG / 2; c.seek(int(t / s.time_base), stream=s, backward=True, any_frame=False); buf = []; got = 0
            for fr in c.decode(s):
                if fr.time is not None and fr.time + fr.samples / sr < t: continue
                a = fr.to_ndarray(); a = a.reshape(-1, s.channels) if fr.format.is_packed else a.T
                if fr.time is not None and fr.time < t: a = a[int((t - fr.time) * sr):]
                buf.append(a.astype(np.float32)); got += len(a)
                if got >= SEG * sr: break
            out.append(np.concatenate(buf)[:int(SEG * sr)])
        return sr, out


def scan_one(path):
    try:
        sr, xs = excerpts(path); x = np.concatenate(xs)
        m = x.mean(1); P = 0
        for e in xs:
            f, p = welch(e.mean(1), sr, nperseg=4096); P = P + p
        P = P / len(xs) + 1e-20
        edges = np.arange(0, sr / 2 + 1, 250.0); idx = np.digitize(f, edges) - 1
        lvl = np.array([10 * np.log10(P[idx == i].mean()) if (idx == i).any() else -200.0 for i in range(len(edges) - 1)])
        ctr = edges[:-1] + 125; ref = lvl[(ctr >= 1000) & (ctr < 4000)].mean()
        ok = np.nonzero(lvl >= ref - DROP)[0]; cut = float(edges[ok[-1] + 1]) if len(ok) else 0.0
        band = lambda lo, hi: round(float(lvl[(ctr >= lo) & (ctr < hi)].mean() - ref), 1) if ((ctr >= lo) & (ctr < hi)).any() else None
        hop = int(0.05 * sr); fr = m[: len(m) // hop * hop].reshape(-1, hop)
        lr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if x.shape[1] > 1 and x[:, 0].std() > 1e-6 and x[:, 1].std() > 1e-6 else 1.0
        return dict(audio_path=path, sr=sr, cut_hz=cut, hb16=band(16000, 20000), hb20=band(20000, sr / 2),
                    rms_db=round(float(20 * np.log10(np.sqrt((m ** 2).mean()) + 1e-9)), 1), peak=round(float(np.abs(x).max()), 4),
                    clip=round(float((np.abs(x) >= 0.999).mean()), 6), silence=round(float((10 * np.log10((fr ** 2).mean(1) + 1e-12) < -60).mean()), 3),
                    lr_corr=round(lr, 4))
    except Exception as e:
        return dict(audio_path=path, err=f"{type(e).__name__}: {str(e)[:80]}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("out"); ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    paths = [json.loads(l)["audio_path"] for l in open(a.manifest, encoding="utf-8")]
    if a.n: paths = paths[:a.n]
    if a.resume and os.path.exists(a.out):
        done = {json.loads(l)["audio_path"] for l in open(a.out, encoding="utf-8")}; paths = [p for p in paths if p not in done]
    print("to scan", len(paths), flush=True); k = 0
    with Pool(a.procs) as pool, open(a.out, "a" if a.resume else "w", encoding="utf-8") as fo:
        for r in pool.imap_unordered(scan_one, paths, chunksize=16):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n"); k += 1
            if k % 50000 == 0: print(k, "/", len(paths), flush=True); fo.flush()


if __name__ == "__main__":
    main()
