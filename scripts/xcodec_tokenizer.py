#!/usr/bin/env python3
"""X-Codec (YuE1 xcodec_mini_infer) audio tokenizer wrapper.

Produces 50 Hz codes, 1024-entry codebooks. Codebook 0 is the semantic layer
used by YuE stage-1; codebooks 0..7 (target_bw=4) are what YuE stage-2 renders.

Usage as a module:
    tok = XCodecTokenizer(device="cuda:0")
    codes = tok.encode_file("song.mp3", n_codebooks=8)   # np.int16 [8, T]
Usage as CLI (smoke test):
    python xcodec_tokenizer.py song.mp3 [song2.mp3 ...]
"""
import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
XCODEC_DIR = ROOT / "models" / "xcodec_mini_infer"
SAMPLE_RATE = 16000
FRAME_RATE = 50
# target_bandwidths [0.5,1,1.5,2,4,6] -> n_q [1,2,3,4,8,12]
BW_FOR_NQ = {1: 0.5, 2: 1, 3: 1.5, 4: 2, 8: 4, 12: 6}


@contextlib.contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_audio_16k_mono(path):
    """Decode any format libsndfile/librosa handles (mp3 included) to float32 mono 16 kHz."""
    import librosa
    wav, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


class XCodecTokenizer:
    def __init__(self, device="cuda:0", ckpt=None, config=None):
        sys.path.insert(0, str(XCODEC_DIR))
        sys.path.insert(0, str(ROOT / "scripts" / "shims"))  # audiotools stub
        sys.path.insert(0, str(XCODEC_DIR / "descriptaudiocodec"))
        from omegaconf import OmegaConf
        from models.soundstream_hubert_new import SoundStream  # noqa: E402

        cfg = OmegaConf.load(str(config or XCODEC_DIR / "final_ckpt" / "config.yaml"))
        # SoundStream hardcodes "./xcodec_mini_infer/semantic_ckpts/..." relative to cwd.
        with _chdir(XCODEC_DIR.parent):
            model = SoundStream(**cfg.generator.config)
        state = torch.load(str(ckpt or XCODEC_DIR / "final_ckpt" / "ckpt_00360000.pth"),
                           map_location="cpu", weights_only=False)
        model.load_state_dict(state["codec_model"])
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode_wave(self, wav, n_codebooks=8, chunk_seconds=None):
        """wav: float32 mono 16 kHz numpy. Returns np.int16 [n_codebooks, T] at 50 Hz.

        chunk_seconds=None encodes the whole track in one pass (HuBERT attention is
        quadratic in length; ~5 min fits on a 40 GB A100). Set e.g. 60 to bound memory.
        """
        bw = BW_FOR_NQ[n_codebooks]
        if chunk_seconds is None:
            pieces = [wav]
        else:
            step = int(chunk_seconds * SAMPLE_RATE)
            pieces = [wav[i:i + step] for i in range(0, len(wav), step)]
            if len(pieces) > 1 and len(pieces[-1]) < SAMPLE_RATE:  # merge a <1 s tail
                pieces[-2] = np.concatenate([pieces[-2], pieces[-1]]); pieces.pop()
        outs = []
        for p in pieces:
            x = torch.from_numpy(p).to(self.device)[None, None, :]
            codes = self.model.encode(x, target_bw=bw)  # [n_q, B, T]
            outs.append(codes[:, 0, :].to(torch.int16).cpu())
        return torch.cat(outs, dim=1).numpy()

    def encode_file(self, path, **kw):
        return self.encode_wave(load_audio_16k_mono(path), **kw)


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = XCodecTokenizer(device=dev)
    print(f"loaded X-Codec on {dev} in {time.time()-t0:.1f}s")
    for f in sys.argv[1:]:
        wav = load_audio_16k_mono(f)
        dur = len(wav) / SAMPLE_RATE
        torch.cuda.reset_peak_memory_stats() if dev.startswith("cuda") else None
        t0 = time.time()
        codes = tok.encode_wave(wav, n_codebooks=8)
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.startswith("cuda") else 0
        print(f"{Path(f).name}: {dur:.1f}s audio -> codes {codes.shape} ({codes.shape[1]/dur:.1f} Hz), "
              f"cb0 range [{codes[0].min()},{codes[0].max()}], {dt:.2f}s, peak {mem:.1f} GiB, "
              f"{dur/dt:.0f}x realtime")
        t0 = time.time()
        codes_c = tok.encode_wave(wav, n_codebooks=8, chunk_seconds=60)
        agree = (codes_c[0, :codes.shape[1]] == codes[0, :codes_c.shape[1]]).mean()
        print(f"   chunked(60s): shape {codes_c.shape}, {time.time()-t0:.2f}s, cb0 agreement with full pass {agree:.3f}")
