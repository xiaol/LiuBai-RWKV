"""Minimal stub of descript-audiotools: enough for descriptaudiocodec's dac Encoder/Decoder
to import without pulling torchaudio and a dozen other deps. Only X-Codec *encoding* is used."""
from . import ml, core  # noqa: F401
from collections import namedtuple
STFTParams = namedtuple("STFTParams", ["window_length", "hop_length", "window_type", "match_stride", "padding_type"],
                        defaults=[None, None, None, None, None])
class AudioSignal:  # placeholder; never instantiated in the encode path
    def __init__(self, *a, **k):
        raise RuntimeError("audiotools stub: AudioSignal not available")
