"""Token layout for R3 stage-1 in YuE2 semantic-token space (PLAN §8.3 R3, §8.6).

RWKV world vocab (65,536) + the §2.3 control ids, then the 32,768 YuE2 codes:

    65536  <EOD>          65537 <SOA>   65538 <EOA>   65539 <xcodec> (cb0 marker, unused here)
    65540  <yue2codec>    codec-type marker for this layout
    65541  <sos>          start of a lyric section (R3.1 section-interleaved layout, scripts/build_yue2_sec_binidx.py):
                          header EOD (SOS section-text SOA YUE2CODEC codes EOA)* 0
    65544 .. 98311        YuE2 semantic codes 0 .. 32767   (= YUE2_BASE + code)
    vocab padded to 98,816 (multiple of 512; 98,312 -> 98,816)

Ids < 65,544 are identical to the cb0 layout, so world-token / control rows of the cb0 stage-1 checkpoint
(out/stage1/step-20000.pth) transfer unchanged; rows >= 65,544 (cb0 codes) are re-initialised (resize_vocab_yue2.py).
"""
WORLD_VOCAB = 65536
EOD, SOA, EOA, XCODEC, YUE2CODEC, SOS = 65536, 65537, 65538, 65539, 65540, 65541
YUE2_BASE = 65544
N_CODES = 32768
VOCAB_SIZE = 98816
N_CONTROL = 5          # EOD SOA YUE2CODEC ... EOA, plus the RWKV document separator 0
assert YUE2_BASE + N_CODES <= VOCAB_SIZE and VOCAB_SIZE % 512 == 0
