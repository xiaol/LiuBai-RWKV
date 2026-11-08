# LiuBai whole-song generation (RWKV-7 5.1B on MSVAE latents)

**Play in the browser:** https://raw.githack.com/xiaol/LiuBai-RWKV/main/samples/song_s1_20k_20261109/player.html

Whole songs from a text prompt only (metadata + timed lyrics), CFG 3, 24 flow steps, MSVAE decode 48 kHz stereo. 16 held-out songs, not cherry-picked. CER = Qwen3-ASR-1.7B per-line letter CER vs the given lyrics; recon ceiling = real songs' own latents decoded, same scorer (scores only). Only generated audio is published. Model: s1 = 3B RWKV-7 + full-rank NAR deltas (5.1B), init msvae_full1 20k, 20k whole-song steps.

| run | CER mean | median | thirds | dB | silent |
|---|---|---|---|---|---|
| s1 20k | 1.178 | 1.064 | 0.95 / 1.23 / 1.38 | -17.8 | 0.06 |
| recon ceiling | 0.511 | 0.536 | 0.48 / 0.53 / 0.52 | -14.6 | 0.01 |

| song | lang | dur | s1 20k | recon ceiling |
|---|---|---|---|---|
| 000 | cmn | 259 s | [2.30](000_s1_20k.mp3) | 0.13 |
| 001 | cmn | 244 s | [0.96](001_s1_20k.mp3) | 0.46 |
| 002 | cmn | 199 s | [0.47](002_s1_20k.mp3) | 0.29 |
| 003 | cmn | 180 s | [1.15](003_s1_20k.mp3) | 0.68 |
| 004 | cmn | 211 s | [1.10](004_s1_20k.mp3) | 0.28 |
| 005 | cmn | 202 s | [1.03](005_s1_20k.mp3) | 0.31 |
| 006 | cmn | 256 s | [1.20](006_s1_20k.mp3) | 0.12 |
| 007 | eng | 280 s | [1.43](007_s1_20k.mp3) | 0.67 |
| 008 | eng | 172 s | [1.01](008_s1_20k.mp3) | 0.30 |
| 009 | eng | 239 s | [0.92](009_s1_20k.mp3) | 0.45 |
| 010 | eng | 222 s | [0.83](010_s1_20k.mp3) | 0.74 |
| 011 | jpn | 281 s | [0.86](011_s1_20k.mp3) | 0.72 |
| 012 | jpn | 245 s | [1.06](012_s1_20k.mp3) | 0.85 |
| 013 | kor | 219 s | [2.17](013_s1_20k.mp3) | 0.69 |
| 014 | kor | 227 s | [1.28](014_s1_20k.mp3) | 0.88 |
| 015 | yue | 199 s | [1.06](015_s1_20k.mp3) | 0.61 |
