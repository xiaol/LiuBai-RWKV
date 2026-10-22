#!/usr/bin/env bash
# Lyric renderers on the robust VAE v2 latent (docs/VAE_PLAN.md). Tokenizer = v2 FSQ 8,8,8,8,6,5 (122,880 codes), final 60k.
# Flow runs in the per-channel NORMALISED latent (v2 channel std ~7.7; the first v2 attempt on the raw latent, out/render/v2_lyr0
# and v2_lyrmms, started at flow loss 65 and sat at cer 0.84 / mel 12 dB after 5k steps: stopped 2026-10-21).
# GPU 0: flow + lyrics in the sequence only; GPU 3: + MMS audio judge (w 0.3 after 10k steps).
# Evaluate with Whisper (tools/yue2_roundtrip_eval.py), not MMS: v2's decoder was trained against MMS.
cd /root/x/rwkv-music
V2=out/vae/v2/G_s60000.pt; T=out/tok/v2_ctc1_92k/last.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 nohup venvs/yue2/bin/python vae/train_lyric_renderer.py --vae $V2 --tok $T --out out/render/v2n_lyr0 --lyr-w 0 --steps 100000 > logs/render_v2n_lyr0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup venvs/yue2/bin/python vae/train_lyric_renderer.py --vae $V2 --tok $T --out out/render/v2n_lyrmms --judge mms --lyr-w 0.3 --lyr-start 10000 --steps 100000 > logs/render_v2n_lyrmms.log 2>&1 &
