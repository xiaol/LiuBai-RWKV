#!/bin/bash
# E1 + Qwen3-ASR stream (vae/asr_stream.py): e1_gated_s0_wkv config (v4 latents via encode_local, cleaned lyrics, --sem-w 3,
# 20k steps, seed 0) but the LM also predicts the bounded-lookahead Qwen3-ASR encoder states through a lyric-CTC-learned
# projection (vae/probe_asr_stream.py --lin W -> lin<W>.pt); audio is decoded from the 64 VAE dims only.
# Gate: the 96 causal crops at CFG 1 / 3 / 5 with own, swap, none, paired vs out/clm_eval/e1_gated_s0_wkv_20k_c96_cfg{1,3,5}.
# CFG 1 decides (binding without guidance: own CER clearly below none).
# usage: run_e1_asr.sh GPU PROJ TAG [extra trainer args]   e.g. run_e1_asr.sh 3 out/probe/asr_stream_v4_lin/lin32.pt _lin32
cd /root/x/rwkv-music; G=$1; P=$2; N=e1_gated_s0_wkv_asr$3; shift 3
export CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=venvs/yue2/bin/python
mkdir -p out/clm/$N; cp $P out/clm/$N/asr_proj.pt                     # frozen copy: the probe dir may be overwritten later
$PY vae/train_cursor_lm.py --mode cursor --head gated --seed 0 --out out/clm/$N --vae out/vae/v4/G_s60000.pt --sem-w 3 --steps 20000 \
  --asr-pca out/clm/$N/asr_proj.pt "$@" \
  --manifest data/real_corpus/manifest/pilot_clm_train_clean.jsonl --val-manifest data/real_corpus/manifest/slim_val_clean.jsonl > logs/clm_$N.log 2>&1
echo "$(date) training rc=$?"
for C in 1 3 5; do
  $PY vae/eval_cursor_lm.py --ckpt out/clm/$N/M_ema_s20000.pt --vae out/vae/v4/G_s60000.pt --n 96 --causal-prompt --cfg $C --conds own,swap,none \
    --out out/clm_eval/${N}_20k_c96_cfg$C > logs/gate1_${N}_cfg$C.log 2>&1
done
echo "$(date) gate done"
