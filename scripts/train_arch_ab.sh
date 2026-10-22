#!/usr/bin/env bash
# Architecture A/B: RWKV-7 time-mix (fixed-size state) vs causal softmax self-attention (RoPE + QK-norm), everything else
# identical (src/model.py Attn_Tmix, selected by RWKV_MIXER). From scratch, L12 D768 head 64, one GPU per run,
# same data order (magic_prime sampling), same LR schedule and token budget. Scored by scripts/arch_ab_eval.py.
# usage: [SLOT_HEADS=K SLOT_M=M] bash scripts/train_arch_ab.sh MIXER(rwkv|attn) FMT(whole|sec|line) GPU [EPOCHS=1] [M_BSZ=4]
#   SLOT_HEADS/SLOT_M (rwkv only): state axes, docs/ARCHITECTURE_NOTES.md section 8 (K protected lyric heads in M line slots)
#   AUX_W/AUX_LAYERS: lyric-cursor CTC loss (section 9), weight and tapped block indices, e.g. AUX_W=0.1 AUX_LAYERS=11 or 5,11
#   TAG: suffix for the run directory (e.g. TAG=seed2 for a repeat)
#   whole = data/binidx/yue2_stage1_*  (G3 format: all lyrics before the song, binding distance = thousands of tokens)
#   sec   = data/binidx/yue2_sec_stage1_* (R3.1 format: each stanza's text right before its own frames)
#   line  = data/binidx/yue2_line_stage1_* (R3.2 format: one SOS block per lyric line)
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
source /root/envs/env_rwkv 2>/dev/null || true

MIXER=$1; FMT=$2; GPU=$3; EPOCHS=${4:-1}; M_BSZ=${5:-4}
SLOT_HEADS=${SLOT_HEADS:-0}; SLOT_M=${SLOT_M:-1}; AUX_W=${AUX_W:-0}; AUX_LAYERS=${AUX_LAYERS:-}; TAG=${TAG:-}
case $FMT in whole) NAME=yue2_stage1 ;; sec) NAME=yue2_sec_stage1 ;; line) NAME=yue2_line_stage1 ;; *) echo "FMT must be whole|sec|line"; exit 1 ;; esac
VAR=${MIXER}_${FMT}; [ "$SLOT_HEADS" -gt 0 ] && VAR=${VAR}_slot${SLOT_HEADS}x${SLOT_M}
[ "$AUX_W" != "0" ] && VAR=${VAR}_aux${AUX_W}L$(echo "$AUX_LAYERS" | tr ',' '-')
[ -n "$TAG" ] && VAR=${VAR}_$TAG
PROJ_DIR=$ROOT/out/arch_ab/$VAR
DATA=$ROOT/data/binidx/${NAME}_train
INFO=$ROOT/data/binidx/${NAME}_binidx.json
read -r N_TOKENS MAGIC_PRIME <<< "$(python3 -c "import json;d=json.load(open('$INFO'))['train'];print(d['tokens'],d['magic_prime'])")"
EXIT_TOKENS=$(python3 -c "print(int($N_TOKENS*$EPOCHS))")

export RWKV_MIXER=$MIXER RWKV_SLOT_HEADS=$SLOT_HEADS RWKV_SLOT_M=$SLOT_M
export RWKV_AUX_W=$AUX_W RWKV_AUX_LAYERS=$AUX_LAYERS RWKV_AUX_VOCAB=$ROOT/data/binidx/yue2_line_aux_vocab.npy
export RWKV_HEAD_CE_VOCAB=98816
export RWKV_SAVE_STEPS=${RWKV_SAVE_STEPS:-2000}
export RWKV_STEP_OFFSET=0
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PROJ_DIR" "$ROOT/logs"
LOG=$ROOT/logs/train_arch_ab_${VAR}_$(date +%Y%m%d_%H%M%S).log
echo "log: $LOG  ($VAR, GPU $GPU, $EXIT_TOKENS tokens = $EPOCHS epochs of $N_TOKENS, micro_bsz $M_BSZ)"

cd "$ROOT/tools/RWKV-LM/RWKV-v7/train_temp"
setsid nohup python train.py --load_model "" --wandb "" --proj_dir "$PROJ_DIR" --my_testing x070 \
  --ctx_len 8192 --train_stage 0 --epoch_count 999999 --epoch_begin 0 \
  --data_file "$DATA" --data_type binidx --vocab_size 98816 --my_exit_tokens $EXIT_TOKENS --magic_prime $MAGIC_PRIME \
  --num_nodes 1 --micro_bsz $M_BSZ --n_layer 12 --n_embd 768 --dim_att 768 --head_size 64 \
  --lr_init 6e-4 --lr_final 6e-5 --warmup_steps 300 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
  --weight_decay 0.1 --epoch_save 10 --head_chunk 65536 --grad_clip 1.0 \
  --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 \
  --enable_progress_bar False > "$LOG" 2>&1 < /dev/null &
echo "pid $!"
