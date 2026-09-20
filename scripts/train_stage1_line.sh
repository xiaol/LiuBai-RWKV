#!/usr/bin/env bash
# R3.2a probe: line-level section-only continued training of the R3.1 model (docs/STATUS_LOG.md 2026-09-20).
# Usage: bash scripts/train_stage1_line.sh [GPUS="0,1,2,3"] [EPOCHS=1]
#
# Data   : data/binidx/yue2_line_stage1_train (build_yue2_sec_binidx.py --unit line --whole-frac 0: one SOS block per lyric line)
# Init   : out/stage1_line/rwkv-init.pth -> symlink to out/stage1_sec/rwkv-final.pth (R3.1)
# Layout : as G3 plus SOS 65541 (scripts/yue2_layout.py)
# Same supervisor as train_stage1.sh: resumes from the newest out/stage1_sec/step-N.pth with RWKV_STEP_OFFSET=N,
# restarts after a non-finite-loss kill (trainer.py patch), stops when rwkv-final.pth exists (cosine horizon).
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
source /root/envs/env_rwkv 2>/dev/null || true

GPUS=${1:-0,1,2,3}
EPOCHS=${2:-1}
N_GPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PROJ_DIR=$ROOT/out/stage1_line
DATA=$ROOT/data/binidx/yue2_line_stage1_train
INFO=$ROOT/data/binidx/yue2_line_stage1_binidx.json
read -r N_TOKENS MAGIC_PRIME <<< "$(python3 -c "import json;d=json.load(open('$INFO'))['train'];print(d['tokens'],d['magic_prime'])")"
CTX_LEN=8192
M_BSZ=1
EXIT_TOKENS=$(python3 -c "print(int($N_TOKENS*$EPOCHS))")
LR_INIT=3e-5                                    # higher than R3.1 (2e-5): the SOS/line rows are new and R3.1 loss was still falling at its tail
LR_FINAL=3e-6
MAX_RESTARTS=8

export RWKV_D_DECAY_LORA=96 RWKV_D_AAA_LORA=96 RWKV_D_MV_LORA=64 RWKV_D_GATE_LORA=320
export RWKV_HEAD_CE_VOCAB=98816
export RWKV_SAVE_STEPS=500
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROOT PROJ_DIR DATA MAGIC_PRIME CTX_LEN M_BSZ EXIT_TOKENS LR_INIT LR_FINAL MAX_RESTARTS N_GPU
mkdir -p "$PROJ_DIR" "$ROOT/logs"
LOG=$ROOT/logs/train_stage1_line_$(date +%Y%m%d_%H%M%S).log
echo "log: $LOG"; echo "train tokens $N_TOKENS, magic_prime $MAGIC_PRIME, exit tokens $EXIT_TOKENS ($EPOCHS epochs), $N_GPU GPUs"

supervise() {
  cd "$ROOT/tools/RWKV-LM/RWKV-v7/train_temp"
  for attempt in $(seq 0 $MAX_RESTARTS); do
    if [ -f "$PROJ_DIR/rwkv-final.pth" ]; then echo "[supervisor] rwkv-final.pth exists; done"; return 0; fi
    LATEST=$(ls "$PROJ_DIR"/step-*.pth 2>/dev/null | sed 's/.*step-\([0-9]*\)\.pth/\1 &/' | sort -n | tail -n 1 | cut -d' ' -f2)
    if [ -n "$LATEST" ]; then LOAD=$LATEST; export RWKV_STEP_OFFSET=$(basename "$LATEST" .pth | cut -d- -f2)
    else LOAD=$PROJ_DIR/rwkv-init.pth; export RWKV_STEP_OFFSET=0; fi
    echo "[supervisor] $(date) attempt $attempt: load $LOAD, step offset $RWKV_STEP_OFFSET"
    echo "$(date) attempt $attempt load $(basename "$LOAD") offset $RWKV_STEP_OFFSET" >> "$PROJ_DIR/events.txt"
    setsid python train.py --load_model "$LOAD" --wandb "" --proj_dir "$PROJ_DIR" --my_testing x070 \
     --ctx_len $CTX_LEN --train_stage 0 --epoch_count 999999 --epoch_begin 0 \
     --data_file "$DATA" --data_type binidx --vocab_size 98816 --my_exit_tokens $EXIT_TOKENS --magic_prime $MAGIC_PRIME \
     --num_nodes 1 --micro_bsz $M_BSZ --n_layer 32 --n_embd 2560 --dim_att 2560 --dim_ffn 10240 --head_size 64 \
     --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 500 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
     --weight_decay 0.1 --epoch_save 1 --head_chunk 65536 --grad_clip 1.0 \
     --accelerator gpu --devices $N_GPU --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 \
     --enable_progress_bar False
    rc=$?
    echo "[supervisor] $(date) train.py exited rc=$rc"
    sleep 20
  done
  echo "[supervisor] giving up after $MAX_RESTARTS restarts"
}

setsid nohup bash -c "$(declare -f supervise); supervise" > "$LOG" 2>&1 < /dev/null &
echo "supervisor pid $!"
