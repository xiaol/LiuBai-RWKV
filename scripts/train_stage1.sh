#!/usr/bin/env bash
# Stage-1 continue-pretraining of rwkv-g1k-3b on suno-94k cb0 documents (PLAN.md §3).
# Usage: bash scripts/train_stage1.sh [GPUS="0,1,2,3"] [EXIT_TOKENS]
#
# Runs a supervisor in the background (log logs/train_stage1_*.log) that:
#   * resumes from the newest out/stage1/step-N.pth (saved every RWKV_SAVE_STEPS=1000 steps by the patched
#     trainer.py) with RWKV_STEP_OFFSET=N, so LR schedule and data windows continue where they stopped;
#     falls back to out/stage1/rwkv-init.pth (resized g1k) when there is none;
#   * restarts automatically when the trainer kills itself on a non-finite loss (up to MAX_RESTARTS);
#   * stops when out/stage1/rwkv-final.pth exists (cosine horizon EXIT_TOKENS reached).
# train_stage 0 is used so train.py does not override --load_model with its own rwkv-N.pth search.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
source /root/envs/env_rwkv 2>/dev/null || true       # CUDA_HOME, nvcc, conda env rwkv_py312

GPUS=${1:-0,1,2,3}
N_GPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PROJ_DIR=$ROOT/out/stage1
DATA=$ROOT/data/binidx/suno94k_stage1_train      # 844,987,943 tokens, magic_prime 51563 @ ctx 16384
MAGIC_PRIME=51563
CTX_LEN=16384
M_BSZ=1                                         # per GPU; real_bsz = N_GPU
EXIT_TOKENS=${2:-1690000000}                    # ≈ 2 epochs of the train set; cosine LR ends here and rwkv-final.pth is written
LR_INIT=6e-5                                    # 1e-4 diverged to NaN at step 4645 on 2026-09-13 (PLAN.md §6)
LR_FINAL=6e-6
MAX_RESTARTS=6

# architecture of rwkv-g1k-3b-temp-5441 (L32 D2560, ffn 4x, LoRA dims 96/96/64/320; see PLAN.md §3)
export RWKV_D_DECAY_LORA=96 RWKV_D_AAA_LORA=96 RWKV_D_MV_LORA=64 RWKV_D_GATE_LORA=320
export RWKV_HEAD_CE_VOCAB=67072                 # compile-time vocab of the chunked head-CE kernel (patched to accept a define)
export RWKV_SAVE_STEPS=1000
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ROOT PROJ_DIR DATA MAGIC_PRIME CTX_LEN M_BSZ EXIT_TOKENS LR_INIT LR_FINAL MAX_RESTARTS N_GPU  # used inside supervise()
mkdir -p "$PROJ_DIR" "$ROOT/logs"
LOG=$ROOT/logs/train_stage1_$(date +%Y%m%d_%H%M%S).log
echo "log: $LOG"

supervise() {
  cd "$ROOT/tools/RWKV-LM/RWKV-v7/train_temp"
  for attempt in $(seq 0 $MAX_RESTARTS); do
    if [ -f "$PROJ_DIR/rwkv-final.pth" ]; then echo "[supervisor] rwkv-final.pth exists; done"; return 0; fi
    LATEST=$(ls "$PROJ_DIR"/step-*.pth 2>/dev/null | sed 's/.*step-\([0-9]*\)\.pth/\1 &/' | sort -n | tail -n 1 | cut -d' ' -f2)
    if [ -n "$LATEST" ]; then
      LOAD=$LATEST; export RWKV_STEP_OFFSET=$(basename "$LATEST" .pth | cut -d- -f2)
    else
      LOAD=$PROJ_DIR/rwkv-init.pth; export RWKV_STEP_OFFSET=0
    fi
    echo "[supervisor] $(date) attempt $attempt: load $LOAD, step offset $RWKV_STEP_OFFSET"
    echo "$(date) attempt $attempt load $(basename "$LOAD") offset $RWKV_STEP_OFFSET" >> "$PROJ_DIR/events.txt"
    setsid python train.py --load_model "$LOAD" --wandb "" --proj_dir "$PROJ_DIR" --my_testing x070 \
     --ctx_len $CTX_LEN --train_stage 0 --epoch_count 999999 --epoch_begin 0 \
     --data_file "$DATA" --data_type binidx --vocab_size 67072 --my_exit_tokens $EXIT_TOKENS --magic_prime $MAGIC_PRIME \
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
