#!/bin/bash
# =============================================================================
# Japanese ASR fine-tuning launcher
#   WavLM-large (encoder, frozen) -> SpeechProjector -> RWKV7-G1 1.5B
#   Target: AMD ROCm (gfx1100) / single GPU / fla triton WKV kernel
# =============================================================================
# Prereqs:
#   - dataset built:  python -m lg_train.prepare.build_reazon_subset --out ./localdataset --frac 0.01
#   - (live monitor)  wandb login        # run once in the shell:  ! wandb login
# Usage:
#   bash scripts/train_asr.sh             # full Stage-1 run
#   QUICK=1 bash scripts/train_asr.sh     # short smoke run (2000 samples, no wandb)
#   STAGE=2 bash scripts/train_asr.sh     # joint fine-tune from a Stage-1 ckpt (edit RESUME)
# =============================================================================
set -e
cd "$(dirname "$0")/.."          # repo root

# ---- paths ----
LOAD_MODEL=models/rwkv7-g1g-1.5b-20260526-ctx8192.pth
DATA_FILE=./localdataset                  # build_reazon_subset.py output (185,621 clips)
ENCODER_PATH=microsoft/wavlm-large
RESUME=/home/client/Projects/mod-rwkv/out/asr-wavlm-rwkv1b5-stage1/rwkv-step-6800.pth                               # Stage-2: path to a Stage-1 rwkv-*.pth

# ---- model dims (RWKV7-G1 1.5B) ----
N_LAYER=24
N_EMBD=2048

# ---- training ----
MICRO_BSZ=16
CTX_LEN=1024
EPOCH_COUNT=2
EPOCH_SAVE=1
SAVE_PER_STEPS=${SAVE_PER_STEPS:-100}   # also dump rwkv-step-N.pth every N steps (0=off)
NUM_WORKERS=8
# DEVICES / STRATEGY get per-stage defaults below (env vars still override).
# WorldDataset trims to epoch_steps SAMPLES (not steps); >= dataset size = use all.
EPOCH_STEPS=185621

# ---- wandb ----
WANDB_PROJECT=mod-rwkv-asr
WANDB_MODE=online

# ---- stage selection ----
STAGE=${STAGE:-2}
if [ "$STAGE" = "1" ]; then
    PROJ_DIR=out/asr-wavlm-rwkv1b5-stage1p
    TRAIN_STEP="proj"                     # freeze LLM, align projector
    LR_INIT=1e-3; LR_FINAL=1e-4; WARMUP=50; LAYERWISE_LR=1
    DEVICES=${DEVICES:-1}                  # proj-only is light: single GPU is plenty
    STRATEGY=${STRATEGY:-auto}
else
    PROJ_DIR=out/asr-wavlm-rwkv1b5-stage2
    TRAIN_STEP="proj att"                # joint fine-tune (small lr!)
    LR_INIT=1e-5; LR_FINAL=1e-6; WARMUP=100; LAYERWISE_LR=0
    [ -n "$RESUME" ] && LOAD_MODEL=$RESUME
    DEVICES=${DEVICES:-2}                  # full 1.5B fine-tune: shard across 2 GPUs
    STRATEGY=${STRATEGY:-deepspeed_stage_2_offload}  # ZeRO-2 shards optimizer states + grads
fi

# ---- quick smoke override ----
if [ "${QUICK:-0}" = "1" ]; then
    PROJ_DIR=out/asr-quick
    EPOCH_STEPS=2000; EPOCH_COUNT=1; NUM_WORKERS=2
    WANDB_PROJECT=""                      # disable wandb for smoke
fi

WANDB_ARGS=""
[ -n "$WANDB_PROJECT" ] && WANDB_ARGS="--wandb $WANDB_PROJECT --wandb_mode $WANDB_MODE"

echo "=== ASR train | stage=$STAGE | train_step='$TRAIN_STEP' | data=$DATA_FILE ==="
echo "=== model=$LOAD_MODEL -> $PROJ_DIR | bsz=$MICRO_BSZ ctx=$CTX_LEN lr=$LR_INIT ==="

python train.py \
  --load_model "$LOAD_MODEL" \
  --proj_dir "$PROJ_DIR" --data_file "$DATA_FILE" \
  --data_type asr --vocab_size 65536 \
  --n_layer $N_LAYER --n_embd $N_EMBD \
  --ctx_len $CTX_LEN --micro_bsz $MICRO_BSZ \
  --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin 0 --epoch_save $EPOCH_SAVE \
  --save_per_steps $SAVE_PER_STEPS --inspect_grad ${INSPECT:-0} --spike_thresh ${SPIKE:-0} \
  --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP \
  --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --layerwise_lr $LAYERWISE_LR \
  --accelerator gpu --devices $DEVICES --precision bf16 --strategy $STRATEGY --grad_cp 1 \
  --encoder_path "$ENCODER_PATH" --encoder_type speech \
  --op fla --my_testing x070 \
  --train_step $TRAIN_STEP \
  $WANDB_ARGS \
  --num_workers $NUM_WORKERS
