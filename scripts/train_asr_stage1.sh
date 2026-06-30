#!/bin/bash
# =============================================================================
# Japanese ASR fine-tuning launcher  (everything is env-overridable)
#   WavLM-large (encoder, frozen) -> SpeechProjector -> RWKV7-G1 1.5B
#   Single GPU / multi-GPU / multi-node cluster.  fla triton WKV kernel.
# =============================================================================
# Just run it:        ./scripts/train_asr_stage1.sh
# Stage-2 (LoRA):     STAGE=2 LORA_TMIX=16 LORA_FFN=8 RESUME=out/.../rwkv-step-N.pth ./scripts/train_asr_stage1.sh
# On-the-fly labels:  DATA_TYPE=label DATA_FILE=/share/voice-dataset ./scripts/train_asr_stage1.sh
# Quick smoke:        QUICK=1 ./scripts/train_asr_stage1.sh
# Multi-node cluster: NUM_NODES=4 DEVICES=8 LAUNCHER=srun ./scripts/train_asr_stage1.sh
#                     (Lightning auto-detects SLURM; LAUNCHER prefixes the python call)
#
# Env knobs (all optional, with sensible defaults):
#   STAGE QUICK
#   LOAD_MODEL DATA_TYPE DATA_FILE ENCODER_PATH RESUME OUT_DIR
#   N_LAYER N_EMBD VOCAB
#   MICRO_BSZ CTX_LEN EPOCH_COUNT EPOCH_STEPS EPOCH_SAVE NUM_WORKERS ACCUM GRAD_CP PRECISION
#   DATA_SHUFFLE  (1=on default, 0=off)
#   LABEL_EXCLUDE (data_type=label: comma-separated keywords to skip, e.g. "misc,noise")
#   AUDIO_ROOT RESOLVE_MAX_UP (data_type=label: audio path resolution; see docs/check-label-dataset.md)
#   DEVICES NUM_NODES STRATEGY ACCELERATOR LAUNCHER
#   TRAIN_STEP LR_INIT LR_FINAL WARMUP LAYERWISE_LR
#   LORA_TMIX LORA_FFN LORA_ALPHA LORA_DROPOUT
#   INSPECT INSPECT_LAYER SPIKE SAVE_PER_STEPS KEEP_LAST_CKPT
#   WANDB_PROJECT WANDB_MODE
# =============================================================================
set -e
cd "$(dirname "$0")/.."          # repo root

# ---- paths / data ----
LOAD_MODEL=${LOAD_MODEL:-models/rwkv7-g1g-1.5b-20260526-ctx8192.pth}
DATA_TYPE=${DATA_TYPE:-asr}                # asr = pre-materialized | label = on-the-fly *.label folder
DATA_FILE=${DATA_FILE:-./localdataset}     # asr: save_to_disk dir | label: root folder of *.label files
AUDIO_ROOT=${AUDIO_ROOT:-}                  # data_type=label: explicit base dir for relative audio (tried first)
RESOLVE_MAX_UP=${RESOLVE_MAX_UP:-2}         # data_type=label: parent levels tried for off-by-one paths
ENCODER_PATH=${ENCODER_PATH:-microsoft/wavlm-large}
RESUME=${RESUME:-}                         # Stage-2: path to a Stage-1 rwkv-*.pth
OUT_DIR=${OUT_DIR:-out}

# ---- model dims (RWKV7-G1 1.5B) ----
N_LAYER=${N_LAYER:-24}
N_EMBD=${N_EMBD:-2048}
VOCAB=${VOCAB:-65536}

# ---- training ----
MICRO_BSZ=${MICRO_BSZ:-8}
CTX_LEN=${CTX_LEN:-1024}
EPOCH_COUNT=${EPOCH_COUNT:-2}
EPOCH_SAVE=${EPOCH_SAVE:-1}
# WorldDataset trims to epoch_steps SAMPLES (not steps); >= dataset size = use all.
EPOCH_STEPS=${EPOCH_STEPS:-185621}
NUM_WORKERS=${NUM_WORKERS:-8}
ACCUM=${ACCUM:-1}                          # accumulate_grad_batches
GRAD_CP=${GRAD_CP:-1}                      # gradient checkpointing
PRECISION=${PRECISION:-bf16}
DATA_SHUFFLE=${DATA_SHUFFLE:-1}            # 1=shuffle index (default), 0=keep scan order
LABEL_EXCLUDE=${LABEL_EXCLUDE:-}           # data_type=label: comma-separated keywords to skip (e.g. "misc,noise")
SAVE_PER_STEPS=${SAVE_PER_STEPS:-100}      # dump rwkv-step-N.pth every N steps (0=off)
KEEP_LAST_CKPT=${KEEP_LAST_CKPT:-5}        # keep only the N most recent rwkv-step-*.pth (0=keep all)

# ---- cluster / devices ----
ACCELERATOR=${ACCELERATOR:-gpu}
NUM_NODES=${NUM_NODES:-1}                  # >1 for multi-node (launch via LAUNCHER=srun ...)
LAUNCHER=${LAUNCHER:-}                     # e.g. "srun" or "torchrun --nnodes=..." ; empty = plain python

# ---- wandb ----
WANDB_PROJECT=${WANDB_PROJECT:-mod-rwkv-asr}
WANDB_MODE=${WANDB_MODE:-online}

# ---- diagnostics (0=off) ----
INSPECT=${INSPECT:-0}                      # per-module/per-layer grad-norm: print every N steps
INSPECT_LAYER=${INSPECT_LAYER:-0}          # block idx for per-parameter grad-norm breakdown
SPIKE=${SPIKE:-0}                          # dump the offending batch when grad_norm > SPIKE
DEBUG=${DEBUG:-0}                          # 1 = print which dataset samples are used each step

# ---- LoRA (independent ranks for time-mix / ffn; 0 = off) ----
LORA_TMIX=${LORA_TMIX:-0}
LORA_FFN=${LORA_FFN:-0}
LORA_ALPHA=${LORA_ALPHA:-0}                # 0 -> alpha=rank (scaling 1.0)
LORA_DROPOUT=${LORA_DROPOUT:-0}

# ---- stage selection (TRAIN_STEP/LR/DEVICES/STRATEGY get per-stage defaults) ----
STAGE=${STAGE:-1}
if [ "$STAGE" = "1" ]; then
    PROJ_DIR=${PROJ_DIR:-$OUT_DIR/asr-wavlm-rwkv1b5-stage1}
    TRAIN_STEP="${TRAIN_STEP:-proj}"                  # freeze LLM, align projector
    LR_INIT=${LR_INIT:-1e-3}; LR_FINAL=${LR_FINAL:-1e-4}; WARMUP=${WARMUP:-50}; LAYERWISE_LR=${LAYERWISE_LR:-1}
    DEVICES=${DEVICES:-1}                             # proj-only is light
    STRATEGY=${STRATEGY:-auto}
else
    [ -n "$RESUME" ] && LOAD_MODEL=$RESUME
    if [ "$LORA_TMIX" != "0" ] || [ "$LORA_FFN" != "0" ]; then
        PROJ_DIR=${PROJ_DIR:-$OUT_DIR/asr-wavlm-rwkv1b5-stage2-lora}
        TRAIN_STEP="${TRAIN_STEP:-proj lora}"          # base frozen, train injected LoRA
        LR_INIT=${LR_INIT:-2e-4}
    else
        PROJ_DIR=${PROJ_DIR:-$OUT_DIR/asr-wavlm-rwkv1b5-stage2}
        TRAIN_STEP="${TRAIN_STEP:-proj att_noln}"      # time-mix w/o ln_x (the layer-0 spike source)
        LR_INIT=${LR_INIT:-1e-5}
    fi
    LR_FINAL=${LR_FINAL:-1e-6}; WARMUP=${WARMUP:-100}; LAYERWISE_LR=${LAYERWISE_LR:-0}
    DEVICES=${DEVICES:-2}
    STRATEGY=${STRATEGY:-deepspeed_stage_2_offload}    # ZeRO-2(+offload): shard optim states + grads
fi

# ---- quick smoke override ----
if [ "${QUICK:-0}" = "1" ]; then
    PROJ_DIR=$OUT_DIR/asr-quick
    EPOCH_STEPS=2000; EPOCH_COUNT=1; NUM_WORKERS=2
    WANDB_PROJECT=""
fi

WANDB_ARGS=""
[ -n "$WANDB_PROJECT" ] && WANDB_ARGS="--wandb $WANDB_PROJECT --wandb_mode $WANDB_MODE"

REAL_BSZ=$(( NUM_NODES * DEVICES * MICRO_BSZ * ACCUM ))
echo "=== ASR train | stage=$STAGE | train_step='$TRAIN_STEP' | data_type=$DATA_TYPE data=$DATA_FILE ==="
echo "=== model=$LOAD_MODEL -> $PROJ_DIR ==="
echo "=== nodes=$NUM_NODES x devices=$DEVICES x bsz=$MICRO_BSZ x accum=$ACCUM = real_bsz $REAL_BSZ | ctx=$CTX_LEN | strat=$STRATEGY | lr=$LR_INIT ==="

$LAUNCHER python train.py \
  --load_model "$LOAD_MODEL" \
  --proj_dir "$PROJ_DIR" --data_file "$DATA_FILE" \
  --data_type $DATA_TYPE --vocab_size $VOCAB --data_shuffle $DATA_SHUFFLE \
  --label_exclude "$LABEL_EXCLUDE" \
  --audio_root "$AUDIO_ROOT" --resolve_max_up $RESOLVE_MAX_UP \
  --n_layer $N_LAYER --n_embd $N_EMBD \
  --ctx_len $CTX_LEN --micro_bsz $MICRO_BSZ \
  --epoch_steps $EPOCH_STEPS --epoch_count $EPOCH_COUNT --epoch_begin 0 --epoch_save $EPOCH_SAVE \
  --save_per_steps $SAVE_PER_STEPS --keep_last_ckpt $KEEP_LAST_CKPT \
  --inspect_grad $INSPECT --inspect_layer $INSPECT_LAYER --spike_thresh $SPIKE --debug_data $DEBUG \
  --lora_tmix $LORA_TMIX --lora_ffn $LORA_FFN --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT \
  --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP \
  --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --layerwise_lr $LAYERWISE_LR \
  --accelerator $ACCELERATOR --devices $DEVICES --num_nodes $NUM_NODES \
  --precision $PRECISION --strategy $STRATEGY --grad_cp $GRAD_CP \
  --accumulate_grad_batches $ACCUM \
  --encoder_path "$ENCODER_PATH" --encoder_type speech \
  --op fla --my_testing x070 \
  --train_step $TRAIN_STEP \
  $WANDB_ARGS \
  --num_workers $NUM_WORKERS
