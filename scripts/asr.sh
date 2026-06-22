#!/bin/bash
# Japanese ASR fine-tuning: WavLM-large encoder -> SpeechProjector -> RWKV7-1.5B
# Target: AMD ROCm (gfx1100). Uses the fla/triton WKV kernel (--op fla) and a
# single GPU (no DeepSpeed -> AdamW fallback in model.py).
#
# Step 0) Build the local subset first (≈1% of reazon_speech_all):
#    python -m lg_train.prepare.build_reazon_subset \
#        --out /DATA/disk0/reazon_1pct --configs subset_0 --max_samples 50000 --max_sec 20
#
# Two-stage recipe (recommended):
#   Stage 1 (this script): freeze the LLM, train ONLY the SpeechProjector to
#     align audio features into the RWKV embedding space.  Stable, loss drops.
#   Stage 2 (optional): unfreeze the LLM with a small lr to jointly fine-tune.
#     Full-LLM fine-tuning with the default layerwise 2x/3x lr on the time-mix
#     params is unstable in bf16 (-> NaN); if you do Stage 2 use a tiny lr
#     (e.g. 1e-5), warmup, and consider --layerwise_lr 0.

load_model=models/rwkv7-g1g-1.5b-20260526-ctx8192.pth
proj_dir=out/asr-wavlm-rwkv1b5-stage1
data_file=./localdataset      # output of build_reazon_subset.py (185,621 clips, ~1%)

n_layer=24
n_embd=2048

encoder_path="microsoft/wavlm-large"
encoder_type=speech
data_type=asr

micro_bsz=8
ctx_len=1024
epoch_save=1
# NOTE: WorldDataset trims the dataset to `epoch_steps` SAMPLES (not steps).
# Set >= dataset size (185,621) to use the full set; lower it for a quick run.
epoch_steps=185621

# ---- Stage 1: projector alignment (LLM frozen) ----
python train.py \
  --load_model $load_model \
  --proj_dir $proj_dir --data_file $data_file \
  --data_type $data_type \
  --vocab_size 65536 \
  --n_layer $n_layer --n_embd $n_embd \
  --ctx_len $ctx_len --micro_bsz $micro_bsz \
  --epoch_steps $epoch_steps --epoch_count 2 --epoch_begin 0 --epoch_save $epoch_save \
  --lr_init 1e-3 --lr_final 1e-4 --warmup_steps 50 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
  --accelerator gpu --devices 1 --precision bf16 --strategy auto --grad_cp 1 \
  --encoder_path $encoder_path --encoder_type $encoder_type \
  --op fla --my_testing "x070" \
  --train_step proj \
  --wandb mod-rwkv-asr --wandb_mode online \
  --num_workers 8

# Before first online run, authenticate once in the shell:  ! wandb login
# Live dashboard logs: loss, acc (token acc on supervised positions), lr, Gtokens, kt/s.

# ---- Stage 2 (optional): joint fine-tune, resume from stage-1 checkpoint ----
# python train.py ... \
#   --load_model out/asr-wavlm-rwkv1b5-stage1/rwkv-<N>.pth \
#   --lr_init 1e-5 --lr_final 1e-6 --warmup_steps 100 --layerwise_lr 0 \
#   --train_step proj rwkv
