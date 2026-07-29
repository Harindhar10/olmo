#!/bin/bash
set -euo pipefail

# Base args shared by every run.
MODEL_NAME="harindhar10/OLMo-7B-fsdp-Pubchem-5M-1epochs"
FINETUNE_STRATEGY="full_finetune"
MAX_LEN=128
GRADIENT_ACCUM=1
VAL_CHECK_INTERVAL=1
LR=1e-6
DATA_DIR="datasets"
WANDB="True"
WANDB_KEY="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"

# dataset:batch_size:epochs, per the config table.
RUNS=(
  "bbbp:8:5"
  "bace_classification:16:5"
  "freesolv:8:5"
  "delaney:16:5"
  "bace_regression:8:5"
  "sider:8:5"
  "tox21:24:5"
  "clearance:8:5"
  "clintox:8:5"
  "lipo:32:5"
  "hiv:128:3"
)

for run in "${RUNS[@]}"; do
  IFS=":" read -r dataset batch_size epochs <<< "$run"
  echo "============================================================"
  echo "Running dataset=${dataset} batch_size=${batch_size} epochs=${epochs}"
  echo "============================================================"
  python olmo/scripts/run_experiment.py \
    --model_name "$MODEL_NAME" \
    --datasets "$dataset" \
    --finetune_strategy "$FINETUNE_STRATEGY" \
    --epochs "$epochs" \
    --max_len "$MAX_LEN" \
    --batch_size "$batch_size" \
    --gradient_accum "$GRADIENT_ACCUM" \
    --val_check_interval "$VAL_CHECK_INTERVAL" \
    --lr "$LR" \
    --data_dir "$DATA_DIR" \
    --wandb "$WANDB" \
    --wandb_key "$WANDB_KEY"
done
