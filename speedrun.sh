#!/bin/bash
# speedrun.sh - Complete pipeline for OLMo molecular property prediction
#
# This file serves as living documentation showing exact commands.
# Run sections individually or the whole script.

set -e

echo "============================================"
echo "OLMo Molecular Property Prediction Pipeline"
echo "============================================"

# ============================================================================
# 0. DATA PREPARATION (run this first)
# ============================================================================

# Prepare all MoleculeNet datasets with scaffold splits
echo "=== Preparing datasets ==="
python3 prepare_data.py\
    --split_type 'deepchem' \
    --datasets 'delaney' \
    --data_dir ./../datasets/deepchem_splits \

# Or prepare specific datasets:
# python scripts/prepare_data.py --datasets bbbp bace hiv clintox tox21 sider
# python scripts/prepare_data.py --datasets clearance esol freesolv lipophilicity

# ============================================================================
# 1. CLASSIFICATION FINE-TUNING
# ============================================================================

# Binary classification on BBBP (single GPU)
echo "=== Training on BBBP (binary classification) ==="
python scripts/train_classification.py \
    --task bbbp \
    --use_qlora \
    --batch_size 4 \
    --gradient_accum 24 \
    --epochs 15

# Multi-GPU training on HIV
echo "=== Training on HIV (4 GPUs) ==="
torchrun --nproc_per_node=4 scripts/train_classification.py \
    --task hiv \
    --use_qlora \
    --batch_size 4 \
    --gradient_accum 6

# Multitask classification on Tox21
echo "=== Training on Tox21 (multitask) ==="
torchrun --nproc_per_node=4 scripts/train_classification.py \
    --task tox21 \
    --use_qlora \
    --batch_size 4 \
    --gradient_accum 6

# Multilabel classification on SIDER
echo "=== Training on SIDER (multilabel) ==="
python scripts/train_classification.py \
    --task sider \
    --use_qlora \
    --batch_size 4 \
    --gradient_accum 24

# LM-head approach (Yes/No prediction)
echo "=== Training on BACE with LM head ==="
python scripts/train_classification.py \
    --task bace \
    --use_lm_head \
    --use_qlora

# Full finetuning (no LoRA)
echo "=== Full finetuning on ClinTox ==="
python scripts/train_classification.py \
    --task clintox \
    --full_finetune \
    --lr 1e-5 \
    --batch_size 2

# ============================================================================
# 2. REGRESSION FINE-TUNING
# ============================================================================

echo "=== Training on clearance (regression) ==="
python scripts/train_regression.py \
    --task clearance \
    --use_qlora \
    --epochs 30

echo "=== Training on ESOL (solubility) ==="
python scripts/train_regression.py \
    --task esol \
    --use_qlora

echo "=== Training on lipophilicity ==="
python scripts/train_regression.py \
    --task lipophilicity \
    --use_qlora

# ============================================================================
# 3. PRETRAINING
# ============================================================================

echo "=== Pretraining on ZINC20 ==="
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --dataset zinc20 \
    --num_samples 1000000 \
    --epochs 1

echo "=== Pretraining on PubChem ==="
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --dataset pubchem \
    --num_samples 1000000

# ============================================================================
# 4. INSTRUCTION TUNING
# ============================================================================

echo "=== Instruction tuning on USPTO ==="
torchrun --nproc_per_node=4 scripts/train_instruction.py \
    --dataset OpenMol/USPTO_1k_TPL-SFT \
    --num_samples 10000

# ============================================================================
# 5. FULL PIPELINE: Pretrain -> Finetune
# ============================================================================

# Pretrain on ZINC20
# torchrun --nproc_per_node=4 scripts/pretrain.py \
#     --dataset zinc20 \
#     --num_samples 10000000 \
#     --hub_name youruser/OLMo-7B-ZINC20

# Instruction tune the ZINC-pretrained model
# torchrun --nproc_per_node=4 scripts/train_instruction.py \
#     --model_name youruser/OLMo-7B-ZINC20 \
#     --dataset OpenMol/USPTO_1k_TPL-SFT \
#     --hub_name youruser/OLMo-7B-ZINC-USPTO

# Finetune on downstream task
# python scripts/train_classification.py \
#     --task bbbp \
#     --model_name youruser/OLMo-7B-ZINC-USPTO

echo "============================================"
echo "Pipeline complete!"
echo "============================================"
