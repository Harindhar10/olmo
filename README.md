# olmochem

Minimal library for molecular property prediction with OLMo-7B.

## Datasets supported

- **Classification**: Binary, multilabel, multitask (BBBP, BACE, HIV, ClinTox, SIDER, Tox21)
- **Regression**: ESOL, FreeSolv, Lipophilicity, Clearance
- **Pretraining**: Causal LM on ZINC20, PubChem
- **Instruction Tuning**: USPTO reaction prediction

## Training strategies

- **For efficient Training**: QLoRA (4-bit) or full finetuning
- **Multi-GPU**: DDP support out of the box

## Installation

```bash
pip install -e .
```

## Quick Start

### Step 1: Prepare Data

```bash
# Prepare a specific dataset (uses DeepChem scaffold splits)
!python3 scripts/prepare_data.py \
        --split_type 'deepchem' \
        --datasets 'bbbp' \
        --data_dir olmo/datasets/deepchem_splits

```

### Step 2: Train

### Classification

```bash

!python scripts/train_classification.py --task 'bbbp' --use_lm_head --epochs 1 --data_dir datasets/deepchem_splits

```

### Regression

```bash
python scripts/train_regression.py --task clearance
python scripts/train_regression.py --task esol
```

### Pretraining

```bash
python scripts/pretrain.py \
    --dataset zinc20 \
    --num_samples 10000
```

### Instruction Tuning

```bash
python \
    --dataset OpenMol/USPTO_1k_TPL-SFT \
    --num_samples 10000
```

## Adding New Tasks

Adding a new dataset is simple:

```python
# In olmochem/tasks/classification.py
from .base import TaskConfig, register_task

register_task(TaskConfig(
    name="clintox",
    task_columns=["CT_TOX"],
    prompt="Is this molecule clinically toxic?",
    task_type="binary",
    monitor_metric="val/roc_auc",
    monitor_mode="max",
))
```

Then train:

```bash
python scripts/train_classification.py --task my_dataset
```

## Repository Structure

```
olmochem/
├── olmochem/                 # Library
│   ├── model.py              # Model wrappers
│   ├── data.py               # Dataset classes
│   ├── trainer.py            # Lightning modules
│   ├── callbacks.py          # MLflow callback
│   ├── utils.py              # Rank-aware utilities
│   └── tasks/                # Task registry
│       ├── classification.py # Classification tasks
│       ├── regression.py     # Regression tasks
│       └── generation.py     # Pretraining tasks
├── scripts/                  # Entry points
│   ├── prepare_data.py       # Data preparation (run first)
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── pretrain.py
│   └── train_instruction.py
└── speedrun.sh               # Pipeline documentation
```

## Data Preparation

`prepare_data.py` uses DeepChem to download datasets and create scaffold splits:

```bash
python scripts/prepare_data.py --datasets bbbp bace hiv
```

This creates CSVs in `datasets/deepchem_splits/{task}/train.csv`, `datasets/deepchem_splits/valid.csv`, `datasets/deepchem_splits/test.csv` with:
- `smiles`: SMILES string column
- Task-specific label columns (see task configs)

Requires `deepchem` (`pip install --pre deepchem`).

## License

MIT
