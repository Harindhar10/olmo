# olmochem

Minimal library for molecular property prediction with OLMo-7B.

Inspired by [nanochat](https://github.com/karpathy/nanochat) - explicit over implicit, minimal abstraction.

## Features

- **Classification**: Binary, multilabel, multitask (BBBP, BACE, HIV, ClinTox, SIDER, Tox21)
- **Regression**: ESOL, FreeSolv, Lipophilicity, Clearance
- **Pretraining**: Causal LM on ZINC20, PubChem
- **Instruction Tuning**: USPTO reaction prediction
- **Efficient Training**: QLoRA (4-bit) or full finetuning
- **Multi-GPU**: DDP support out of the box

## Installation

```bash
pip install -e .
```

## Quick Start

### Step 1: Prepare Data

```bash
# Prepare a specific dataset (uses DeepChem scaffold splits)
python scripts/prepare_data.py --datasets bbbp

# Prepare all registered datasets
python scripts/prepare_data.py --all

# Custom SMILES length filter
python scripts/prepare_data.py --datasets bbbp bace hiv --max_smiles_len 150
```

### Step 2: Train

### Classification

```bash
# Single GPU
python scripts/train_classification.py --task bbbp --use_qlora

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train_classification.py --task hiv

# LM head approach (Yes/No prediction)
python scripts/train_classification.py --task bace --use_lm_head
```

### Regression

```bash
python scripts/train_regression.py --task clearance
python scripts/train_regression.py --task esol
```

### Pretraining

```bash
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --dataset zinc20 \
    --num_samples 1000000
```

### Instruction Tuning

```bash
torchrun --nproc_per_node=4 scripts/train_instruction.py \
    --dataset OpenMol/USPTO_1k_TPL-SFT \
    --num_samples 10000
```

## Adding New Tasks

Adding a new dataset is simple:

```python
# In olmochem/tasks/classification.py
from .base import TaskConfig, register_task

register_task(TaskConfig(
    name="my_dataset",
    task_columns=["activity"],
    prompt="Is this molecule active?",
    task_type="binary",
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

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Task registry | Easy to add new datasets |
| QLoRA default | Memory efficient |
| Two classification modes | Classification head vs LM head (Yes/No) |
| Rank-aware utilities | Clean DDP support |
| No config files | CLI args with sensible defaults |

## Data Preparation

`prepare_data.py` uses DeepChem to download datasets and create scaffold splits:

```bash
python scripts/prepare_data.py --datasets bbbp bace hiv
```

This creates CSVs in `splits/{task}/train.csv`, `valid.csv`, `test.csv` with:
- `smiles`: SMILES string column
- Task-specific label columns (see task configs)

Requires `deepchem` (`pip install deepchem`).

## License

MIT
