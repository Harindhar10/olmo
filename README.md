# chemberta4

Minimal library for molecular property prediction with [OLMo-7B](https://huggingface.co/allenai/OLMo-7B-hf).

Supports classification (binary, multilabel, multitask), regression, causal LM pretraining on SMILES, and instruction tuning. Training uses QLoRA (4-bit) by default and automatically scales across all available GPUs via PyTorch Lightning DDP. New datasets can be added through a task registry in ~5 lines. Experiments are tracked with MLflow.

## Supported Datasets

| Category | Datasets | Task Type |
|---|---|---|
| **Classification** | BBBP, BACE, HIV, ClinTox | Binary |
| **Classification** | SIDER (27 side-effect labels) | Multilabel |
| **Classification** | Tox21 (12 toxicity assays) | Multitask |
| **Regression** | Delaney (ESOL), FreeSolv, Lipophilicity, Clearance, BACE | Continuous |
| **Pretraining** | ZINC20, PubChem | Causal LM |
| **Instruction Tuning** | USPTO (via OpenMol) | Reaction prediction |

## Requirements

- Python >= 3.9
- CUDA-capable GPU (16 GB+ VRAM recommended for QLoRA)

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd olmo-1

# Install in editable mode
pip install -e .

# (Optional) Install DeepChem for data preparation
pip install --pre deepchem
```

<details>
<summary>Core dependencies (installed automatically)</summary>

| Package | Purpose |
|---|---|
| `torch >= 2.0` | Deep learning framework |
| `pytorch-lightning >= 2.0` | Training loop & DDP |
| `transformers >= 4.35` | OLMo model & tokenizer |
| `peft >= 0.6` | LoRA adapters |
| `bitsandbytes >= 0.41` | 4-bit quantization |
| `datasets >= 2.14` | HuggingFace dataset loading |
| `pandas >= 1.5` | Data manipulation |
| `numpy >= 1.24` | Numerical utilities |
| `torchmetrics >= 1.0` | Metric computation |
| `mlflow >= 2.8` | Experiment tracking |

</details>

## Quick Start

### 1. Prepare data

Download MoleculeNet datasets and create scaffold splits:

```bash
# Single dataset
python scripts/prepare_data.py \
    --split_type deepchem \
    --datasets bbbp \
    --data_dir datasets/deepchem_splits

# Multiple datasets at once
python scripts/prepare_data.py \
    --split_type deepchem \
    --datasets bbbp bace hiv clintox tox21 sider delaney freesolv lipophilicity \
    --data_dir datasets/deepchem_splits
```

This creates `train.csv`, `valid.csv`, and `test.csv` under `datasets/deepchem_splits/<task>/`, each containing a `smiles` column and task-specific label columns.

### 2. Train

**Classification** (binary, multilabel, or multitask):

```bash
# QLoRA (default) — automatically scales across all available GPUs
python scripts/train_classification.py \
    --task bbbp \
    --use_qlora \
    --batch_size 4 \
    --gradient_accum 24 \
    --epochs 15

# LM-head approach (Yes/No token prediction instead of a linear head)
python scripts/train_classification.py \
    --task bace \
    --use_lm_head \
    --use_qlora

# Full fine-tuning (no LoRA)
python scripts/train_classification.py \
    --task clintox \
    --full_finetune \
    --lr 1e-5 \
    --batch_size 2
```

**Regression**:

```bash
python scripts/train_regression.py --task clearance --use_qlora --epochs 30
python scripts/train_regression.py --task esol --use_qlora
```

**Pretraining** (causal LM on SMILES):

```bash
python scripts/pretrain.py \
    --dataset zinc20 \
    --num_samples 1000000 \
    --epochs 1
```

**Instruction tuning** (reaction prediction):

```bash
python scripts/train_instruction.py \
    --dataset OpenMol/USPTO_1k_TPL-SFT \
    --num_samples 10000
```

All training scripts automatically use DDP across every available GPU (`devices=-1, strategy="ddp"` in PyTorch Lightning). No `torchrun` wrapper needed.

## Training Strategies

### QLoRA (default)

4-bit NF4 quantization with LoRA adapters on attention projections (`q_proj`, `k_proj`, `v_proj`). Trains a 7B model in ~16 GB VRAM.

| Hyperparameter | Default |
|---|---|
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-4 |
| Gradient checkpointing | Enabled |

### Full fine-tuning

Disable LoRA and train all parameters. Requires significantly more VRAM.

```bash
python scripts/train_classification.py --task bbbp --full_finetune --lr 1e-5
```

### Classification head options

| Mode | Flag | Description |
|---|---|---|
| Linear head | *(default)* | Last-token pooling + linear projection |
| LM head | `--use_lm_head` | Extracts Yes/No logits from the pretrained LM head |

## Full Pipeline: Pretrain &#8594; Instruct &#8594; Fine-tune

For best results, chain the stages:

```bash
# Stage 1: Pretrain on SMILES
python scripts/pretrain.py \
    --dataset zinc20 \
    --num_samples 10000000 \
    --hub_name youruser/OLMo-7B-ZINC20

# Stage 2: Instruction-tune on reactions
python scripts/train_instruction.py \
    --model_name youruser/OLMo-7B-ZINC20 \
    --dataset OpenMol/USPTO_1k_TPL-SFT \
    --hub_name youruser/OLMo-7B-ZINC-USPTO

# Stage 3: Fine-tune on downstream task
python scripts/train_classification.py \
    --task bbbp \
    --model_name youruser/OLMo-7B-ZINC-USPTO
```

## Adding a New Task

Register a task in the appropriate file under `chemberta4/tasks/`:

```python
# chemberta4/tasks/classification.py

register_task(TaskConfig(
    name="bace_classification",
    task_columns=["Class"],
    prompt="Is this molecule a BACE-1 inhibitor?",
    task_type="binary",
    monitor_metric="val/roc_auc",
    monitor_mode="max",
))
```

Then prepare your data (a CSV with `smiles` and label columns) and train:

```bash
python scripts/train_classification.py --task my_dataset --data_dir path/to/splits
```

## Repository Structure

```
chemberta4/
├── chemberta4/                  # Core library
│   ├── model.py               # ClassificationHead, CausalLMClassificationHead, RegressionHead
│   ├── data.py                # MoleculeDataset, PretrainingDataset, InstructionDataset
│   ├── trainer.py             # Lightning modules (OLMoClassifier, OLMoRegressor, OLMoPretrainer)
│   ├── callbacks.py           # MLflow logging callback
│   ├── utils.py               # Rank-aware utilities for DDP
│   └── tasks/                 # Task registry
│       ├── base.py            # TaskConfig dataclass & register_task()
│       ├── classification.py  # BBBP, BACE, HIV, ClinTox, SIDER, Tox21
│       ├── regression.py      # ESOL, FreeSolv, Lipophilicity, Clearance
│       └── generation.py      # ZINC20, PubChem, USPTO
├── scripts/                   # CLI entry points
│   ├── prepare_data.py        # Download & scaffold-split datasets
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── pretrain.py
│   └── train_instruction.py
├── requirements.txt
└── README.md
```

## Metrics & Experiment Tracking

Training metrics are logged automatically via MLflow:

| Task Type | Metrics |
|---|---|
| Classification | Accuracy, ROC-AUC |
| Regression | RMSE, MAE (denormalized) |

Early stopping (patience = 7) and model checkpointing are enabled by default, monitored on the validation metric defined in each task config.

## License

MIT
