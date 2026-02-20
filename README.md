# ChemBERTa4

Minimal library for molecular property prediction with [OLMo-7B](https://huggingface.co/allenai/OLMo-7B-hf).

Supports classification (single_task, multi_task), regression, causal LM pretraining on SMILES, and instruction tuning. Training uses QLoRA (4-bit) by default and automatically scales across all available GPUs via PyTorch Lightning DDP. New datasets can be added through a task registry in ~5 lines. Experiments are tracked with wandb.

## Acknowledgements

Thanks to Saurav and Arjit for openly sharing their work. Parts of this repository are inspired by their approach.

## Supported Datasets

| Category | Datasets | Task Type |
|---|---|---|
| **Classification** | BBBP, BACE, HIV, ClinTox | `single_task` |
| **Classification** | SIDER (27 side-effect labels) | `multi_task` |
| **Classification** | Tox21 (12 toxicity assays) | `multi_task` |
| **Regression** | Delaney (ESOL), FreeSolv, Lipophilicity, Clearance, BACE | Continuous |
| **Pretraining** | ZINC20, PubChem | Causal LM |
| **Instruction Tuning** | USPTO | Reaction prediction |

## Requirements

- Python >= 3.9
- CUDA-capable GPU (16 GB+ VRAM recommended for QLoRA)

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd olmo-1

# Install dependencies
pip install -r requirements.txt

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

</details>

## Quick Start

### 1. Prepare data

Download MoleculeNet datasets and create scaffold splits:

```bash
# Single dataset
python prepare_data.py \
    --split_type deepchem \
    --datasets bbbp \
    --data_dir datasets/deepchem_splits

# Multiple datasets at once
python prepare_data.py \
    --split_type deepchem \
    --datasets bbbp bace hiv clintox tox21 sider delaney freesolv lipophilicity \
    --data_dir datasets/deepchem_splits
```

This creates `train.csv`, `valid.csv`, and `test.csv` under `datasets/deepchem_splits/<task>/`, each containing a `smiles` column and task-specific label columns.

### 2. Train

All experiments are launched via `run_experiment.py`. Pass `--datasets` followed by one or more dataset names (see `configs/tasks.yaml` for the full list). Defaults for each task type live in `configs/`.

```bash
# Single dataset — QLoRA by default, scales across all available GPUs
python run_experiment.py --datasets bbbp

# Multiple datasets in sequence (can mix task types)
python run_experiment.py --datasets bbbp clearance zinc20

# Override defaults
python run_experiment.py --datasets bbbp --lr 0.001 --epochs 20 --batch_size 8
```

**Classification** (single_task or multi_task):

```bash
# QLoRA (default)
python run_experiment.py --datasets bbbp

# LM-head approach (Yes/No token prediction instead of a linear head)
python run_experiment.py --datasets bace_classification --use_lm_head

# LoRA without 4-bit quantization
python run_experiment.py --datasets bbbp --finetune_strategy lora

# Full fine-tuning (no LoRA)
python run_experiment.py --datasets clintox --finetune_strategy full_finetune --lr 1e-5
```

**Regression**:

```bash
python run_experiment.py --datasets clearance --epochs 30
python run_experiment.py --datasets delaney
```

**Pretraining** (causal LM on SMILES):

```bash
python run_experiment.py --datasets zinc20 --num_samples 1000000 --epochs 1
```

**Instruction tuning** (reaction prediction):

```bash
python run_experiment.py --datasets uspto --num_samples 10000
```

Training automatically uses DDP across every available GPU (`devices=-1, strategy="ddp"` in PyTorch Lightning). No `torchrun` wrapper needed.

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

### LoRA

LoRA without quantization — lower memory reduction than QLoRA, but no 4-bit precision loss.

```bash
python run_experiment.py --datasets bbbp --finetune_strategy lora
```

### Full fine-tuning

Disable LoRA and train all parameters. Requires significantly more VRAM.

```bash
python run_experiment.py --datasets bbbp --finetune_strategy full_finetune --lr 1e-5
```

### Finetune strategy summary

| Strategy | Flag | Description |
|---|---|---|
| QLoRA | `--finetune_strategy qlora` *(default)* | 4-bit NF4 quantization + LoRA |
| LoRA | `--finetune_strategy lora` | LoRA adapters, no quantization |
| Full fine-tuning | `--finetune_strategy full_finetune` | All parameters trainable |

### Classification head options

| Mode | Flag | Description |
|---|---|---|
| Linear head | *(default)* | Last-token pooling + linear projection |
| LM head | `--use_lm_head` | Extracts Yes/No logits from the pretrained LM head |

## Full Pipeline: Pretrain &#8594; Instruct &#8594; Fine-tune

For best results, chain the stages:

```bash
# Stage 1: Pretrain on SMILES
python run_experiment.py \
    --datasets zinc20 \
    --num_samples 10000000 \
    --hub_name youruser/OLMo-7B-ZINC20

# Stage 2: Instruction-tune on reactions
python run_experiment.py \
    --datasets uspto \
    --model_name youruser/OLMo-7B-ZINC20 \
    --hub_name youruser/OLMo-7B-ZINC-USPTO

# Stage 3: Fine-tune on downstream task
python run_experiment.py \
    --datasets bbbp \
    --model_name youruser/OLMo-7B-ZINC-USPTO
```

## Adding a New Task

Add an entry to `configs/tasks.yaml`:

```yaml
# configs/tasks.yaml

my_dataset:
  task_columns: [label]
  prompt: "Is this molecule active?"
  task_type: single_task
  experiment_type: classification
  monitor_metric: val/roc_auc
  monitor_mode: max
```

Then prepare your data (a CSV with `smiles` and label columns) and train:

```bash
python run_experiment.py --datasets my_dataset --data_dir path/to/splits
```

## Repository Structure

```
olmo-1/
├── chemberta4/                  # Core library
│   ├── model.py               # ClassificationHead, CausalLMClassificationHead, RegressionHead
│   ├── data.py                # MoleculeNetDataset, PretrainingDataset, InstructionDataset
│   ├── trainer.py             # Lightning modules (OLMoClassifier, OLMoRegressor, OLMoPretrainer)
│   ├── callbacks.py           # Wandb logging callback
│   └── utils.py               # Rank-aware utilities for DDP
├── configs/                    # Default hyperparameters and task registry
│   ├── tasks.yaml             # All supported datasets and their configs
│   ├── classification.yaml    # Default classification hyperparameters
│   ├── regression.yaml        # Default regression hyperparameters
│   ├── pretrain.yaml          # Default pretraining hyperparameters
│   └── instruction.yaml       # Default instruction tuning hyperparameters
├── training/                   # Training functions (called by run_experiment.py)
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── pretrain.py
│   └── train_instruction.py
├── run_experiment.py           # Unified entry point for all experiments
├── prepare_data.py             # Download & scaffold-split datasets
├── requirements.txt
└── README.md
```

## Metrics & Experiment Tracking

Training metrics are logged via wandb:

| Task Type | Metrics |
|---|---|
| Classification | Accuracy, ROC-AUC |
| Regression | RMSE, MAE (denormalized) |

Early stopping (patience = 7) and model checkpointing are enabled by default, monitored on the validation metric defined in each task config.

## License

MIT
