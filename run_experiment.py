#!/usr/bin/env python3
"""
Unified experiment runner for all task types.

Users are expected to run this file to start training experiments. Supported datasets
and their configs are defined in chemberta4/configs/tasks.yaml. 
To run this file, --datasets is the only required argument. The other arguments have defualts 
stored in chemberta4/configs. Each type of task (classification/regression/pretrain/instruction)
has its configs defined in chemberta4/configs and loaded based on the dataset_name.

If the default arguments need to be overridden, it can be done 
by passing needed arguments while running this file.

Examples:
    # Single classification task
    python run_experiment.py bbbp

    # Multiple datasets (can mix types)
    python run_experiment.py bbbp clearance zinc20

    # Override defaults
    python run_experiment.py bbbp --lr 0.001 --epochs 20 --batch_size 8

    # Pretraining with custom settings
    python run_experiment.py zinc20 --num_samples 50000 --hub_name my-org/my-model
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl

from chemberta4.utils import get_task, prepare_config, print0, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with 'None' defaults for all optional fields
        (overrides are merged with task-specific config in 'prepare_config').
    """
    parser = argparse.ArgumentParser(description="Run experiments on molecular datasets")

    # Required
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset name(s) from tasks.yaml")

    # Model
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--finetune_strategy", type=str, choices=["qlora", "lora", "full_finetune"], default=None,
                        help="Finetuning strategy: qlora (4-bit quantized LoRA), lora (LoRA without quantization), full_finetune (all parameters)")
    parser.add_argument("--use_lm_head", action="store_true", default=None)

    # Training
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)

    # Infrastructure
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--delete_checkpoint", action="store_true", default=None)
    parser.add_argument("--wandb", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_key", type=str, default=None)

    # Classification/Regression specific
    parser.add_argument("--data_dir", type=str, default=None)

    # Pretraining/Instruction specific
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--smiles_column", type=str, default=None)
    parser.add_argument("--hub_name", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--val_check_interval", type=int, default=None)

    args = parser.parse_args()
    return args


def main() -> None:
    """Entry point: iterate over requested datasets and dispatch to the appropriate trainer."""
    args = parse_args()

    for dataset_name in args.datasets:
        task = get_task(dataset_name)
        config = prepare_config(args, task)

        # Set seed
        seed = config.seed
        set_seed(seed)
        pl.seed_everything(seed, workers=True)

        print0(f"\n{'='*60}")
        print0(f"Running {task.experiment_type} experiment: {dataset_name}")
        print0(f"{'='*60}")

        if task.experiment_type == "classification":
            from training.train_classification import run_classification_experiment
            run_classification_experiment(config, dataset_name)

        elif task.experiment_type == "regression":
            from training.train_regression import run_regression_experiment
            run_regression_experiment(config, dataset_name)

        elif task.experiment_type == "pretraining":
            from training.pretrain import run_pretraining_experiment
            run_pretraining_experiment(config, dataset_name)

        elif task.experiment_type == "instruction":
            from training.train_instruction import run_instruction_experiment
            run_instruction_experiment(config, dataset_name)

        else:
            raise ValueError(f"Unknown experiment type '{task.experiment_type}' for dataset '{dataset_name}'")

    print0("\nAll experiments complete!")


if __name__ == "__main__":
    main()
