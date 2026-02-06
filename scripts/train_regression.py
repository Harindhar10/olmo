#!/usr/bin/env python3
"""
Regression finetuning on MoleculeNet datasets.

Examples:
    # Single GPU with QLoRA
    python scripts/train_regression.py --task clearance

    # Multi-GPU
    torchrun --nproc_per_node=4 scripts/train_regression.py --task esol

    # Full finetuning
    python scripts/train_regression.py --task lipophilicity --full_finetune --lr 1e-5
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from olmochem.callbacks import MLflowCallback
from olmochem.data import MoleculeDataset
from olmochem.tasks import get_task, list_tasks
from olmochem.trainer import OLMoRegressor
from olmochem.utils import is_main_process, print0, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="OLMo Regression Fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Task Selection ----
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[t for t in list_tasks() if get_task(t).task_type == "regression"],
        help="Regression task name",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="olmo/datasets/deepchem_splits",
        help="Directory containing dataset splits",
    )

    # ---- Model ----
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/OLMo-7B-hf",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--use_qlora",
        default=True,
        help="Use 4-bit quantized LoRA",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Full finetuning without LoRA",
    )

    # ---- Training ----
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument(
        "--gradient_accum", type=int, default=2, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

    # ---- LoRA ----
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # ---- Infrastructure ----
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output dir")
    parser.add_argument("--mlflow_uri", type=str, default="./mlruns", help="MLflow URI")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    # Get task config
    task_config = get_task(args.task)
    assert task_config.task_type == "regression", f"Task {args.task} is not a regression task"

    print0(f"Task: {args.task}")
    print0(f"Target column: {task_config.target_column}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_df = pd.read_csv(f"{args.data_dir}/{args.task}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/{args.task}/valid.csv")
    test_df = pd.read_csv(f"{args.data_dir}/{args.task}/test.csv")

    # Create training dataset (computes normalization stats)
    train_ds = MoleculeDataset(
        train_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        args.max_len,
    )

    # Get normalization stats from training set
    label_stats = train_ds.get_label_stats()
    print0(f"Label normalization - Mean: {label_stats['mean']:.4f}, Std: {label_stats['std']:.4f}")

    # Create val/test datasets with training stats
    val_ds = MoleculeDataset(
        val_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        args.max_len,
        label_stats=label_stats,
    )
    test_ds = MoleculeDataset(
        test_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        args.max_len,
        label_stats=label_stats,
    )

    print0(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # DataLoaders
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # Model
    use_qlora = args.use_qlora and not args.full_finetune
    model = OLMoRegressor(
        model_name=args.model_name,
        use_qlora=use_qlora,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        label_mean=label_stats["mean"],
        label_std=label_stats["std"],
    )

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        EarlyStopping(
            monitor=task_config.monitor_metric,
            patience=args.patience,
            mode=task_config.monitor_mode,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=f"{args.output_dir}/{args.task}/{timestamp}",
            filename="best-{val/rmse:.4f}",
            monitor=task_config.monitor_metric,
            mode=task_config.monitor_mode,
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        MLflowCallback(),
    ]

    # MLflow
    if is_main_process():
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(f"olmochem-{args.task}")
        mlflow.start_run(run_name=f"{args.task}_{timestamp}")
        mlflow.log_params(vars(args))
        mlflow.log_params({
            "label_mean": label_stats["mean"],
            "label_std": label_stats["std"],
        })

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        precision="bf16-mixed",
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accum,
        callbacks=callbacks,
        logger=True,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
    )

    # Train
    print0("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test
    print0("\nRunning test evaluation...")
    trainer.test(model, test_loader)

    # Finalize
    if is_main_process():
        for key, value in trainer.callback_metrics.items():
            mlflow.log_metric(f"final_{key.replace('/', '_')}", float(value))

        checkpoint_callback = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
        if checkpoint_callback.best_model_path:
            mlflow.log_artifact(checkpoint_callback.best_model_path)

        mlflow.end_run()
        print0(f"\nDone! Best RMSE: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
