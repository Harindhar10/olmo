#!/usr/bin/env python3
"""
Classification finetuning on MoleculeNet datasets.

Examples:
    # Single GPU with QLoRA
    python scripts/train_classification.py --task bbbp --use_qlora

    # Multi-GPU
    torchrun --nproc_per_node=4 scripts/train_classification.py --task hiv

    # LM head approach (Yes/No prediction)
    python scripts/train_classification.py --task bace --use_lm_head

    # Full finetuning (no LoRA)
    python scripts/train_classification.py --task hiv --full_finetune --lr 1e-5
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent to path for local development
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
from olmochem.trainer import OLMoClassifier
from olmochem.utils import is_main_process, print0, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="OLMo Classification Fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Task Selection ----
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list_tasks(),
        help="MoleculeNet task name",
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
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--use_lm_head",
        action="store_true",
        help="Use Yes/No prediction instead of classification head",
    )
    parser.add_argument(
        "--use_qlora",
        default=True,
        help="Use 4-bit quantized LoRA (default: True)",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Full finetuning without LoRA",
    )

    # ---- Training ----
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument(
        "--gradient_accum", type=int, default=24, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=15, help="Max epochs")
    parser.add_argument(
        "--patience", type=int, default=7, help="Early stopping patience"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="Warmup ratio"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )

    # ---- LoRA ----
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # ---- Infrastructure ----
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--mlflow_uri", type=str, default="./mlruns", help="MLflow tracking URI"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    # Get task config
    task_config = get_task(args.task)
    print0(f"Task: {args.task} ({task_config.task_type})")
    print0(f"Columns: {task_config.task_columns[:3]}{'...' if len(task_config.task_columns) > 3 else ''}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_df = pd.read_csv(f"{args.data_dir}/{args.task}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/{args.task}/valid.csv")
    test_df = pd.read_csv(f"{args.data_dir}/{args.task}/test.csv")

    # Create datasets
    train_ds = MoleculeDataset(
        train_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        args.max_len,
        args.use_lm_head,
    )
    val_ds = MoleculeDataset(
        val_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        args.max_len,
        args.use_lm_head,
    )
    test_ds = MoleculeDataset(
        test_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        args.max_len,
        args.use_lm_head,
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
    model = OLMoClassifier(
        model_name=args.model_name,
        num_tasks=task_config.num_tasks,
        task_type=task_config.task_type,
        use_lm_head=args.use_lm_head,
        use_qlora=use_qlora,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
            filename="best",
            monitor=task_config.monitor_metric,
            mode=task_config.monitor_mode,
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        MLflowCallback(),
    ]

    # MLflow (rank 0 only)
    if is_main_process():
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(f"olmochem-{args.task}")
        mlflow.start_run(run_name=f"{args.task}_{timestamp}")
        mlflow.log_params(vars(args))
        mlflow.log_params({
            "task_type": task_config.task_type,
            "num_tasks": task_config.num_tasks,
            "effective_batch_size": args.batch_size * args.gradient_accum,
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
        logger=False,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    print0("Starting training...")
    print0(f"Model approach: {'LM Head (Yes/No)' if args.use_lm_head else 'Classification Head'}")
    print0(f"LoRA: {use_qlora}")
    trainer.fit(model, train_loader, val_loader)

    # Test
    print0("\nRunning test evaluation...")
    trainer.test(model, test_loader)

    # Finalize MLflow
    if is_main_process():
        for key, value in trainer.callback_metrics.items():
            metric_name = f"final_{key.replace('/', '_')}"
            mlflow.log_metric(metric_name, float(value))

        checkpoint_callback = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
        if checkpoint_callback.best_model_path:
            mlflow.log_artifact(checkpoint_callback.best_model_path)

        mlflow.end_run()
        print0(f"\nDone! Best {task_config.monitor_metric}: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
