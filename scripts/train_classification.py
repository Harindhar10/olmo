#!/usr/bin/env python3
"""
Classification finetuning on MoleculeNet datasets.

Examples:
    # Single/Multi GPU with QLoRA
    python scripts/train_classification.py --tasks bbbp --use_qlora

    # Multiple tasks (runs sequentially)
    python scripts/train_classification.py --tasks bbbp bace hiv

    # LM head approach (Yes/No prediction)
    python scripts/train_classification.py --tasks bace --use_lm_head

    # Full finetuning (no LoRA)
    python scripts/train_classification.py --tasks hiv --full_finetune --lr 1e-5
"""

import argparse
import gc
import os
import sys
from datetime import datetime

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from chemberta4.callbacks import MLflowCallback, WandbCallback
from chemberta4.data import MoleculeDataset
from chemberta4.tasks import get_task, list_tasks
from chemberta4.trainer import OLMoClassifier
from chemberta4.utils import is_main_process, print0, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="OLMo Classification Fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Task Selection ----
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        choices=list_tasks(),
        help="MoleculeNet task name(s)",
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
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (flash_attention_2, sdpa, or eager)",
    )

    # ---- Training ----
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument(
        "--gradient_accum", type=int, default=4, help="Gradient accumulation steps"
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
        "--delete_checkpoint",
        action="store_true",
        help="Delete saved checkpoint after test evaluation",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="mlflow",
        choices=["mlflow", "wandb"],
        help="Experiment tracker to use",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI (used when --tracker=mlflow)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (used when --tracker=wandb, default: chemberta4-{task_name})",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team name)",
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help="W&B API key (optional, can also use WANDB_API_KEY env var)",
    )

    return parser.parse_args()


def run_task(args, task_name):
    """Run training and evaluation for a single task."""
    # Get task config
    task_config = get_task(task_name)
    print0(f"\nTask: {task_name} ({task_config.task_type})")
    print0(f"Columns: {task_config.task_columns[:3]}{'...' if len(task_config.task_columns) > 3 else ''}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_df = pd.read_csv(f"{args.data_dir}/{task_name}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/{task_name}/valid.csv")
    test_df = pd.read_csv(f"{args.data_dir}/{task_name}/test.csv")

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
        attn_implementation=args.attn_implementation,
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
        LearningRateMonitor(logging_interval="step"),
        MLflowCallback() if args.tracker == "mlflow" else WandbCallback(),
        ModelCheckpoint(
            dirpath=f"{args.output_dir}/{task_name}/{timestamp}",
            filename="best",
            monitor=task_config.monitor_metric,
            mode=task_config.monitor_mode,
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ),
    ]

    num_devices = torch.cuda.device_count() or 1

    # Tracker init (rank 0 only)
    if is_main_process():
        if args.tracker == "mlflow":
            import mlflow

            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(f"chemberta4-{task_name}")
            mlflow.start_run(run_name=f"{task_name}_{timestamp}")
            log_params = {k: v for k, v in vars(args).items() if k != "wandb_key"}
            mlflow.log_params(log_params)
            mlflow.log_params({
                "task": task_name,
                "task_type": task_config.task_type,
                "num_tasks": task_config.num_tasks,
                "num_devices": num_devices,
                "effective_batch_size": args.batch_size * args.gradient_accum * num_devices,
            })
        elif args.tracker == "wandb":
            import wandb

            if args.wandb_key:
                wandb.login(key=args.wandb_key)

            project_name = args.wandb_project or f"chemberta4-{task_name}"
            log_params = {k: v for k, v in vars(args).items() if k not in ("wandb_key", "mlflow_uri")}
            wandb.init(
                entity=args.wandb_entity,
                project=project_name,
                name=f"{task_name}_{timestamp}",
                config=log_params,
                reinit=True,
            )
            wandb.config.update({
                "task": task_name,
                "task_type": task_config.task_type,
                "num_tasks": task_config.num_tasks,
                "num_devices": num_devices,
                "effective_batch_size": args.batch_size * args.gradient_accum * num_devices,
            })

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        precision="16-mixed",
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accum,
        callbacks=callbacks,
        logger=True,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    print0("Starting training...")
    print0(f"Model approach: {'LM Head (Yes/No Token)' if args.use_lm_head else 'Classification Head'}")
    print0(f"LoRA: {use_qlora}")
    trainer.fit(model, train_loader, val_loader)

    # Test
    print0("\nRunning test evaluation...")
    trainer.test(model, test_loader)

    # Finalize tracker
    if is_main_process():
        if args.tracker == "mlflow":
            import mlflow

            for key, value in trainer.callback_metrics.items():
                metric_name = f"final_{key.replace('/', '_')}"
                mlflow.log_metric(metric_name, float(value))
            mlflow.end_run()
        elif args.tracker == "wandb":
            import wandb

            for key, value in trainer.callback_metrics.items():
                metric_name = f"final_{key.replace('/', '_')}"
                wandb.log({metric_name: float(value)})
            wandb.finish()

        checkpoint_callback = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
        print0(f"\nDone! Best {task_config.monitor_metric}: {checkpoint_callback.best_model_score:.4f}")

        if args.delete_checkpoint:
            import shutil

            checkpoint_dir = f"{args.output_dir}/{task_name}/{timestamp}"
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            print0(f"Deleted checkpoint directory: {checkpoint_dir}")

    # Cleanup GPU memory for next task
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    print0(f"Running {len(args.tasks)} task(s): {', '.join(args.tasks)}")
    for task_name in args.tasks:
        run_task(args, task_name)


if __name__ == "__main__":
    main()
