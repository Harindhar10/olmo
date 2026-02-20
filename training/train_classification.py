import gc
from datetime import datetime
from types import SimpleNamespace

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

from chemberta4.callbacks import WandbCallback
from chemberta4.data import MoleculeNetDataset
from chemberta4.trainer import OLMoClassifier
from chemberta4.utils import get_task, is_main_process, log0


def run_classification_experiment(args: SimpleNamespace, task_name: str) -> None:
    """Run training and evaluation on a MoleculeNet classification dataset.

    Parameters
    ----------
    args : SimpleNamespace
        Training arguments (model, data, optimizer, and logging settings).
    task_name : str
        Name of the MoleculeNet dataset to run the experiment on.
    """
    # Get task config
    task_config = get_task(task_name)
    log0(f"Task: {task_name} ({task_config.task_type})")
    log0(f"Columns: {task_config.task_columns[:3]}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_df = pd.read_csv(f"{args.data_dir}/{task_name}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/{task_name}/valid.csv")
    test_df = pd.read_csv(f"{args.data_dir}/{task_name}/test.csv")

    # Create datasets
    train_ds = MoleculeNetDataset(
        train_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
        args.use_lm_head,
    )
    val_ds = MoleculeNetDataset(
        val_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
        args.use_lm_head,
    )
    test_ds = MoleculeNetDataset(
        test_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
        args.use_lm_head,
    )

    log0(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

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
    model = OLMoClassifier(
        model_name=args.model_name,
        num_tasks=len(task_config.task_columns),
        task_type=task_config.task_type,
        use_lm_head=args.use_lm_head,
        finetune_strategy=args.finetune_strategy,
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
        LearningRateMonitor(logging_interval="step"),
        *([ WandbCallback() ] if args.wandb else []),
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
    if is_main_process() and args.wandb:
        import wandb

        if args.wandb_key:
            wandb.login(key=args.wandb_key)

        project_name = args.wandb_project or f"chemberta4-{task_name}"
        log_params = {k: v for k, v in vars(args).items() if k != "wandb_key"}
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
            "num_tasks": len(task_config.task_columns),
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
    log0("Starting training...")
    log0(f"Model approach: {'LM Head (Yes/No Token)' if args.use_lm_head else 'Classification Head'}")
    log0(f"Finetune strategy: {args.finetune_strategy}")
    trainer.fit(model, train_loader, val_loader)

    # Test
    log0("Running test evaluation...")
    trainer.test(model, test_loader)

    # Finalize tracker
    if is_main_process() and args.wandb:
        import wandb

        for key, value in trainer.callback_metrics.items():
            metric_name = f"final_{key.replace('/', '_')}"
            wandb.log({metric_name: float(value)})
        wandb.finish()

        checkpoint_callback = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
        log0(f"Done! Best RMSE: {checkpoint_callback.best_model_score:.4f}")

        if args.delete_checkpoint:
            import shutil

            checkpoint_dir = f"{args.output_dir}/{task_name}/{timestamp}"
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            log0(f"Deleted checkpoint directory: {checkpoint_dir}")

    # Cleanup GPU memory for next task
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
