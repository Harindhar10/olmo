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

from chemberta4.data import MoleculeNetDataset
from chemberta4.trainer import OLMoRegressor
from chemberta4.utils import get_task, is_main_process, log0

from functools import partial

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchmetrics import Accuracy, AUROC

from transformers.models.olmo.modeling_olmo import OlmoDecoderLayer


def run_regression_experiment(args: SimpleNamespace, task_name: str) -> None:
    """Run training and evaluation on a MoleculeNet regression dataset.

    Parameters
    ----------
    args : SimpleNamespace
        Training arguments (model, data, optimizer, and logging settings).
    task_name : str
        Name of the MoleculeNet dataset to run the experiment on.
    """
    # Get task config
    task_config = get_task(task_name)
    assert task_config.experiment_type == "regression", f"Task {task_name} is not a regression task"

    log0(f"Task: {task_name}")
    log0(f"Target column: {task_config.target_column}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load data
    train_df = pd.read_csv(f"{args.data_dir}/{task_name}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/{task_name}/valid.csv")
    test_df = pd.read_csv(f"{args.data_dir}/{task_name}/test.csv")

    use_lm_head = getattr(args, "use_lm_head", False)

    train_ds = MoleculeNetDataset(
        train_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
        use_lm_head=use_lm_head,
    )
    val_ds = MoleculeNetDataset(
        val_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
        use_lm_head=use_lm_head,
    )
    test_ds = MoleculeNetDataset(
        test_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
        use_lm_head=use_lm_head,
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
    model = OLMoRegressor(
        model_name=args.model_name,
        finetune_strategy=args.finetune_strategy,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_lm_head=use_lm_head,
    )

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor=task_config.monitor_metric,
            mode=task_config.monitor_mode,
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ),
    ]

        # EarlyStopping(
        #     monitor=task_config.monitor_metric,
        #     patience=args.patience,
        #     mode=task_config.monitor_mode,
        #     verbose=True,
        # ),
    # ----------------------------
    # W&B Setup
    # ----------------------------
    wandb_logger = None
    if args.wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        wandb_logger = WandbLogger(
            project=args.wandb_project or f"chemberta4-{task_name}",
            log_model=False,
            config=vars(args),
            notes=args.wandb_notes if hasattr(args, 'wandb_notes') else None,
        )

    # ----------------------------
    # FSDP Strategy
    # ----------------------------
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={OlmoDecoderLayer}
    )

    
    fsdp_strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing_policy=auto_wrap_policy,
        cpu_offload=True,
        use_orig_params=True,
        sync_module_states=True,
    )


    num_devices = torch.cuda.device_count() or 1

    # Trainer
    # trainer = pl.Trainer(
    #     max_epochs=args.epochs,
    #     accelerator="gpu",
    #     devices=-1,
    #     strategy="ddp",
    #     precision="16-mixed",
    #     gradient_clip_val=args.max_grad_norm,
    #     accumulate_grad_batches=args.gradient_accum,
    #     callbacks=callbacks,
    #     logger=True,
    #     log_every_n_steps=10,
    #     deterministic=True,
    #     enable_progress_bar=True,
    # )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=fsdp_strategy,
        precision="bf16-mixed",
        max_epochs=args.epochs,
        accumulate_grad_batches=args.gradient_accum,
        log_every_n_steps=1,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        enable_progress_bar=True,
        enable_model_summary=True
        )

    # Train
    log0("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    # Test
    #log0("Running test evaluation...")
    # trainer.test(model, test_loader)


    checkpoint_callback = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
    log0(f"Done! Best validation ROC AUC: {checkpoint_callback.best_model_score:.4f}")

    if args.delete_checkpoint:
        import shutil

        checkpoint_dir = f"{args.output_dir}/{task_name}/{timestamp}"
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        log0(f"Deleted checkpoint directory: {checkpoint_dir}")

    # Cleanup GPU memory for next task
    del model

    model = OLMoRegressor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    test_results = trainer.test(model, test_loader)

    if args.wandb and test_results:
        import wandb
        if wandb.run is not None:
            for key, value in test_results[0].items():
                wandb.run.summary[key.replace("/", "_")] = value
            wandb.finish()

    if args.delete_checkpoint:
        import shutil

        checkpoint_dir = f"{args.output_dir}/{task_name}/{timestamp}"
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        log0(f"Deleted checkpoint directory: {checkpoint_dir}")

    del trainer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

