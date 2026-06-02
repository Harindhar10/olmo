import gc
import os
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

from chemberta4.data import MoleculeNetDataset, make_collate_fn
from chemberta4.trainer import OLMoRegressor
from chemberta4.utils import get_task, is_main_process, log0

from functools import partial

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchmetrics import Accuracy, AUROC


def _is_transformer_layer(module) -> bool:
    """Model-agnostic match for a transformer decoder block.

    Matches OLMo, Qwen, Llama, etc. (including trust_remote_code classes like
    Qwen3_5DecoderLayer) by class-name suffix, so the FSDP wrap / activation
    checkpointing policy follows whatever ``--model_name`` is used rather than
    being pinned to a single architecture.
    """
    return module.__class__.__name__.endswith("DecoderLayer")


class QLoRAAdapterCheckpoint(pl.Callback):
    """Save only the PEFT adapter and regression head for the best validation score."""

    def __init__(self, monitor: str, mode: str, dirpath: str):
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.monitor = monitor
        self.mode = mode
        self.best_model_score = None
        self.best_adapter_path = os.path.abspath(
            os.path.join(dirpath, "best_qlora_adapter")
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: OLMoRegressor) -> None:
        # DDP subprocesses can construct this callback at slightly different
        # times, which gives each rank a different timestamped output path.
        # Rank 0 owns the save path; every other rank must reuse it for eval.
        self.best_adapter_path = trainer.strategy.broadcast(self.best_adapter_path, 0)
        if trainer.is_global_zero:
            log0(f"QLoRA adapter checkpoint path: {self.best_adapter_path}")

    def _is_better(self, current: torch.Tensor) -> bool:
        if self.best_model_score is None:
            return True
        if self.mode == "min":
            return current < self.best_model_score
        return current > self.best_model_score

    def on_validation_end(self, trainer: pl.Trainer, pl_module: OLMoRegressor) -> None:
        if trainer.sanity_checking:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        current = current.detach().float().cpu()
        if not self._is_better(current):
            return

        self.best_model_score = current
        if trainer.is_global_zero:
            self._save_adapter(pl_module)
            log0(
                f"Saved best QLoRA adapter to {self.best_adapter_path} "
                f"({self.monitor}={current.item():.4f})"
            )
        trainer.strategy.barrier("qlora_adapter_checkpoint")

    def _save_adapter(self, pl_module: OLMoRegressor) -> None:
        os.makedirs(self.best_adapter_path, exist_ok=True)

        backbone = pl_module.model.backbone
        backbone.save_pretrained(self.best_adapter_path)

        if pl_module.tokenizer is not None:
            pl_module.tokenizer.save_pretrained(self.best_adapter_path)

        regressor_state = {
            name: tensor.detach().cpu()
            for name, tensor in pl_module.model.regressor.state_dict().items()
        }
        torch.save(
            {
                "regressor": regressor_state,
                "monitor": self.monitor,
                "mode": self.mode,
                "score": self.best_model_score.item(),
            },
            os.path.join(self.best_adapter_path, "regressor.pt"),
        )


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

    train_ds = MoleculeNetDataset(
        train_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
    )
    val_ds = MoleculeNetDataset(
        val_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
    )
    test_ds = MoleculeNetDataset(
        test_df,
        tokenizer,
        task_config.task_columns,
        task_config.prompt,
        task_config.task_type,
        task_config.experiment_type,
        args.max_len,
    )

    log0(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # DataLoaders
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": make_collate_fn(tokenizer, args.max_len),
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
    )

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = f"{args.output_dir}/{task_name}/{timestamp}"
    qlora_checkpoint = None
    if args.finetune_strategy == "qlora":
        qlora_checkpoint = QLoRAAdapterCheckpoint(
            monitor=task_config.monitor_metric,
            mode=task_config.monitor_mode,
            dirpath=run_output_dir,
        )
        checkpoint_callback = qlora_checkpoint
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor=task_config.monitor_metric,
            mode=task_config.monitor_mode,
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
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
        lambda_auto_wrap_policy,
        lambda_fn=_is_transformer_layer,
    )

    
    fsdp_strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        # No activation checkpointing: the qwen3_5 modeling code rebuilds its
        # causal mask with a doubled KV length on recompute, so its forward is
        # not reproducible under checkpointing. SMILES sequences are short, so
        # FULL_SHARD param sharding alone keeps memory in budget.
        cpu_offload=False,
        use_orig_params=True,
        sync_module_states=True,
    )


    if args.finetune_strategy == 'qlora':
        strategy = 'ddp'
    else:
        strategy = fsdp_strategy
        
    # Trainer
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=-1,
        strategy=strategy,
        # bf16-true casts the whole module to bf16 so FSDP's flatten group is
        # uniform (vs bf16-mixed which keeps fp32 master params alongside the
        # bf16-storage quantized base, reintroducing the dtype mismatch).
        precision="bf16-mixed",
        accumulate_grad_batches=args.gradient_accum,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=1,
        enable_checkpointing=args.finetune_strategy != "qlora",
        enable_progress_bar=True,
        enable_model_summary=True
    )


    # Train
    log0("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    # Test
    #log0("Running test evaluation...")
    # trainer.test(model, test_loader)

    best_score = checkpoint_callback.best_model_score
    if best_score is None:
        log0("Done! No best validation score was recorded.")
    else:
        log0(f"Done! Best validation score: {float(best_score):.4f}")

    # Cleanup GPU memory for next task
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.finetune_strategy == "qlora":
        best_adapter_path = qlora_checkpoint.best_adapter_path
        if qlora_checkpoint.best_model_score is None:
            raise RuntimeError(
                f"No QLoRA adapter was saved because monitor {task_config.monitor_metric!r} "
                "was never logged during validation."
            )

        log0(f"Loading base model and applying QLoRA adapter from: {best_adapter_path}")
        adapter_config_path = os.path.join(best_adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(
                f"Missing QLoRA adapter config at {adapter_config_path}. "
                "Check that all DDP ranks are using rank 0's adapter path."
            )

        model = OLMoRegressor(
            model_name=args.model_name,
            finetune_strategy=args.finetune_strategy,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            adapter_path=best_adapter_path,
            regressor_path=os.path.join(best_adapter_path, "regressor.pt"),
        )
    else:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        model = OLMoRegressor.load_from_checkpoint(best_ckpt)

    test_results = trainer.test(model, test_loader)

    if args.wandb and test_results:
        import wandb
        if wandb.run is not None:
            for key, value in test_results[0].items():
                wandb.run.summary[key.replace("/", "_")] = value
            wandb.finish()

    if args.delete_checkpoint:
        import shutil

        shutil.rmtree(run_output_dir, ignore_errors=True)
        log0(f"Deleted checkpoint directory: {run_output_dir}")

    del trainer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
