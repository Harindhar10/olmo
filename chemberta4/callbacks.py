"""
Custom PyTorch Lightning callbacks.

Provides WandbCallback for experiment tracking.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class WandbCallback(Callback):
    """
    Log training and validation metrics to Weights & Biases.

    Only logs on rank 0 (global zero) to avoid duplicate entries in DDP.
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log training metrics at the end of each epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        """
        if not trainer.is_global_zero:
            return

        import wandb

        for key, value in trainer.callback_metrics.items():
            if "train" in key:
                metric_name = key.replace("/", "_")
                wandb.log({metric_name: float(value)}, step=trainer.current_epoch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log validation metrics and learning rate at the end of each epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        """
        if not trainer.is_global_zero:
            return

        import wandb

        for key, value in trainer.callback_metrics.items():
            if "val" in key:
                metric_name = key.replace("/", "_")
                wandb.log({metric_name: float(value)}, step=trainer.current_epoch)

        # Log learning rate
        if trainer.lr_scheduler_configs:
            try:
                lr = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
                wandb.log({"learning_rate": lr}, step=trainer.current_epoch)
            except Exception:
                pass

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log test metrics at the end of testing.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The Lightning module being evaluated.
        """
        if not trainer.is_global_zero:
            return

        import wandb

        for key, value in trainer.callback_metrics.items():
            if "test" in key:
                metric_name = f"final_{key.replace('/', '_')}"
                wandb.log({metric_name: float(value)})
