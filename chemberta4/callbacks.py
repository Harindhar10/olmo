"""
Custom PyTorch Lightning callbacks.

Provides WandbCallback for experiment tracking.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class WandbCallback(Callback):
    """
    This class logs training and validation metrics to Weights & Biases.

    It runs only on rank 0 (global zero) to avoid duplicate entries in DDP.
    
    Examples
    --------
    >>> import wandb
    >>> from pytorch_lightning import Trainer
    >>> from chemberta4 import WandbCallback
    >>>
    >>> wandb.init(project="my-project", name="run-1")
    >>> callback = WandbCallback()
    >>> trainer = Trainer(callbacks=[callback], max_epochs=10)
    >>> trainer.fit(model, datamodule=dm)
    """

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log training and validation metrics at the end of each validation epoch.

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

        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if "train" in key or "val" in key:
                metrics[key.replace("/", "_")] = float(value)

        # Log learning rate
        if trainer.lr_scheduler_configs:
            try:
                lr = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
                metrics["learning_rate"] = lr
            except Exception:
                pass

        if metrics:
            wandb.log(metrics, step=trainer.current_epoch)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log test metrics as wandb summary values.

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
                wandb.run.summary[metric_name] = float(value)
