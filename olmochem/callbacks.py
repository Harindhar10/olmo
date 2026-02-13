"""
Custom PyTorch Lightning callbacks.

Provides MLflowCallback and WandbCallback for experiment tracking.
"""

from pytorch_lightning.callbacks import Callback


class MLflowCallback(Callback):
    """
    Log training and validation metrics to MLflow.

    Only logs on rank 0 (global zero) to avoid duplicate entries in DDP.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at end of epoch."""
        if not trainer.is_global_zero:
            return

        import mlflow

        for key, value in trainer.callback_metrics.items():
            if "train" in key:
                metric_name = key.replace("/", "_")
                mlflow.log_metric(metric_name, float(value), step=trainer.current_epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at end of epoch."""
        if not trainer.is_global_zero:
            return

        import mlflow

        for key, value in trainer.callback_metrics.items():
            if "val" in key:
                metric_name = key.replace("/", "_")
                mlflow.log_metric(metric_name, float(value), step=trainer.current_epoch)

        # Log learning rate
        if trainer.lr_scheduler_configs:
            try:
                lr = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
                mlflow.log_metric("learning_rate", lr, step=trainer.current_epoch)
            except Exception:
                pass

    def on_test_epoch_end(self, trainer, pl_module):
        """Log test metrics at end of testing."""
        if not trainer.is_global_zero:
            return

        import mlflow

        for key, value in trainer.callback_metrics.items():
            if "test" in key:
                metric_name = f"final_{key.replace('/', '_')}"
                mlflow.log_metric(metric_name, float(value))


class WandbCallback(Callback):
    """
    Log training and validation metrics to Weights & Biases.

    Only logs on rank 0 (global zero) to avoid duplicate entries in DDP.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at end of epoch."""
        if not trainer.is_global_zero:
            return

        import wandb

        for key, value in trainer.callback_metrics.items():
            if "train" in key:
                metric_name = key.replace("/", "_")
                wandb.log({metric_name: float(value)}, step=trainer.current_epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at end of epoch."""
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

    def on_test_epoch_end(self, trainer, pl_module):
        """Log test metrics at end of testing."""
        if not trainer.is_global_zero:
            return

        import wandb

        for key, value in trainer.callback_metrics.items():
            if "test" in key:
                metric_name = f"final_{key.replace('/', '_')}"
                wandb.log({metric_name: float(value)})
