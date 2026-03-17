"""
PyTorch Lightning training modules.

Provides OLMoClassifier, OLMoRegressor, and OLMoPretrainer modules
with support for QLoRA and full finetuning.
"""


from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torchmetrics import Accuracy, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from chemberta4.model import ClassificationHead, CausalLMClassificationHead, RegressionHead, CausalLMRegressionHead
from chemberta4.utils import get_device_map


class OLMoClassifier(pl.LightningModule):
    """This class implements a PyTorch Lightning module for molecular classification tasks.

    It supports single-task and multi-task classification.
    It can use either a classification head or an LM head for Yes/No token prediction.
    It supports QLoRA (4-bit quantization), LoRA, and full finetuning strategies.

    Orchestrates the full classification training loop on top of OLMo (or any
    decoder-only model). Model loading and LoRA/QLoRA setup are deferred to
    'configure_model()' so the module can be safely instantiated on CPU before
    a trainer is attached. Accuracy and AUROC are tracked per split; for
    multi-task datasets, rows with all labels missing are excluded from the
    metric update.

    Examples
    --------
    >>> from chemberta4.trainer import OLMoClassifier
    >>> clf = OLMoClassifier(
    ...     model_name='allenai/OLMo-7B-hf',
    ...     num_tasks=1,
    ...     task_type='single_task',
    ...     finetune_strategy='qlora',
    ...     lr=2e-4,
    ... )
    >>> clf.hparams.num_tasks
    1
    >>> clf.hparams.finetune_strategy
    'qlora'
    >>> clf.model is None
    True
    """

    def __init__(
        self,
        model_name: str = "allenai/OLMo-7B-hf",
        num_tasks: int = 1,
        task_type: str = "single_task",
        use_lm_head: bool = False,
        finetune_strategy: str = "qlora",
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
    ):
        """Initialise OLMoClassifier.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.
        num_tasks : int
            Number of classification tasks/labels.
        task_type : str
            One of 'single_task' or 'multi_task'.
        use_lm_head : bool
            If 'True', use Yes/No LM-head prediction instead of a classification head.
        finetune_strategy : str
            One of 'qlora' (4-bit + LoRA), 'lora' (LoRA only), or
            'full_finetune' (all parameters trainable).
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay for AdamW.
        warmup_ratio : float
            Fraction of total steps used for linear warmup.
        lora_r : int
            LoRA rank.
        lora_alpha : int
            LoRA alpha (typically 2× rank).
        lora_dropout : float
            LoRA dropout rate.

        Examples
        --------
        >>> from chemberta4.trainer import OLMoClassifier
        >>> clf = OLMoClassifier(
        ...     model_name='allenai/OLMo-7B-hf',
        ...     num_tasks=1,
        ...     task_type='single_task',
        ...     finetune_strategy='qlora',
        ...     lr=2e-4,
        ... )
        >>> clf.hparams.num_tasks
        1
        >>> clf.hparams.finetune_strategy
        'qlora'
        >>> clf.model is None
        True
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = None
        self.tokenizer = None

        # Setup metrics based on task type
        if task_type == "single_task":
            metric_kwargs = {"task": "binary"}
        else:
            metric_kwargs = {
                "task": "multilabel",
                "num_labels": num_tasks,
                "average": "macro",
            }

        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_auroc = AUROC(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)
        self.test_auroc = AUROC(**metric_kwargs)

    def configure_model(self) -> None:
        """Initialise the backbone model and optional LoRA adapters.

        Called by the trainer before training starts. Loads the base model,
        applies quantization (QLoRA) and LoRA adapters based on
        'finetune_strategy', then wraps with the appropriate head.
        """
        if self.model is not None:
            return

        hp = self.hparams

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if hp.use_lm_head:
            # Use AutoModelForCausalLM with LM head
            base = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(hp.model_name),
            )
            if hp.finetune_strategy == "qlora":
                base = prepare_model_for_kbit_training(
                    base, use_gradient_checkpointing=True
                )
            if hp.finetune_strategy != "full_finetune":
                lora_cfg = LoraConfig(
                    r=hp.lora_r,
                    lora_alpha=hp.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=hp.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                base = get_peft_model(base, lora_cfg)

            if self.global_rank == 0:
                base.print_trainable_parameters()

            self.model = CausalLMClassificationHead(
                base, self.tokenizer, hp.num_tasks, hp.task_type
            )
        else:
            # Use AutoModel with classification head
            base = AutoModel.from_config(
                AutoConfig.from_pretrained(hp.model_name),
            )
            if hp.finetune_strategy == "qlora":
                base = prepare_model_for_kbit_training(
                    base, use_gradient_checkpointing=True
                )
            if hp.finetune_strategy != "full_finetune":
                lora_cfg = LoraConfig(
                    r=hp.lora_r,
                    lora_alpha=hp.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=hp.lora_dropout,
                    bias="none",
                    task_type="FEATURE_EXTRACTION",
                )
                base = get_peft_model(base, lora_cfg)

            if self.global_rank == 0:
                base.print_trainable_parameters()

            self.model = ClassificationHead(base, hp.num_tasks, hp.task_type)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """Run the forward pass through the classification model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '(batch, seq_len)'.
        attention_mask : torch.Tensor
            Attention mask of shape '(batch, seq_len)'.
        labels : torch.Tensor, optional
            Ground-truth labels for loss computation.

        Returns
        -------
        tuple
            '(logits, loss)' returned by the classification head.
        """
        return self.model(input_ids, attention_mask, labels)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Compute loss and update metrics for a single batch.

        Parameters
        ----------
        batch : tuple
            DeepChem 4-tuple (X_dict, y, w, ids) from the DataLoader.
        stage : str
            One of 'train', 'val', or 'test'.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        x_dict, _, _, _ = batch
        input_ids = x_dict["input_ids"].squeeze(1)            # (B,1,seq) -> (B,seq)
        attention_mask = x_dict["attention_mask"].squeeze(1)   # (B,1,seq) -> (B,seq)
        labels = x_dict["labels"]                              # (B,)

        logits, loss = self(input_ids, attention_mask, labels)

        # Get metrics for this stage
        acc_metric = getattr(self, f"{stage}_acc")
        auroc_metric = getattr(self, f"{stage}_auroc", None)

        # Compute predictions and update metrics
        if self.hparams.task_type == "single_task":
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            acc_metric(preds, labels)
            if auroc_metric is not None:
                auroc_metric(probs, labels)
        else:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            acc_metric(preds, labels.int())
            if auroc_metric is not None:
                auroc_metric(probs, labels.int())

        # Log metrics
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/acc", acc_metric, on_epoch=True, sync_dist=True)
        if auroc_metric is not None:
            self.log(
                f"{stage}/roc_auc", auroc_metric, on_epoch=True, prog_bar=True, sync_dist=True
            )

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Execute a single validation step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar validation loss.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Execute a single test step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar test loss.
        """
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict:
        """Set up AdamW optimizer with warmup + cosine annealing scheduler.

        Returns
        -------
        Dict
            Dict with 'optimizer' and 'lr_scheduler' keys, as expected
            by PyTorch Lightning.
        """
        hp = self.hparams

        # Separate params for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "layer_norm" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": hp.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=hp.lr,
        )

        # Warmup + cosine annealing
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * hp.warmup_ratio)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
                ),
                CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
                ),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


class OLMoRegressor(pl.LightningModule):
    """This class implements a PyTorch Lightning module for molecular regression tasks.

    Supports two head types selected by 'use_lm_head':

    * 'False' (default): 'RegressionHead' — last-token pooling + linear layer,
      trained with RMSE loss on raw labels.
    * 'True': 'CausalLMRegressionHead' — teacher-forced causal LM with
      cross-entropy loss during training; generates text and parses a float
      with regex during validation/test for RMSE reporting.

    Examples
    --------
    >>> from chemberta4.trainer import OLMoRegressor
    >>> reg = OLMoRegressor()
    >>> reg.hparams.use_lm_head
    False
    """

    def __init__(
        self,
        model_name: str = "allenai/OLMo-7B-hf",
        finetune_strategy: str = "qlora",
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        use_lm_head: bool = False,
    ):
        """Initialise OLMoRegressor.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.
        finetune_strategy : str
            One of 'qlora', 'lora', or 'full_finetune'.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay for AdamW.
        warmup_ratio : float
            Fraction of total steps used for linear warmup.
        lora_r : int
            LoRA rank.
        lora_alpha : int
            LoRA alpha.
        lora_dropout : float
            LoRA dropout rate.
        use_lm_head : bool
            If 'True', use 'CausalLMRegressionHead' with cross-entropy training
            and text-generation evaluation. If 'False', use 'RegressionHead'
            with direct RMSE loss.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = None
        self.tokenizer = None

    def configure_model(self) -> None:
        """Initialise the backbone model and optional LoRA adapters.

        Called by the trainer before training starts. Applies quantization
        and LoRA based on 'finetune_strategy', then wraps with a regression head.
        """
        if self.model is not None:
            return

        hp = self.hparams

        self.tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if hp.use_lm_head:
            base = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(hp.model_name),
            )
        else:
            base = AutoModel.from_config(
                AutoConfig.from_pretrained(hp.model_name),
            )

        if hp.finetune_strategy == "qlora":
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        if hp.finetune_strategy != "full_finetune":
            lora_task_type = TaskType.CAUSAL_LM if hp.use_lm_head else "FEATURE_EXTRACTION"
            lora_cfg = LoraConfig(
                r=hp.lora_r,
                lora_alpha=hp.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=hp.lora_dropout,
                bias="none",
                task_type=lora_task_type,
            )
            base = get_peft_model(base, lora_cfg)

        if self.global_rank == 0:
            base.print_trainable_parameters()

        if hp.use_lm_head:
            self.model = CausalLMRegressionHead(base, self.tokenizer)
        else:
            self.model = RegressionHead(base)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """Run the forward pass through the regression model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '(batch, seq_len)'.
        attention_mask : torch.Tensor
            Attention mask of shape '(batch, seq_len)'.
        labels : torch.Tensor, optional
            Ground-truth labels for loss computation.

        Returns
        -------
        tuple
            '(predictions, loss)' returned by the regression head.
        """
        return self.model(input_ids, attention_mask, labels)


    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Compute loss and log RMSE/MAE for a single batch.

        Parameters
        ----------
        batch : tuple
            DeepChem 4-tuple (X_dict, y, w, ids) from the DataLoader.
        stage : str
            One of 'train', 'val', or 'test'.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        x_dict, _, _, _ = batch
        input_ids = x_dict["input_ids"].squeeze(1)            # (B,1,seq) -> (B,seq)
        attention_mask = x_dict["attention_mask"].squeeze(1)   # (B,1,seq) -> (B,seq)
        labels = x_dict["labels"]                              # (B,)

        if self.hparams.use_lm_head:
            return self._clm_step(input_ids, attention_mask, labels, stage)

        preds, loss = self(input_ids, attention_mask, labels)

        rmse = torch.sqrt(torch.nn.functional.mse_loss(preds, labels) + 1e-6)
        mae = torch.mean(torch.abs(preds - labels))

        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/rmse", rmse, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/mae", mae, on_epoch=True, sync_dist=True)

        return loss

    def _clm_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """Handle a batch for the CausalLMRegressionHead.

        During training, computes and logs cross-entropy loss.
        During validation/test, generates text, parses floats with regex,
        and logs RMSE/MAE.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '(batch, seq_len)'.
        attention_mask : torch.Tensor
            Attention mask of shape '(batch, seq_len)'.
        labels : torch.Tensor
            Ground-truth labels (raw float targets).
        stage : str
            One of 'train', 'val', or 'test'.

        Returns
        -------
        torch.Tensor
            Cross-entropy loss for training, RMSE for validation/test.
        """
        if stage == "train":
            _, loss = self(input_ids, attention_mask, labels)
            self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss

        # Evaluation: generate text and parse predicted numbers
        parsed_preds = self.model.generate_and_parse(
            input_ids, attention_mask, labels,
        )
        true_vals = labels.float().to(parsed_preds.device)

        valid = ~(torch.isnan(parsed_preds) | torch.isnan(true_vals))
        if valid.any():
            rmse = torch.sqrt(
                ((parsed_preds[valid] - true_vals[valid]) ** 2).mean() + 1e-6
            )
            mae = torch.abs(parsed_preds[valid] - true_vals[valid]).mean()
        else:
            rmse = torch.tensor(float("nan"), device=self.device)
            mae = torch.tensor(float("nan"), device=self.device)

        self.log(f"{stage}/rmse", rmse, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/mae", mae, on_epoch=True, sync_dist=True)

        return rmse

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Execute a single validation step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar validation loss.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Execute a single test step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of tokenized samples from the DataLoader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar test loss.
        """
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict:
        """Set up AdamW optimizer with warmup + cosine annealing scheduler.

        Returns
        -------
        Dict
            Dict with 'optimizer' and 'lr_scheduler' keys, as expected
            by PyTorch Lightning.
        """
        hp = self.hparams

        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "layer_norm" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": hp.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=hp.lr,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * hp.warmup_ratio)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
                ),
                CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
                ),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


class OLMoPretrainer(pl.LightningModule):
    """This class implements a PyTorch Lightning module for causal language model pretraining.

    It handles pretraining on SMILES corpora (ZINC20, PubChem) and instruction tuning on
    reaction datasets (USPTO).

    The module handles causal LM training for both SMILES pretraining (ZINC20, PubChem)
    and instruction tuning (USPTO). The same module is reused for both tasks
    because both reduce to next-token prediction with cross-entropy loss. The
    validation step additionally computes perplexity.

    Examples
    --------
    >>> from chemberta4.trainer import OLMoPretrainer
    >>> pt = OLMoPretrainer(
    ...     model_name='allenai/OLMo-7B-hf',
    ...     finetune_strategy='qlora',
    ...     lr=1e-4,
    ... )
    >>> pt.hparams.finetune_strategy
    'qlora'
    >>> pt.model is None
    True
    """

    def __init__(
        self,
        model_name: str = "allenai/OLMo-7B-hf",
        finetune_strategy: str = "qlora",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_ratio: float = 0.15,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ):
        """Initialise OLMoPretrainer.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier or path to a pretrained model.
        finetune_strategy : str
            One of 'qlora', 'lora', or 'full_finetune'.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay.
        warmup_ratio : float
            Fraction of total steps used for linear warmup.
        lora_r : int
            LoRA rank.
        lora_alpha : int
            LoRA alpha.
        lora_dropout : float
            LoRA dropout rate.
        gradient_checkpointing : bool
            Whether to enable gradient checkpointing to reduce VRAM usage.

        Examples
        --------
        >>> from chemberta4.trainer import OLMoPretrainer
        >>> pt = OLMoPretrainer(
        ...     model_name='allenai/OLMo-7B-hf',
        ...     finetune_strategy='qlora',
        ...     lr=1e-4,
        ... )
        >>> pt.hparams.finetune_strategy
        'qlora'
        >>> pt.model is None
        True
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = None
        self.tokenizer = None

    def configure_model(self) -> None:
        """Initialize the causal LM model with the configured fine-tuning strategy."""
        if self.model is not None:
            return

        hp = self.hparams

        self.tokenizer = AutoTokenizer.from_pretrained(hp.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(hp.model_name),
        )

        model.config.use_cache = False
        if hp.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if hp.finetune_strategy == "qlora":
            model = prepare_model_for_kbit_training(model)

        if hp.finetune_strategy != "full_finetune":
            lora_cfg = LoraConfig(
                r=hp.lora_r,
                lora_alpha=hp.lora_alpha,
                target_modules="all-linear",
                lora_dropout=hp.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)

        self.model = model

        if self.trainer.is_global_zero:
            self.model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """Run a forward pass through the causal LM.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '(batch, seq_len)'.
        attention_mask : torch.Tensor
            Attention mask of shape '(batch, seq_len)'.
        labels : torch.Tensor, optional
            Target token IDs for language modelling loss.

        Returns
        -------
        Any
            Model output with 'loss' and 'logits' attributes.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Compute causal LM loss for a training batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch with 'input_ids', 'attention_mask', and 'labels'.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Compute loss and perplexity for a validation batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch with 'input_ids', 'attention_mask', and 'labels'.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar validation loss.
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Perplexity
        perplexity = torch.exp(loss)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", perplexity, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Dict:
        """Build AdamW optimizer with linear warmup and cosine annealing schedule.

        Returns
        -------
        Dict
            Dict with 'optimizer' and 'lr_scheduler' keys.
        """
        hp = self.hparams

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(hp.warmup_ratio * total_steps)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps
                ),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
