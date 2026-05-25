"""
Model wrappers for classification, regression, and causal LM tasks.

These are lightweight wrappers around the backbone (OLMo with LoRA).
Each wrapper handles the task-specific output head and loss computation.
"""

import re
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PreTrainedTokenizerBase


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """This function extracts the last non-padding token representation.

    For decoder-only models like OLMo, we use the last token's representation
    for classification/regression tasks. Decoder-only transformers process
    tokens left-to-right, so the final non-padding position has attended to
    the entire input sequence. This function uses
    'attention_mask.sum(dim=1) - 1' to locate that position for each item
    in the batch and extracts the corresponding hidden vector with
    'torch.gather', avoiding any loop over batch elements.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Hidden states of shape '[batch, seq_len, hidden_size]'.
    attention_mask : torch.Tensor
        Attention mask of shape '[batch, seq_len]'.

    Returns
    -------
    torch.Tensor
        Pooled output of shape '[batch, hidden_size]'.

    Examples
    --------
    >>> import torch
    >>> from chemberta4.model import last_token_pool
    >>> hidden = torch.zeros(2, 4, 8)
    >>> hidden[0, 2, :] = 1.0   # last real token at position 2 for sample 0
    >>> hidden[1, 3, :] = 1.0   # last real token at position 3 for sample 1
    >>> mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    >>> out = last_token_pool(hidden, mask)
    >>> out.shape
    torch.Size([2, 8])
    >>> out[0].sum().item()
    8.0
    >>> out[1].sum().item()
    8.0
    """
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]

    # Create indices for gathering
    indices = sequence_lengths.view(-1, 1, 1).expand(
        batch_size, 1, hidden_states.size(-1)
    )
    indices = indices.to(hidden_states.device)

    # Gather and squeeze
    return torch.gather(hidden_states, 1, indices).squeeze(1)


class ClassificationHead(nn.Module):
    """This class implements a classification head with last-token pooling for binary and multi-label prediction.

    It supports both single-task and multi-task classification.
    It uses CrossEntropyLoss for single_task and BCEWithLogitsLoss for multi_task.

    Takes a backbone 'nn.Module' (typically OLMo with LoRA adapters) and adds
    a single linear layer on top of the last-token hidden state. For
    'single_task' the output has 2 logits (binary) and is trained with
    CrossEntropyLoss; for 'multi_task' there are 'num_tasks' sigmoid outputs
    trained with BCEWithLogitsLoss. Missing labels in multi-task datasets are
    excluded from the loss via 'label_mask'.

    Examples
    --------
    >>> import torch, torch.nn as nn
    >>> from types import SimpleNamespace
    >>> from chemberta4.model import ClassificationHead
    >>> class DummyBackbone(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.config = SimpleNamespace(hidden_size=16)
    ...         self.embed = nn.Embedding(100, 16)
    ...     def forward(self, input_ids, attention_mask, output_hidden_states=False):
    ...         h = self.embed(input_ids)
    ...         return SimpleNamespace(hidden_states=[h])
    >>> head = ClassificationHead(DummyBackbone(), num_tasks=1, task_type='single_task')
    >>> input_ids = torch.zeros(2, 8, dtype=torch.long)
    >>> mask = torch.ones(2, 8, dtype=torch.long)
    >>> logits, loss = head(input_ids, mask)
    >>> logits.shape
    torch.Size([2, 2])
    >>> loss is None
    True
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_tasks: int = 1,
        task_type: str = "single_task"
    ):
        """Initialise ClassificationHead.

        Parameters
        ----------
        backbone : nn.Module
            The base model (OLMo with LoRA).
        num_tasks : int
            Number of output classes/tasks.
        task_type : str
            'single_task' or 'multi_task'.
        """
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.num_tasks = num_tasks

        # Output dimension: 2 for single_task (class logits), num_tasks for multi_task
        output_dim = 2 if task_type == "single_task" else num_tasks

        self.classifier = nn.Linear(backbone.config.hidden_size, output_dim, dtype=torch.bfloat16)

        # Initialize with small weights
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the forward pass for classification.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '[batch, seq_len]'.
        attention_mask : torch.Tensor
            Attention mask of shape '[batch, seq_len]'.
        labels : torch.Tensor, optional
            Labels of shape '[batch]' for single_task or '[batch, num_tasks]' for multi_task.
        label_mask : torch.Tensor, optional
            Boolean mask of shape '[batch, num_tasks]' for missing labels (multi_task).

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Logits of shape '[batch, 2]' for single_task or '[batch, num_tasks]' for multi_task,
            and a scalar loss tensor if labels are provided, else None.

        Examples
        --------
        >>> import torch, torch.nn as nn
        >>> from types import SimpleNamespace
        >>> from chemberta4.model import ClassificationHead
        >>> class DummyBackbone(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.config = SimpleNamespace(hidden_size=16)
        ...         self.embed = nn.Embedding(100, 16)
        ...     def forward(self, input_ids, attention_mask, output_hidden_states=False):
        ...         h = self.embed(input_ids)
        ...         return SimpleNamespace(hidden_states=[h])
        >>> head = ClassificationHead(DummyBackbone(), num_tasks=1, task_type='single_task')
        >>> input_ids = torch.zeros(2, 8, dtype=torch.long)
        >>> mask = torch.ones(2, 8, dtype=torch.long)
        >>> logits, loss = head(input_ids, mask)
        >>> logits.shape
        torch.Size([2, 2])
        >>> loss is None
        True
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use the last hidden state
        last_hidden_state = out.hidden_states[-1]
        pooled_output = last_token_pool(last_hidden_state, attention_mask)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels, label_mask)

        return logits, loss

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the task-appropriate classification loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits.
        labels : torch.Tensor
            Ground-truth labels.
        label_mask : torch.Tensor, optional
            Boolean mask for valid labels (multi_task only).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        if self.task_type == "single_task":
            return nn.CrossEntropyLoss()(logits, labels)
        else:
            # multi_task: use BCE with logits
            if label_mask is not None:
                # Masked loss for missing labels
                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(logits, labels)
                # Returns mean BCE loss over only valid (non-missing) labels
                return (loss * label_mask.float()).sum() / label_mask.float().sum()
            return nn.BCEWithLogitsLoss()(logits, labels)


class RegressionHead(nn.Module):
    """This class implements a regression head with last-token pooling for scalar molecular property prediction.

    It uses RMSE loss by default.

    Takes a backbone 'nn.Module' and appends a single linear unit that maps
    the last-token hidden state to a scalar. Loss is the square-root of MSE
    (RMSE) with a small epsilon (1e-6) added for numerical stability.

    Examples
    --------
    >>> import torch, torch.nn as nn
    >>> from types import SimpleNamespace
    >>> from chemberta4.model import RegressionHead
    >>> class DummyBackbone(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.config = SimpleNamespace(hidden_size=16)
    ...         self.embed = nn.Embedding(100, 16)
    ...     def forward(self, input_ids, attention_mask):
    ...         h = self.embed(input_ids)
    ...         return SimpleNamespace(last_hidden_state=h)
    >>> head = RegressionHead(DummyBackbone())
    >>> input_ids = torch.zeros(2, 8, dtype=torch.long)
    >>> mask = torch.ones(2, 8, dtype=torch.long)
    >>> preds, loss = head(input_ids, mask)
    >>> preds.shape
    torch.Size([2])
    >>> loss is None
    True
    """

    def __init__(self, backbone: nn.Module):
        """Initialise RegressionHead.

        Parameters
        ----------
        backbone : nn.Module
            The base model (OLMo with LoRA).
        """
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Linear(backbone.config.hidden_size, 1, dtype=torch.bfloat16)

        # Initialize with small weights
        nn.init.normal_(self.regressor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.regressor.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the forward pass for regression.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape '[batch, seq_len]'.
        attention_mask : torch.Tensor
            Attention mask of shape '[batch, seq_len]'.
        labels : torch.Tensor, optional
            Normalized regression targets of shape '[batch]'.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Predicted values of shape '[batch]' and scalar RMSE loss if labels are provided, else None.

        Examples
        --------
        >>> import torch, torch.nn as nn
        >>> from types import SimpleNamespace
        >>> from chemberta4.model import RegressionHead
        >>> class DummyBackbone(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.config = SimpleNamespace(hidden_size=16)
        ...         self.embed = nn.Embedding(100, 16)
        ...     def forward(self, input_ids, attention_mask):
        ...         h = self.embed(input_ids)
        ...         return SimpleNamespace(last_hidden_state=h)
        >>> head = RegressionHead(DummyBackbone())
        >>> input_ids = torch.zeros(2, 8, dtype=torch.long)
        >>> mask = torch.ones(2, 8, dtype=torch.long)
        >>> preds, loss = head(input_ids, mask)
        >>> preds.shape
        torch.Size([2])
        >>> loss is None
        True
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_hidden_state = out.last_hidden_state
        pooled_output = last_token_pool(last_hidden_state, attention_mask)
        preds = self.regressor(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            # RMSE loss with epsilon for numerical stability
            loss = torch.sqrt(nn.functional.mse_loss(preds, labels) + 1e-6)

        return preds, loss
