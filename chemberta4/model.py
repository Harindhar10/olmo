"""
Model wrappers for classification, regression, and causal LM tasks.

These are lightweight wrappers around the backbone (OLMo with LoRA).
Each wrapper handles the task-specific output head and loss computation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract the last non-padding token representation.

    For decoder-only models like OLMo, we use the last token's representation
    for classification/regression tasks.

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        attention_mask: [batch, seq_len]

    Returns:
        Pooled output: [batch, hidden_size]
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
    """
    Classification head with last-token pooling.

    Supports single-task and multi-task classification.
    Uses CrossEntropy for single_task, BCEWithLogits for multi_task.

    Args:
        backbone: The base model (OLMo with LoRA)
        num_tasks: Number of output classes/tasks
        task_type: 'single_task' or 'multi_task'
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_tasks: int = 1,
        task_type: str = "single_task"
    ):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.num_tasks = num_tasks

        # Output dimension: 2 for single_task (class logits), num_tasks for multi_task
        output_dim = 2 if task_type == "single_task" else num_tasks

        self.classifier = nn.Linear(backbone.config.hidden_size, output_dim)

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
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch] for single_task, [batch, num_tasks] for multi_task
            label_mask: [batch, num_tasks] mask for missing labels (multi_task)

        Returns:
            logits: [batch, 2] for single_task, [batch, num_tasks] for multi_task
            loss: scalar loss if labels provided
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
        """Compute task-appropriate loss."""
        if self.task_type == "single_task":
            return nn.CrossEntropyLoss()(logits, labels)
        else:
            # multi_task: use BCE with logits
            if label_mask is not None:
                # Masked loss for missing labels
                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(logits, labels)
                return (loss * label_mask.float()).sum() / label_mask.float().sum()
            return nn.BCEWithLogitsLoss()(logits, labels)


class CausalLMClassificationHead(nn.Module):
    """
    Use the LM head to predict Yes/No tokens for classification.

    Instead of a separate classification head, this approach leverages
    the pretrained LM head to predict "Yes" or "No" tokens.

    Args:
        model: The causal LM model (OLMo with LoRA)
        tokenizer: Tokenizer for encoding Yes/No tokens
        num_tasks: Number of tasks (for multi_task)
        task_type: 'single_task' or 'multi_task'
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_tasks: int = 1,
        task_type: str = "single_task"
    ):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.num_tasks = num_tasks

        # Get token IDs for Yes/No
        self.yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

        # For multi_task: project Yes/No diff to multiple outputs
        if task_type != "single_task":
            self.task_projector = nn.Linear(1, num_tasks)
            nn.init.normal_(self.task_projector.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.task_projector.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using LM head for Yes/No prediction.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch] for single_task, [batch, num_tasks] for multi_task
            label_mask: [batch, num_tasks] mask for missing labels

        Returns:
            logits: [batch, 2] for single_task, [batch, num_tasks] for multi_task
            loss: scalar loss if labels provided
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get logits at the last token position
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = outputs.logits.shape[0]

        last_logits = outputs.logits[
            torch.arange(batch_size, device=outputs.logits.device),
            sequence_lengths
        ]  # [batch, vocab_size]

        # Extract Yes/No logits
        yes_logits = last_logits[:, self.yes_token_id]
        no_logits = last_logits[:, self.no_token_id]

        if self.task_type == "single_task":
            # Stack as [No, Yes] for class indices [0, 1]
            logits = torch.stack([no_logits, yes_logits], dim=-1)
        else:
            # Project Yes-No difference to multiple tasks
            yes_no_diff = (yes_logits - no_logits).unsqueeze(-1)
            logits = self.task_projector(yes_no_diff)

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
        """Compute task-appropriate loss."""
        if self.task_type == "single_task":
            return nn.CrossEntropyLoss()(logits, labels)
        else:
            if label_mask is not None:
                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(logits, labels)
                return (loss * label_mask.float()).sum() / label_mask.float().sum()
            return nn.BCEWithLogitsLoss()(logits, labels)


class RegressionHead(nn.Module):
    """
    Regression head with last-token pooling.

    Uses RMSE loss by default.

    Args:
        backbone: The base model (OLMo with LoRA)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Linear(backbone.config.hidden_size, 1)

        # Initialize with small weights
        nn.init.normal_(self.regressor.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.regressor.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for regression.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch] normalized regression targets

        Returns:
            predictions: [batch]
            loss: scalar RMSE loss if labels provided
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
