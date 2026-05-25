"""Unit tests for classification model heads.

Tests the core forward-pass logic, output shapes, and loss computation
for ClassificationHead and the shared last_token_pool utility.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from chemberta4.model import (
    ClassificationHead,
    last_token_pool,
)


# ---------------------------------------------------------------------------
# Dummy backbone / tokenizer stubs
# ---------------------------------------------------------------------------


class DummyBackboneClassification(nn.Module):
    """Minimal backbone stub for ClassificationHead tests.

    Replaces: The real OLMo transformer (too large for unit tests).
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(256, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        h = self.embed(input_ids)
        return SimpleNamespace(hidden_states=[h])


# ---------------------------------------------------------------------------
# last_token_pool
# ---------------------------------------------------------------------------


class TestLastTokenPool:
    """Tests for the last_token_pool utility function.

    last_token_pool is the shared pooling strategy used by both
    ClassificationHead and RegressionHead. Correctness here is
    foundational: if the wrong hidden vector is extracted, every
    downstream prediction will be wrong.
    """

    def test_extracts_last_real_token_per_sample(self):
        """Verify that last_token_pool returns the hidden state at the last
        non-padding position for each sample in a batch.

        This is critical because decoder-only models accumulate context
        left-to-right, so only the last real token has attended to the full
        input. An off-by-one error here would silently corrupt every
        classification and regression prediction.

        The test uses two samples with different padding lengths (3 real
        tokens and 5 real tokens out of 6) and places distinctive values
        at the expected positions, then checks both the output shape and
        the extracted values.
        """
        B, S, H = 2, 6, 8
        hidden = torch.zeros(B, S, H)
        hidden[0, 2, :] = 1.0  # sample 0: 3 real tokens -> last at idx 2
        hidden[1, 4, :] = 2.0  # sample 1: 5 real tokens -> last at idx 4

        mask = torch.zeros(B, S, dtype=torch.long)
        mask[0, :3] = 1
        mask[1, :5] = 1

        out = last_token_pool(hidden, mask)

        assert out.shape == (B, H)
        assert out[0].sum().item() == pytest.approx(H * 1.0)
        assert out[1].sum().item() == pytest.approx(H * 2.0)


# ---------------------------------------------------------------------------
# ClassificationHead
# ---------------------------------------------------------------------------


class TestClassificationHead:
    """Tests for ClassificationHead.

    ClassificationHead adds a linear classifier on top of the backbone's
    last-token hidden state. It must correctly handle two distinct code
    paths: single_task (binary CrossEntropyLoss) and multi_task
    (BCEWithLogitsLoss with optional label masking).
    """

    B, S = 2, 8

    def _input(self) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros(self.B, self.S, dtype=torch.long)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        return ids, mask

    def test_single_task_forward_and_loss(self):
        """Verify that single-task classification produces [B, 2] logits
        and a finite scalar CrossEntropyLoss when labels are provided.

        This is the primary binary classification path. The output must
        have exactly 2 logits (negative / positive class) and the loss
        must be a scalar so the optimizer can call .backward().
        A None-loss check without labels is included to confirm the
        inference path works too.
        """
        head = ClassificationHead(DummyBackboneClassification(), task_type="single_task")
        ids, mask = self._input()

        # Inference (no labels) -> loss should be None
        logits, loss = head(ids, mask)
        assert logits.shape == (self.B, 2)
        assert loss is None

        # Training (with labels) -> loss should be finite scalar
        labels = torch.tensor([0, 1], dtype=torch.long)
        logits, loss = head(ids, mask, labels=labels)
        assert logits.shape == (self.B, 2)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_multi_task_forward_and_loss(self):
        """Verify that multi-task classification produces [B, num_tasks]
        logits and a finite scalar BCEWithLogitsLoss when labels are
        provided.

        Multi-task mode treats each task as an independent sigmoid output.
        The output dimension must equal num_tasks (not 2), and the loss
        must use BCEWithLogitsLoss which expects float labels.
        """
        n_tasks = 3
        head = ClassificationHead(
            DummyBackboneClassification(), num_tasks=n_tasks, task_type="multi_task"
        )
        ids, mask = self._input()
        labels = torch.zeros(self.B, n_tasks, dtype=torch.float32)

        logits, loss = head(ids, mask, labels=labels)
        assert logits.shape == (self.B, n_tasks)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_multi_task_label_mask_changes_loss(self):
        """Verify that the label_mask correctly excludes masked tasks from
        the loss computation.

        In multi-task molecular property datasets, some labels are missing.
        The label_mask mechanism zeros out those entries in the
        element-wise BCE loss and normalizes by the count of valid entries.
        If the mask is ignored, models would be penalized for predicting
        on tasks with no ground truth, corrupting training.

        We compare full-mask loss vs partial-mask loss: they must differ
        because the partial mask excludes one task column from the average.
        """
        n_tasks = 4
        head = ClassificationHead(
            DummyBackboneClassification(), num_tasks=n_tasks, task_type="multi_task"
        )
        ids, mask = self._input()
        labels = torch.ones(self.B, n_tasks, dtype=torch.float32)

        full_mask = torch.ones(self.B, n_tasks, dtype=torch.bool)
        partial_mask = full_mask.clone()
        partial_mask[:, -1] = False

        _, loss_full = head(ids, mask, labels=labels, label_mask=full_mask)
        _, loss_partial = head(ids, mask, labels=labels, label_mask=partial_mask)

        assert not torch.isclose(loss_full, loss_partial)
