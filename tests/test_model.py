"""Unit tests for chemberta4.model."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from chemberta4.model import (
    ClassificationHead,
    CausalLMClassificationHead,
    RegressionHead,
    last_token_pool,
)


# ---------------------------------------------------------------------------
# Shared dummy backbone classes
# ---------------------------------------------------------------------------

class DummyBackboneCLF(nn.Module):
    """Backbone for ClassificationHead tests.

    Returns ``output.hidden_states[-1]`` as required by ClassificationHead.
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        h = self.embed(input_ids)
        return SimpleNamespace(hidden_states=[h])


class DummyBackboneREG(nn.Module):
    """Backbone for RegressionHead tests.

    Returns ``output.last_hidden_state`` as required by RegressionHead.
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask):
        h = self.embed(input_ids)
        return SimpleNamespace(last_hidden_state=h)


class DummyLM(nn.Module):
    """Causal LM backbone for CausalLMClassificationHead tests."""

    VOCAB_SIZE = 256

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, hidden_size)
        self.lm_head = nn.Linear(hidden_size, self.VOCAB_SIZE)

    def forward(self, input_ids, attention_mask):
        h = self.embed(input_ids)
        return SimpleNamespace(logits=self.lm_head(h))


class DummyTokenizerForLM:
    """Returns fixed token ids so Yes/No IDs are predictable."""

    def encode(self, text: str, add_special_tokens: bool = True):
        # 'Y' → ord('Y') = 89, 'N' → ord('N') = 78
        return [ord(text[0])]


# ---------------------------------------------------------------------------
# last_token_pool
# ---------------------------------------------------------------------------

class TestLastTokenPool:
    B, S, H = 2, 6, 8

    def _make_tensors(self):
        hidden = torch.zeros(self.B, self.S, self.H)
        # put distinctive values at the last real token per sample
        hidden[0, 2, :] = 1.0   # mask: 3 ones → last idx = 2
        hidden[1, 4, :] = 2.0   # mask: 5 ones → last idx = 4
        mask = torch.zeros(self.B, self.S, dtype=torch.long)
        mask[0, :3] = 1
        mask[1, :5] = 1
        return hidden, mask

    def test_output_shape(self):
        hidden, mask = self._make_tensors()
        out = last_token_pool(hidden, mask)
        assert out.shape == (self.B, self.H)

    def test_extracts_correct_token(self):
        hidden, mask = self._make_tensors()
        out = last_token_pool(hidden, mask)
        assert out[0].sum().item() == pytest.approx(8.0)   # 8 ones
        assert out[1].sum().item() == pytest.approx(16.0)  # 8 twos

    def test_no_padding(self):
        hidden = torch.ones(3, 4, 5)
        mask = torch.ones(3, 4, dtype=torch.long)
        out = last_token_pool(hidden, mask)
        assert out.shape == (3, 5)
        assert out.sum().item() == pytest.approx(3 * 5 * 1.0)

    def test_single_token(self):
        hidden = torch.zeros(1, 4, 8)
        hidden[0, 0, :] = 3.0
        mask = torch.tensor([[1, 0, 0, 0]])
        out = last_token_pool(hidden, mask)
        assert out[0].sum().item() == pytest.approx(24.0)  # 8 × 3


# ---------------------------------------------------------------------------
# ClassificationHead
# ---------------------------------------------------------------------------

class TestClassificationHead:
    B, S = 2, 8

    def _input(self):
        ids = torch.zeros(self.B, self.S, dtype=torch.long)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        return ids, mask

    def test_single_task_output_dim(self):
        head = ClassificationHead(DummyBackboneCLF(), num_tasks=1, task_type="single_task")
        assert head.classifier.out_features == 2

    def test_multi_task_output_dim(self):
        head = ClassificationHead(DummyBackboneCLF(), num_tasks=5, task_type="multi_task")
        assert head.classifier.out_features == 5

    def test_forward_no_labels_returns_none_loss(self):
        head = ClassificationHead(DummyBackboneCLF(), task_type="single_task")
        ids, mask = self._input()
        logits, loss = head(ids, mask)
        assert loss is None
        assert logits.shape == (self.B, 2)

    def test_single_task_forward_with_labels(self):
        head = ClassificationHead(DummyBackboneCLF(), task_type="single_task")
        ids, mask = self._input()
        labels = torch.zeros(self.B, dtype=torch.long)
        logits, loss = head(ids, mask, labels=labels)
        assert logits.shape == (self.B, 2)
        assert loss is not None
        assert loss.shape == ()   # scalar

    def test_multi_task_forward_with_labels(self):
        n_tasks = 3
        head = ClassificationHead(DummyBackboneCLF(), num_tasks=n_tasks, task_type="multi_task")
        ids, mask = self._input()
        labels = torch.zeros(self.B, n_tasks, dtype=torch.float32)
        logits, loss = head(ids, mask, labels=labels)
        assert logits.shape == (self.B, n_tasks)
        assert loss.shape == ()

    def test_multi_task_label_mask_affects_loss(self):
        """Loss with partial mask should differ from loss over full labels."""
        n_tasks = 4
        head = ClassificationHead(DummyBackboneCLF(), num_tasks=n_tasks, task_type="multi_task")
        ids, mask = self._input()
        labels = torch.ones(self.B, n_tasks, dtype=torch.float32)

        full_mask = torch.ones(self.B, n_tasks, dtype=torch.bool)
        partial_mask = full_mask.clone()
        partial_mask[:, -1] = False   # mask the last task column

        _, loss_full = head(ids, mask, labels=labels, label_mask=full_mask)
        _, loss_partial = head(ids, mask, labels=labels, label_mask=partial_mask)

        assert not torch.isclose(loss_full, loss_partial)

    def test_single_task_loss_is_finite(self):
        head = ClassificationHead(DummyBackboneCLF(), task_type="single_task")
        ids, mask = self._input()
        labels = torch.tensor([0, 1], dtype=torch.long)
        _, loss = head(ids, mask, labels=labels)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# CausalLMClassificationHead
# ---------------------------------------------------------------------------

class TestCausalLMClassificationHead:
    B, S = 2, 8

    def _input(self):
        ids = torch.zeros(self.B, self.S, dtype=torch.long)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        return ids, mask

    def test_single_forward_no_labels(self):
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=1, task_type="single_task"
        )
        ids, mask = self._input()
        logits, loss = head(ids, mask)
        assert loss is None
        assert logits.shape == (self.B, 2)

    def test_single_forward_with_labels(self):
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=1, task_type="single_task"
        )
        ids, mask = self._input()
        labels = torch.zeros(self.B, dtype=torch.long)
        logits, loss = head(ids, mask, labels=labels)
        assert logits.shape == (self.B, 2)
        assert loss is not None and loss.shape == ()

    def test_multi_task_has_projector(self):
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=4, task_type="multi_task"
        )
        assert hasattr(head, "task_projector")
        assert head.task_projector.out_features == 4

    def test_multi_task_forward_no_labels(self):
        n_tasks = 3
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=n_tasks, task_type="multi_task"
        )
        ids, mask = self._input()
        logits, loss = head(ids, mask)
        assert logits.shape == (self.B, n_tasks)
        assert loss is None

    def test_multi_task_forward_with_labels(self):
        n_tasks = 2
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=n_tasks, task_type="multi_task"
        )
        ids, mask = self._input()
        labels = torch.zeros(self.B, n_tasks, dtype=torch.float32)
        _, loss = head(ids, mask, labels=labels)
        assert loss is not None and loss.shape == ()

    def test_yes_no_token_ids_stored(self):
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=1, task_type="single_task"
        )
        assert head.yes_token_id == ord("Y")
        assert head.no_token_id == ord("N")

    def test_single_task_no_projector(self):
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=1, task_type="single_task"
        )
        assert not hasattr(head, "task_projector")


# ---------------------------------------------------------------------------
# RegressionHead
# ---------------------------------------------------------------------------

class TestRegressionHead:
    B, S = 3, 8

    def _input(self):
        ids = torch.zeros(self.B, self.S, dtype=torch.long)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        return ids, mask

    def test_forward_no_labels(self):
        head = RegressionHead(DummyBackboneREG())
        ids, mask = self._input()
        preds, loss = head(ids, mask)
        assert loss is None
        assert preds.shape == (self.B,)

    def test_forward_with_labels_returns_scalar_loss(self):
        head = RegressionHead(DummyBackboneREG())
        ids, mask = self._input()
        labels = torch.tensor([1.0, 2.0, 3.0])
        preds, loss = head(ids, mask, labels=labels)
        assert preds.shape == (self.B,)
        assert loss is not None
        assert loss.shape == ()

    def test_loss_is_nonnegative(self):
        head = RegressionHead(DummyBackboneREG())
        ids, mask = self._input()
        labels = torch.zeros(self.B)
        _, loss = head(ids, mask, labels=labels)
        assert loss.item() >= 0.0

    def test_loss_finite(self):
        head = RegressionHead(DummyBackboneREG())
        ids, mask = self._input()
        labels = torch.randn(self.B)
        _, loss = head(ids, mask, labels=labels)
        assert torch.isfinite(loss)

    def test_perfect_prediction_near_zero_loss(self):
        """When predictions exactly match labels the RMSE loss should be ~sqrt(1e-6)."""
        head = RegressionHead(DummyBackboneREG())

        # Override the regressor so predictions are forced to a fixed constant
        with torch.no_grad():
            head.regressor.weight.zero_()
            head.regressor.bias.fill_(5.0)

        ids, mask = self._input()
        labels = torch.full((self.B,), 5.0)
        _, loss = head(ids, mask, labels=labels)
        # mse = 0, loss = sqrt(0 + 1e-6) ≈ 0.001
        assert loss.item() == pytest.approx(math.sqrt(1e-6), rel=1e-3)
