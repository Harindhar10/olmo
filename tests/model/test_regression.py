"""Unit tests for regression model heads.

Tests the core forward-pass logic, output shapes, and loss computation
for RegressionHead.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from chemberta4.model import (
    RegressionHead,
)


# ---------------------------------------------------------------------------
# Dummy backbone / tokenizer stubs
# ---------------------------------------------------------------------------


class DummyBackboneRegression(nn.Module):
    """Minimal backbone stub for RegressionHead tests.

    Replaces: The real OLMo transformer (too large for unit tests).
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(256, hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        h = self.embed(input_ids)
        return SimpleNamespace(last_hidden_state=h)


# ---------------------------------------------------------------------------
# RegressionHead
# ---------------------------------------------------------------------------


class TestRegressionHead:
    """Tests for RegressionHead.

    RegressionHead maps the last-token hidden state to a scalar prediction
    and uses RMSE loss (with a 1e-6 epsilon for numerical stability).
    The key behaviors to verify are the output shape, loss formula, and
    the epsilon floor that prevents sqrt(0) gradient issues.
    """

    B, S = 3, 8

    def _input(self) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros(self.B, self.S, dtype=torch.long)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        return ids, mask

    def test_forward_shape_and_loss(self):
        """Verify that the forward pass produces [B] predictions and a
        finite, non-negative scalar RMSE loss when labels are provided.

        The prediction must be a 1-D tensor (one scalar per sample), not
        [B, 1], because downstream metric code expects shape [B]. The loss
        must be non-negative (RMSE is always >= 0) and finite to allow
        stable training.
        """
        head = RegressionHead(DummyBackboneRegression())
        ids, mask = self._input()

        # Inference path
        preds, loss = head(ids, mask)
        assert preds.shape == (self.B,)
        assert loss is None

        # Training path
        labels = torch.randn(self.B)
        preds, loss = head(ids, mask, labels=labels)
        assert preds.shape == (self.B,)
        assert loss.shape == ()
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_perfect_prediction_gives_epsilon_loss(self):
        """Verify that when predictions exactly match labels, the RMSE
        loss equals sqrt(1e-6) rather than zero.

        The epsilon prevents a zero-valued sqrt whose gradient is
        undefined. This test forces predictions to a known constant by
        zeroing the regressor weights and setting the bias, then checks
        that the loss matches sqrt(1e-6) within tolerance. If the epsilon
        were missing, training could produce NaN gradients on easy
        samples.
        """
        head = RegressionHead(DummyBackboneRegression())

        with torch.no_grad():
            head.regressor.weight.zero_()
            head.regressor.bias.fill_(5.0)

        ids, mask = self._input()
        labels = torch.full((self.B,), 5.0)
        _, loss = head(ids, mask, labels=labels)

        assert loss.item() == pytest.approx(math.sqrt(1e-6), rel=1e-3)
