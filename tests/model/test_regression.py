"""Unit tests for regression model heads.

Tests the core forward-pass logic, output shapes, and loss computation
for RegressionHead and CausalLMRegressionHead.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from chemberta4.model import (
    RegressionHead,
    CausalLMRegressionHead,
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


class DummyCausalLM(nn.Module):
    """Minimal causal LM stub for CausalLMRegressionHead tests.

    Replaces: A full AutoModelForCausalLM (too large for unit tests).
    """

    VOCAB_SIZE = 256

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, hidden_size)
        self.lm_head = nn.Linear(hidden_size, self.VOCAB_SIZE)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                loss = nn.CrossEntropyLoss()(logits[mask], labels[mask])
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Append a few dummy token ids so generated_ids slice is non-empty.

        Appending allows it to resemble a real generation output and lets
        us verify that the caller's slicing logic correctly isolates the
        generated portion.
        """
        answer_tokens = torch.zeros(
            1, 5, dtype=torch.long, device=input_ids.device
        )
        return torch.cat([input_ids, answer_tokens], dim=1)


class DummyTokenizerForReg:
    """Minimal tokenizer stub for CausalLMRegressionHead tests.

    Replaces: The real HuggingFace tokenizer (would decode dummy tokens
        to meaningless text instead of a parseable number).
    """

    eos_token = "<eos>"
    eos_token_id = 0

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "3.14"


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


# ---------------------------------------------------------------------------
# CausalLMRegressionHead
# ---------------------------------------------------------------------------


class TestCausalLMRegressionHead:
    """Tests for CausalLMRegressionHead.

    Unlike RegressionHead, this wrapper delegates entirely to the
    underlying causal LM for both logits and loss computation. The
    forward pass is a thin wrapper; the real complexity is in
    generate_and_parse which must locate the prompt boundary, generate
    tokens, and parse a float from free-form text.
    """

    B, S = 2, 8

    def test_forward_delegates_to_model(self):
        """Verify that forward returns the model's logits and loss
        unchanged, with correct shapes.

        CausalLMRegressionHead.forward is intentionally a pass-through:
        logits should be [B, seq_len, vocab_size] (not pooled) and loss
        should be the model's own cross-entropy scalar. If this delegation
        is broken, training with labels fails silently.
        """
        model = DummyCausalLM()
        head = CausalLMRegressionHead(model, DummyTokenizerForReg())

        ids = torch.zeros(self.B, self.S, dtype=torch.long)
        mask = torch.ones(self.B, self.S, dtype=torch.long)

        # Without labels -> loss is None
        logits, loss = head(ids, mask)
        assert logits.shape == (self.B, self.S, DummyCausalLM.VOCAB_SIZE)
        assert loss is None

        # With labels (prompt masked to -100, last 2 tokens are answer)
        labels = torch.full((self.B, self.S), -100, dtype=torch.long)
        labels[:, -2:] = 42
        logits, loss = head(ids, mask, labels=labels)
        assert logits.shape == (self.B, self.S, DummyCausalLM.VOCAB_SIZE)
        assert loss is not None
        assert loss.shape == ()

    def test_generate_and_parse_extracts_float(self):
        """Verify that generate_and_parse locates the prompt boundary,
        generates tokens, decodes the output, and parses the first float.

        This is the inference-time path for causal LM regression. The
        method must correctly identify where the answer starts (first
        position where label_ids != -100), slice the prompt, call
        model.generate, decode, and regex-parse a number. The dummy
        tokenizer always decodes to "3.14", so we expect that value.

        Also verifies the NaN fallback: when the entire label sequence
        is -100 (no answer tokens), the method should return NaN for
        that sample instead of crashing.
        """
        model = DummyCausalLM()
        head = CausalLMRegressionHead(model, DummyTokenizerForReg())

        ids = torch.zeros(1, self.S, dtype=torch.long)
        mask = torch.ones(1, self.S, dtype=torch.long)

        # label_ids: prompt is -100, answer starts at position 5
        label_ids = torch.full((1, self.S), -100, dtype=torch.long)
        label_ids[:, 5:] = 42

        preds = head.generate_and_parse(ids, mask, label_ids)
        assert preds.shape == (1,)
        assert preds[0].item() == pytest.approx(3.14)

        # All -100 -> no answer tokens -> should return NaN
        all_prompt = torch.full((1, self.S), -100, dtype=torch.long)
        preds_nan = head.generate_and_parse(ids, mask, all_prompt)
        assert math.isnan(preds_nan[0].item())
