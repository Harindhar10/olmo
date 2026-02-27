"""Unit tests for chemberta4.model.

Tests the core forward-pass logic, output shapes, and loss computation
for every model head: ClassificationHead, CausalLMClassificationHead,
RegressionHead, and CausalLMRegressionHead, plus the shared
last_token_pool utility.
"""

import math
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from chemberta4.model import (
    ClassificationHead,
    CausalLMClassificationHead,
    RegressionHead,
    CausalLMRegressionHead,
    last_token_pool,
)


# ---------------------------------------------------------------------------
# Dummy backbone / tokenizer stubs
# ---------------------------------------------------------------------------


class DummyBackboneCLF(nn.Module):
    """Minimal backbone for ClassificationHead.

    Returns ``output.hidden_states[-1]`` which is what ClassificationHead
    reads after calling the backbone with ``output_hidden_states=True``.
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        h = self.embed(input_ids)
        return SimpleNamespace(hidden_states=[h])


class DummyBackboneREG(nn.Module):
    """Minimal backbone for RegressionHead.

    Returns ``output.last_hidden_state`` which is what RegressionHead reads.
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask):
        h = self.embed(input_ids)
        return SimpleNamespace(last_hidden_state=h)


class DummyLM(nn.Module):
    """Minimal causal LM for CausalLMClassificationHead.

    Returns ``output.logits`` of shape ``[batch, seq_len, vocab_size]``.
    """

    VOCAB_SIZE = 256

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, hidden_size)
        self.lm_head = nn.Linear(hidden_size, self.VOCAB_SIZE)

    def forward(self, input_ids, attention_mask):
        h = self.embed(input_ids)
        return SimpleNamespace(logits=self.lm_head(h))


class DummyCausalLM(nn.Module):
    """Minimal causal LM for CausalLMRegressionHead.

    Accepts an optional ``labels`` kwarg and returns ``.logits`` and ``.loss``.
    Also implements a trivial ``generate`` that just returns the input ids
    concatenated with a fixed answer token sequence.
    """

    VOCAB_SIZE = 256

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, hidden_size)
        self.lm_head = nn.Linear(hidden_size, self.VOCAB_SIZE)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                loss = nn.CrossEntropyLoss()(logits[mask], labels[mask])
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, attention_mask, **kwargs):
        # Append a few dummy token ids so generated_ids slice is non-empty
        answer_tokens = torch.zeros(
            1, 5, dtype=torch.long, device=input_ids.device
        )
        return torch.cat([input_ids, answer_tokens], dim=1)


class DummyTokenizerForLM:
    """Tokenizer stub for CausalLMClassificationHead.

    Maps the first character of input text to its ASCII ordinal so that
    Yes -> 89, No -> 78.
    """

    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(text[0])]


class DummyTokenizerForReg:
    """Tokenizer stub for CausalLMRegressionHead.

    Always decodes to the string ``"3.14"`` so the regex parser can
    extract a predictable float.
    """

    eos_token = "<eos>"
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "3.14"


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

    def _input(self):
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
        head = ClassificationHead(DummyBackboneCLF(), task_type="single_task")
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
            DummyBackboneCLF(), num_tasks=n_tasks, task_type="multi_task"
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
            DummyBackboneCLF(), num_tasks=n_tasks, task_type="multi_task"
        )
        ids, mask = self._input()
        labels = torch.ones(self.B, n_tasks, dtype=torch.float32)

        full_mask = torch.ones(self.B, n_tasks, dtype=torch.bool)
        partial_mask = full_mask.clone()
        partial_mask[:, -1] = False

        _, loss_full = head(ids, mask, labels=labels, label_mask=full_mask)
        _, loss_partial = head(ids, mask, labels=labels, label_mask=partial_mask)

        assert not torch.isclose(loss_full, loss_partial)


# ---------------------------------------------------------------------------
# CausalLMClassificationHead
# ---------------------------------------------------------------------------


class TestCausalLMClassificationHead:
    """Tests for CausalLMClassificationHead.

    This head re-uses the LM vocabulary head to score Yes/No tokens
    instead of adding a separate classifier. It has two distinct paths:
    single_task stacks [No, Yes] logits, while multi_task projects the
    Yes-No difference through a learned linear layer. Correctness of
    token ID lookup and the presence/absence of the projector layer
    are the key things to verify.
    """

    B, S = 2, 8

    def _input(self):
        ids = torch.zeros(self.B, self.S, dtype=torch.long)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        return ids, mask

    def test_single_task_yes_no_logits(self):
        """Verify that single-task mode stores the correct Yes/No token IDs
        and produces [B, 2] logits with no loss during inference.

        The Yes/No token IDs are looked up once at init and used every
        forward pass to slice into the vocabulary logits. If these IDs are
        wrong, the model scores the wrong tokens and classification is
        meaningless. The output must be [B, 2] with ordering [No, Yes]
        so that class index 1 = positive.
        """
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=1, task_type="single_task"
        )
        ids, mask = self._input()

        assert head.yes_token_id == ord("Y")
        assert head.no_token_id == ord("N")

        logits, loss = head(ids, mask)
        assert logits.shape == (self.B, 2)
        assert loss is None

    def test_single_task_with_labels(self):
        """Verify that providing labels produces a finite scalar
        CrossEntropyLoss for single-task mode.

        Without a working loss, the model cannot train. This test
        confirms the loss computation path activates and returns a
        properly shaped, finite tensor.
        """
        head = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=1, task_type="single_task"
        )
        ids, mask = self._input()
        labels = torch.zeros(self.B, dtype=torch.long)

        logits, loss = head(ids, mask, labels=labels)
        assert logits.shape == (self.B, 2)
        assert loss is not None
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_multi_task_uses_projector(self):
        """Verify that multi_task mode creates a task_projector layer that
        maps the scalar Yes-No difference to num_tasks outputs, and that
        single_task mode does NOT have this layer.

        The projector is the architectural difference between single and
        multi-task CausalLM classification. Its absence in single_task
        and correct output dimension in multi_task are both critical.
        """
        n_tasks = 4
        head_multi = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=n_tasks, task_type="multi_task"
        )
        head_single = CausalLMClassificationHead(
            DummyLM(), DummyTokenizerForLM(), num_tasks=1, task_type="single_task"
        )

        assert hasattr(head_multi, "task_projector")
        assert head_multi.task_projector.out_features == n_tasks
        assert not hasattr(head_single, "task_projector")

        ids, mask = self._input()
        logits, _ = head_multi(ids, mask)
        assert logits.shape == (self.B, n_tasks)


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

    def _input(self):
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
        head = RegressionHead(DummyBackboneREG())
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
        head = RegressionHead(DummyBackboneREG())

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
        is broken, teacher-forced training fails silently.
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
