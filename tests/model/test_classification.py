"""Unit tests for classification model heads.

Tests the core forward-pass logic, output shapes, and loss computation
for ClassificationHead, CausalLMClassificationHead, and the shared
last_token_pool utility.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from chemberta4.model import (
    ClassificationHead,
    CausalLMClassificationHead,
    last_token_pool,
)


# ---------------------------------------------------------------------------
# Dummy backbone / tokenizer stubs
# ---------------------------------------------------------------------------


class DummyBackboneCLF(nn.Module):
    """Fake OLMo backbone for ClassificationHead tests.

    Replaces: The real OLMo transformer with LoRA adapters — a billion-
        parameter model that is too large and slow to load in unit tests.
    What the head expects: ClassificationHead calls the backbone with
        ``output_hidden_states=True`` and reads ``.hidden_states[-1]``
        (the last layer's output). It also reads ``config.hidden_size``
        to know how wide the classifier layer should be.
    How it works: A single nn.Embedding turns token IDs into vectors.
        The output is wrapped in a SimpleNamespace with a one-element
        ``hidden_states`` list so ``[-1]`` returns the embeddings.
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        h = self.embed(input_ids)
        return SimpleNamespace(hidden_states=[h])


class DummyLM(nn.Module):
    """Fake causal language model for CausalLMClassificationHead tests.

    Replaces: A full AutoModelForCausalLM (e.g. OLMo-7B). The real model
        has a vocabulary of ~50k tokens; this stub uses 256 to keep memory
        low while still covering the Yes/No token IDs (89 and 78).
    What the head expects: CausalLMClassificationHead calls the model with
        ``(input_ids, attention_mask)`` and reads ``.logits`` of shape
        ``[batch, seq_len, vocab_size]``. It then slices out the Yes and
        No token positions from the last-token logits. No
        ``config.hidden_size`` is needed — only the logits matter.
    How it works: nn.Embedding -> nn.Linear produces a
        ``[batch, seq_len, 256]`` logits tensor.
    """

    VOCAB_SIZE = 256

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, hidden_size)
        self.lm_head = nn.Linear(hidden_size, self.VOCAB_SIZE)

    def forward(self, input_ids, attention_mask):
        h = self.embed(input_ids)
        return SimpleNamespace(logits=self.lm_head(h))


class DummyTokenizerForLM:
    """Fake tokenizer for CausalLMClassificationHead tests.

    Replaces: A real HuggingFace tokenizer (e.g. the OLMo tokenizer
        loaded by the ``load_tokenizer`` fixture in conftest.py).
    Why not use load_tokenizer: The real OLMo tokenizer encodes "Yes"
        and "No" to token IDs in the thousands (e.g. ~5765, ~2822). But
        DummyLM only has a vocabulary of 256. Indexing token ID 5765
        into a 256-wide logits tensor would crash with an IndexError.
    What the head expects: CausalLMClassificationHead calls
        ``tokenizer.encode("Yes", add_special_tokens=False)[0]`` at init
        to get the Yes/No token IDs for slicing into logits.
    How it works: Returns ``[ord(first_char)]`` for any input string,
        so "Yes" -> 89, "No" -> 78 — both safely within 256.
    """

    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(text[0])]


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
