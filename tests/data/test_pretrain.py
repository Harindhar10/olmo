import pandas as pd
import pytest
import torch

from chemberta4.data import PretrainingDataset


class TestPretrainingDataset:
    """Tests for PretrainingDataset.

    PretrainingDataset formats SMILES for causal LM next-token prediction.
    The critical behaviors are: padding tokens must be masked to -100 in
    labels so cross-entropy ignores them.
    """

    def get_sample_dataset(self, load_tokenizer, smiles=None, **kwargs):
        if smiles is None:
            smiles = ["CC", "CCO", "c1ccccc1"]
        defaults = dict(max_len=16)
        defaults.update(kwargs)
        return PretrainingDataset(smiles, load_tokenizer, **defaults)

    def test_output_structure(self, load_tokenizer):
        """Verify that samples contain all required keys with matching shapes.

        The training loop expects exactly these four keys. A missing key or
        shape mismatch would crash the forward pass or loss computation.
        """
        ds = self.get_sample_dataset(load_tokenizer)
        sample = ds[0]
        assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}
        assert sample["labels"].shape == sample["input_ids"].shape

    def test_padding_masked_in_labels(self, load_tokenizer):
        """Verify that padding positions are -100 in labels while real tokens
        are not.

        PyTorch's CrossEntropyLoss ignores targets set to -100. Without this
        masking, the model would be trained to predict pad tokens, degrading
        generation quality.
        """
        ds = self.get_sample_dataset(load_tokenizer)
        labels = ds[0]["labels"]
        assert (labels == -100).any(), "padding positions should be masked"
        assert labels[0].item() != -100, "first token should be real"
