import pandas as pd
import pytest
import torch

from chemberta4.data import InstructionDataset


class TestInstructionDataset:
    """Tests for InstructionDataset.

    InstructionDataset formats instruction/input/output tuples for causal LM
    fine-tuning. It tokenizes lazily per sample and must correctly mask
    padding in labels.
    """

    DATA = [
        {"instruction": "Predict product.", "input": "CC + O", "output": "CCO"},
        {"instruction": "Name this.", "input": "c1ccccc1", "output": "benzene"},
    ]

    def get_sample_dataset(self, load_tokenizer, data=None, **kwargs):
        if data is None:
            data = self.DATA
        defaults = dict(max_len=32)
        defaults.update(kwargs)
        return InstructionDataset(data, load_tokenizer, **defaults)

    def test_output_structure_and_padding_mask(self, load_tokenizer):
        """Verify output keys, tensor shapes, and that padding positions in
        labels are masked to -100.

        The training loop expects these exact keys with matching shapes.
        Unmasked padding would train the model to generate pad tokens instead
        of meaningful completions.
        """
        ds = self.get_sample_dataset(load_tokenizer, max_len=64)
        sample = ds[0]
        assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}
        assert sample["labels"].shape == sample["input_ids"].shape
        assert (sample["labels"] == -100).any()

