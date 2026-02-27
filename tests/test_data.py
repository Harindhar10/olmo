"""Unit tests for chemberta4.data."""

import numpy as np
import pandas as pd
import pytest
import torch

from chemberta4.data import InstructionDataset, MoleculeNetDataset, PretrainingDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clf_df_single():
    return pd.DataFrame({
        "smiles": ["CC", "CCO", "CCC", "CCCO"],
        "label": [0, 1, 0, 1],
    })


def _clf_df_single_with_nan():
    return pd.DataFrame({
        "smiles": ["CC", "CCO", "CCC"],
        "label": [0, float("nan"), 1],
    })


def _clf_df_multi():
    return pd.DataFrame({
        "smiles": ["CC", "CCO", "CCC"],
        "task1": [0.0, 1.0, float("nan")],
        "task2": [1.0, float("nan"), 0.0],
    })


def _reg_df():
    return pd.DataFrame({
        "smiles": ["CC", "CCO", "CCC"],
        "value": [1.0, 2.0, 3.0],
    })


# ---------------------------------------------------------------------------
# MoleculeNetDataset — single_task
# ---------------------------------------------------------------------------

class TestMoleculeNetDatasetSingleTask:
    def _make(self, tokenizer, df=None, **kwargs):
        if df is None:
            df = _clf_df_single()
        defaults = dict(
            task_columns=["label"],
            prompt="Is it active?",
            task_type="single_task",
            experiment_type="classification",
            max_len=32,
        )
        defaults.update(kwargs)
        return MoleculeNetDataset(df, tokenizer, **defaults)

    def test_len(self, tokenizer):
        ds = self._make(tokenizer)
        assert len(ds) == 4

    def test_getitem_keys(self, tokenizer):
        ds = self._make(tokenizer)
        sample = ds[0]
        assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_no_label_mask_key(self, tokenizer):
        ds = self._make(tokenizer)
        assert "label_mask" not in ds[0]

    def test_label_dtype_is_long(self, tokenizer):
        ds = self._make(tokenizer)
        assert ds[0]["labels"].dtype == torch.long

    def test_label_values(self, tokenizer):
        ds = self._make(tokenizer)
        labels = [ds[i]["labels"].item() for i in range(len(ds))]
        assert labels == [0, 1, 0, 1]

    def test_drops_nan_rows(self, tokenizer):
        df = _clf_df_single_with_nan()
        ds = self._make(tokenizer, df=df)
        # Row with NaN label is dropped → 2 samples remain
        assert len(ds) == 2

    def test_input_ids_shape(self, tokenizer):
        ds = self._make(tokenizer, max_len=20)
        assert ds[0]["input_ids"].shape == (20,)

    def test_attention_mask_shape(self, tokenizer):
        ds = self._make(tokenizer, max_len=20)
        assert ds[0]["attention_mask"].shape == (20,)

    def test_prompt_in_encoding(self, tokenizer):
        ds = self._make(tokenizer)
        # The prompt should be encoded (non-zero IDs present beyond padding)
        assert ds[0]["input_ids"].max().item() > 0

    def test_clf_prompt_format(self, tokenizer):
        """Without use_lm_head the prompt should NOT contain 'Answer:'."""
        ds = self._make(tokenizer, max_len=64)
        text = tokenizer.decode(ds[0]["input_ids"].tolist(), skip_special_tokens=True)
        assert "Answer:" not in text

    def test_lm_head_prompt_format(self, tokenizer):
        """With use_lm_head=True the prompt should contain 'Answer:'."""
        ds = self._make(tokenizer, use_lm_head=True, max_len=64)
        text = tokenizer.decode(ds[0]["input_ids"].tolist(), skip_special_tokens=True)
        assert "Answer:" in text


    def test_unknown_experiment_type_raises(self, tokenizer):
        # task_type must NOT be "single_task" or "multi_task" to reach the else branch.
        with pytest.raises(ValueError, match="Unknown experiment_type"):
            MoleculeNetDataset(
                _clf_df_single(),
                tokenizer,
                task_columns=["label"],
                prompt="x",
                task_type="other_type",
                experiment_type="unknown_type",
                max_len=16,
            )


# ---------------------------------------------------------------------------
# MoleculeNetDataset — multi_task
# ---------------------------------------------------------------------------

class TestMoleculeNetDatasetMultiTask:
    def _make(self, tokenizer, df=None, **kwargs):
        if df is None:
            df = _clf_df_multi()
        defaults = dict(
            task_columns=["task1", "task2"],
            prompt="Is it toxic?",
            task_type="multi_task",
            experiment_type="classification",
            max_len=32,
        )
        defaults.update(kwargs)
        return MoleculeNetDataset(df, tokenizer, **defaults)

    def test_label_mask_present(self, tokenizer):
        ds = self._make(tokenizer)
        assert "label_mask" in ds[0]

    def test_label_mask_shape(self, tokenizer):
        ds = self._make(tokenizer)
        assert ds[0]["label_mask"].shape == (2,)

    def test_nan_position_masked_false(self, tokenizer):
        ds = self._make(tokenizer)
        # Row 0: task1=0, task2=1 → both valid
        assert ds[0]["label_mask"].all()
        # Row 2: task1=NaN, task2=0 → task1 invalid
        assert ds[2]["label_mask"][0].item() is False
        assert ds[2]["label_mask"][1].item() is True

    def test_label_dtype_float32(self, tokenizer):
        ds = self._make(tokenizer)
        assert ds[0]["labels"].dtype == torch.float32

    def test_nan_replaced_with_zero(self, tokenizer):
        ds = self._make(tokenizer)
        # Row 1: task2=NaN → replaced with 0
        assert ds[1]["labels"][1].item() == pytest.approx(0.0)

    def test_len(self, tokenizer):
        ds = self._make(tokenizer)
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# MoleculeNetDataset — regression
# ---------------------------------------------------------------------------

class TestMoleculeNetDatasetRegression:
    def _make(self, tokenizer, df=None, **kwargs):
        if df is None:
            df = _reg_df()
        defaults = dict(
            task_columns=["value"],
            prompt="Predict logP.",
            task_type="regression",
            experiment_type="regression",
            max_len=32,
        )
        defaults.update(kwargs)
        return MoleculeNetDataset(df, tokenizer, **defaults)

    def test_labels_are_z_scored(self, tokenizer):
        ds = self._make(tokenizer)
        # NumPy uses population std (ddof=0) by default.
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mean, std = values.mean(), values.std()
        expected = (values - mean) / std
        labels = [ds[i]["labels"].item() for i in range(3)]
        assert labels[0] == pytest.approx(float(expected[0]), abs=1e-5)
        assert labels[1] == pytest.approx(float(expected[1]), abs=1e-5)
        assert labels[2] == pytest.approx(float(expected[2]), abs=1e-5)

    def test_label_mask_is_none_for_regression(self, tokenizer):
        ds = self._make(tokenizer)
        assert "label_mask" not in ds[0]

    def test_drops_nan_rows_regression(self, tokenizer):
        df = pd.DataFrame({"smiles": ["CC", "CCO"], "value": [1.0, float("nan")]})
        ds = self._make(tokenizer, df=df)
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# PretrainingDataset
# ---------------------------------------------------------------------------

class TestPretrainingDataset:
    SMILES = ["CC", "CCO", "c1ccccc1"]

    def _make(self, tokenizer, smiles=None, **kwargs):
        if smiles is None:
            smiles = self.SMILES
        defaults = dict(max_len=16)
        defaults.update(kwargs)
        return PretrainingDataset(smiles, tokenizer, **defaults)

    def test_len(self, tokenizer):
        ds = self._make(tokenizer)
        assert len(ds) == 3

    def test_getitem_keys(self, tokenizer):
        ds = self._make(tokenizer)
        assert set(ds[0].keys()) == {"input_ids", "attention_mask", "labels", "num_bytes"}

    def test_labels_have_minus100_at_padding(self, tokenizer):
        ds = self._make(tokenizer)
        # At least one padding position should be masked to -100
        assert (ds[0]["labels"] == -100).any()

    def test_real_tokens_not_minus100(self, tokenizer):
        ds = self._make(tokenizer)
        # The first token is always real (not padding)
        assert ds[0]["labels"][0].item() != -100

    def test_num_bytes_correct_default_prefix(self, tokenizer):
        ds = self._make(tokenizer, smiles=["CC"])
        expected = len("SMILES: CC".encode("utf-8"))
        assert ds[0]["num_bytes"].item() == expected

    def test_num_bytes_custom_prefix(self, tokenizer):
        ds = self._make(tokenizer, smiles=["CC"], prefix="MOL: ")
        expected = len("MOL: CC".encode("utf-8"))
        assert ds[0]["num_bytes"].item() == expected

    def test_input_ids_shape(self, tokenizer):
        ds = self._make(tokenizer, max_len=16)
        assert ds[0]["input_ids"].shape == (16,)

    def test_labels_shape_matches_input_ids(self, tokenizer):
        ds = self._make(tokenizer)
        for i in range(len(ds)):
            assert ds[i]["labels"].shape == ds[i]["input_ids"].shape


# ---------------------------------------------------------------------------
# InstructionDataset
# ---------------------------------------------------------------------------

class TestInstructionDataset:
    DATA = [
        {"instruction": "Predict product.", "input": "CC + O", "output": "CCO"},
        {"instruction": "Name this.", "input": "c1ccccc1", "output": "benzene"},
    ]

    def _make(self, tokenizer, data=None, **kwargs):
        if data is None:
            data = self.DATA
        defaults = dict(max_len=32)
        defaults.update(kwargs)
        return InstructionDataset(data, tokenizer, **defaults)

    def test_len(self, tokenizer):
        ds = self._make(tokenizer)
        assert len(ds) == 2

    def test_getitem_keys(self, tokenizer):
        ds = self._make(tokenizer)
        assert set(ds[0].keys()) == {"input_ids", "attention_mask", "labels", "num_bytes"}

    def test_num_bytes_correct(self, tokenizer):
        ds = self._make(tokenizer)
        item = self.DATA[0]
        expected = len(
            f"Instruction: {item['instruction']}\n"
            f"Input: {item['input']}\n"
            f"Output: {item['output']}".encode("utf-8")
        )
        assert ds[0]["num_bytes"].item() == expected

    def test_labels_have_minus100_at_padding(self, tokenizer):
        ds = self._make(tokenizer, max_len=64)
        assert (ds[0]["labels"] == -100).any()

    def test_input_ids_shape(self, tokenizer):
        ds = self._make(tokenizer, max_len=32)
        assert ds[0]["input_ids"].shape == (32,)

    def test_labels_shape_matches_input_ids(self, tokenizer):
        ds = self._make(tokenizer)
        for i in range(len(ds)):
            assert ds[i]["labels"].shape == ds[i]["input_ids"].shape

    def test_num_bytes_is_positive(self, tokenizer):
        ds = self._make(tokenizer)
        for i in range(len(ds)):
            assert ds[i]["num_bytes"].item() > 0

    def test_accepts_streaming_data(self, tokenizer):
        """InstructionDataset should materialise generators."""
        data_gen = iter(self.DATA)
        ds = InstructionDataset(data_gen, tokenizer, max_len=32)
        assert len(ds) == 2
