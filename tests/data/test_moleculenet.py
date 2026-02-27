import pandas as pd
import pytest
import torch

from chemberta4.data import MoleculeNetDataset


class TestMoleculeNetSingleTask:
    """Tests for MoleculeNetDataset in single_task classification mode.

    Single-task classification is the simplest label regime: one binary
    column per molecule, NaN rows dropped, labels cast to torch.long for
    CrossEntropyLoss compatibility.
    """

    def get_sample_dataset(self, load_tokenizer, df=None, **kwargs) -> MoleculeNetDataset:
        if df is None:
            df = pd.DataFrame({
                "smiles": ["CC", "CCO", "CCC", "CCCO"],
                "label": [0, 1, 0, 1],
            })
        defaults = dict(
            task_columns=["label"],
            prompt="Is it active?",
            task_type="single_task",
            experiment_type="classification",
            max_len=32,
        )
        defaults.update(kwargs)
        return MoleculeNetDataset(df, load_tokenizer, **defaults)

    def test_output_structure_and_label_dtype(self, load_tokenizer):
        """Verify that samples contain the expected keys with correct tensor
        shapes and that labels are torch.long.

        The training loop indexes samples by these exact keys and passes labels
        to CrossEntropyLoss, which requires long-typed targets. A wrong key set
        or dtype would cause a runtime crash mid-training.
        """
        ds = self.get_sample_dataset(load_tokenizer)
        sample = ds[0]
        assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}
        assert sample["input_ids"].shape == (32,)
        assert sample["attention_mask"].shape == (32,)
        assert sample["labels"].dtype == torch.long
        assert "label_mask" not in sample

    def test_nan_rows_are_dropped(self, load_tokenizer):
        """Verify that rows with NaN labels are excluded from the dataset.

        MoleculeNet CSVs often contain NaN for unmeasured endpoints. Feeding
        NaN labels to the loss function would produce NaN gradients and corrupt
        the model, so they must be silently dropped during construction.
        """
        df = pd.DataFrame({
            "smiles": ["CC", "CCO", "CCC"],
            "label": [0, float("nan"), 1],
        })
        ds = self.get_sample_dataset(load_tokenizer, df=df)
        assert len(ds) == 2

    def test_label_values_preserved(self, load_tokenizer):
        """Verify that integer label values survive the DataFrame-to-tensor
        conversion without reordering or corruption.

        A subtle bug in indexing or type casting could silently shuffle labels,
        leading to a model that trains on wrong targets.
        """
        ds = self.get_sample_dataset(load_tokenizer)
        labels = [ds[i]["labels"].item() for i in range(len(ds))]
        assert labels == [0, 1, 0, 1]


class TestMoleculeNetMultiTask:
    """Tests for MoleculeNetDataset in multi_task classification mode.

    Multi-task datasets like Tox21 have multiple label columns where some
    entries are NaN (unmeasured). The dataset must produce a boolean
    label_mask so the loss function can ignore missing targets, and must
    replace NaN values with 0 to avoid tensor contamination.
    """

    def get_sample_dataset(self, load_tokenizer, **kwargs):
        df = pd.DataFrame({
            "smiles": ["CC", "CCO", "CCC"],
            "task1": [0.0, 1.0, float("nan")],
            "task2": [1.0, float("nan"), 0.0],
        })
        defaults = dict(
            task_columns=["task1", "task2"],
            prompt="Is it toxic?",
            task_type="multi_task",
            experiment_type="classification",
            max_len=32,
        )
        defaults.update(kwargs)
        return MoleculeNetDataset(df, load_tokenizer, **defaults)

    def test_nan_masking_and_replacement(self, load_tokenizer):
        """Verify that NaN positions are marked False in label_mask and that
        their corresponding label values are replaced with 0.

        The loss function multiplies by label_mask to zero out gradients for
        unmeasured tasks. If NaN values leak into the labels tensor instead of
        being replaced, they propagate NaN through the entire backward pass.
        """
        ds = self.get_sample_dataset(load_tokenizer)
        # Row 0: both tasks present
        assert ds[0]["label_mask"].all()
        # Row 2: task1=NaN, task2=0
        assert ds[2]["label_mask"][0].item() is False
        assert ds[2]["label_mask"][1].item() is True
        # Row 1: task2=NaN → replaced with 0
        assert ds[1]["labels"][1].item() == pytest.approx(0.0)

    def test_label_dtype_is_float32(self, load_tokenizer):
        """Verify that multi-task labels are float32.

        Multi-task classification uses BCEWithLogitsLoss which requires
        float targets, unlike single-task CrossEntropyLoss which needs long.
        """
        ds = self.get_sample_dataset(load_tokenizer)
        assert ds[0]["labels"].dtype == torch.float32
        assert ds[0]["label_mask"].shape == (2,)


class TestMoleculeNetRegression:
    """Tests for MoleculeNetDataset in regression mode.

    Regression targets are stored as float32 tensors, NaN rows are dropped,
    and no label_mask is produced (single continuous target per molecule).
    """

    def get_sample_dataset(self, load_tokenizer, df=None, **kwargs):
        if df is None:
            df = pd.DataFrame({
                "smiles": ["CC", "CCO", "CCC"],
                "value": [1.0, 2.0, 3.0],
            })
        defaults = dict(
            task_columns=["value"],
            prompt="Predict logP.",
            task_type="regression",
            experiment_type="regression",
            max_len=32,
        )
        defaults.update(kwargs)
        return MoleculeNetDataset(df, load_tokenizer, **defaults)

    def test_regression_labels_and_nan_dropping(self, load_tokenizer):
        """Verify that regression labels are float32 tensors and that rows
        with NaN targets are excluded.

        MSELoss requires float targets, and NaN values would produce NaN
        loss. Both constraints must hold for stable training.
        """
        ds = self.get_sample_dataset(load_tokenizer)
        assert ds[0]["labels"].dtype == torch.float32
        assert "label_mask" not in ds[0]

        df_with_nan = pd.DataFrame({
            "smiles": ["CC", "CCO"],
            "value": [1.0, float("nan")],
        })
        ds_nan = self.get_sample_dataset(load_tokenizer, df=df_with_nan)
        assert len(ds_nan) == 1


class TestMoleculeNetErrors:
    """Tests for MoleculeNetDataset input validation."""

    def test_invalid_experiment_type_raises(self, load_tokenizer):
        """Verify that an unrecognised experiment_type raises ValueError.

        The constructor dispatches label processing by experiment_type. An
        unknown value would silently skip label setup, causing cryptic
        AttributeErrors later. A clear ValueError at construction time is
        far easier to debug.
        """
        with pytest.raises(ValueError, match="Unknown experiment_type"):
            MoleculeNetDataset(
                pd.DataFrame({"smiles": ["CC"], "label": [0]}),
                load_tokenizer,
                task_columns=["label"],
                prompt="x",
                task_type="other_type",
                experiment_type="unknown_type",
                max_len=16,
            )

class TestMoleculeNetCLMRegression:
    """Tests for MoleculeNetDataset in causal LM regression mode.

    When use_lm_head=True and experiment_type='regression', the dataset
    embeds the regression target directly in the prompt text and tokenizes
    lazily per sample. The prompt portion of labels is masked to -100 so
    the loss is computed only on the response tokens (teacher forcing).
    """

    def get_sample_dataset(self, load_tokenizer, df=None, **kwargs):
        if df is None:
            df = pd.DataFrame({
                "smiles": ["CC", "CCO", "CCC"],
                "value": [1.0, 2.0, 3.0],
            })
        defaults = dict(
            task_columns=["value"],
            prompt="Predict logP.",
            task_type="regression",
            experiment_type="regression",
            use_lm_head=True,
            max_len=64,
        )
        defaults.update(kwargs)
        return MoleculeNetDataset(df, load_tokenizer, **defaults)

    def test_clm_regression_output_structure(self, load_tokenizer):
        """Verify that CLM regression samples return the expected keys
        including label_values, and that prompt tokens are masked in labels.

        Unlike standard regression which returns a scalar label, CLM
        regression returns full-sequence labels for next-token prediction
        plus label_values for metric computation. The prompt portion must
        be masked to -100 so the model is only trained to generate the
        numeric response, not to memorise the prompt.
        """
        ds = self.get_sample_dataset(load_tokenizer)
        sample = ds[0]
        assert set(sample.keys()) == {"input_ids", "attention_mask", "labels", "label_values"}
        assert sample["input_ids"].shape == sample["labels"].shape
        assert sample["label_values"].dtype == torch.float32
        # Prompt tokens should be masked
        assert (sample["labels"] == -100).any(), "prompt/padding tokens should be masked"
        # At least some response tokens should NOT be masked
        assert (sample["labels"] != -100).any(), "response tokens should not be masked"

