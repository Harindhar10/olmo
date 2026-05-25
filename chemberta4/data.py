"""
Dataset classes for molecular property prediction.

Provides unified dataset interfaces for classification, regression,
pretraining, and instruction tuning tasks.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List
from transformers import PreTrainedTokenizerBase


class MoleculeNetDataset(Dataset):
    """
    A PyTorch dataset that wraps MoleculeNet CSV data for molecular property
    prediction. It supports classification (single-task and multi-task) and
    regression.

    Label processing depends on the combination of 'task_type' and
    'experiment_type':

    * **single_task classification** — rows with missing labels are dropped;
      labels are stored as integers for cross-entropy loss.
    * **multi_task classification** — all rows are kept; a boolean mask tracks
      which labels are present so that missing values (NaN) are excluded from
      the loss.
    * **Standard regression** — rows with missing labels are dropped; labels
      are stored as floats for RMSE loss.

    Examples
    --------
    >>> import pandas as pd
    >>> from transformers import AutoTokenizer
    >>> from chemberta4.data import MoleculeNetDataset
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> df = pd.DataFrame({"smiles": ["CC", "CCO"], "label": [0, 1]})
    >>> ds = MoleculeNetDataset(
    ...     df, tokenizer, ["label"], "Is it soluble?",
    ...     "single_task", "classification", max_len=32)
    >>> sample = ds[0]
    >>> list(sample.keys())
    ['text', 'labels']
    >>> sample["labels"].item()
    0
    >>> isinstance(sample["text"], str)
    True
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        task_columns: List[str],
        prompt: str,
        task_type: str,
        experiment_type: str,
        max_len: int = 128,
        smiles_column: str = "smiles",
    ) -> None:
        """Initialise MoleculeNetDataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with SMILES and target columns.
        tokenizer : PreTrainedTokenizerBase
            HuggingFace tokenizer.
        task_columns : List[str]
            Column names containing the target labels.
        prompt : str
            Task-specific prompt text prepended to each molecule.
        task_type : str
            One of 'single_task' or 'multi_task' (for classification).
        experiment_type : str
            One of 'classification' or 'regression'.
        max_len : int
            Maximum token sequence length for truncation/padding.
        smiles_column : str
            Name of the column containing SMILES strings.
        """

        self.task_type = task_type
        self.experiment_type = experiment_type
        self.num_tasks = len(task_columns)
        self.max_len = max_len

        # Process labels based on task type
        if task_type == "single_task":
            df = df.dropna(subset=task_columns).copy()
            self.labels = torch.tensor(
                df[task_columns[0]].values.astype(int), dtype=torch.long
            )
            self.label_mask = None

        elif task_type == "multi_task":
            df = df.copy()
            labels_array = df[task_columns].values.astype(np.float32)
            # Mask for missing labels (NaN values)
            self.label_mask = torch.tensor(~np.isnan(labels_array), dtype=torch.bool)
            # Replace NaN with 0 for computation
            labels_array = np.nan_to_num(labels_array, nan=0.0)
            self.labels = torch.tensor(labels_array, dtype=torch.float32)

        elif experiment_type == "regression":
            df = df.dropna(subset=task_columns).copy()
            labels = df[task_columns[0]].values.astype(np.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
            self.label_mask = None

        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

        # Build prompts; tokenization is deferred to collate_fn
        self.texts = [f"Molecule: {s}\n{prompt}" for s in df[smiles_column]]

        self.num_samples = len(df)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single dataset sample formatted for transformer training.

        Returns the raw prompt text and the sample's label. For multi-task
        datasets a boolean ``label_mask`` is included so the loss can ignore
        missing entries. Tokenization is deferred to the collate function.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dict with 'text', 'labels', and optionally 'label_mask'
            (for multi_task tasks).

        Examples
        --------
        >>> import pandas as pd
        >>> from transformers import AutoTokenizer
        >>> from chemberta4.data import MoleculeNetDataset
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> df = pd.DataFrame({"smiles": ["CC", "CCO"], "label": [0, 1]})
        >>> ds = MoleculeNetDataset(
        ...     df, tokenizer, ["label"], "Is it soluble?",
        ...     "single_task", "classification", max_len=32)
        >>> sample = ds[0]
        >>> list(sample.keys())
        ['text', 'labels']
        >>> sample["labels"].item()
        0
        >>> isinstance(sample["text"], str)
        True
        """
        item = {
            "text": self.texts[idx],
            "labels": self.labels[idx],
        }
        if self.label_mask is not None:
            item["label_mask"] = self.label_mask[idx]
        return item


class PretrainingDataset(Dataset):
    """
    This class wraps SMILES strings as a PyTorch dataset for causal language model pretraining.

    It formats each SMILES string for next-token prediction. Each SMILES is
    prefixed with 'prefix' (default "SMILES: ") and tokenised to a fixed
    length with right-padding. The 'labels' tensor is identical to
    'input_ids' except that padding positions are set to '-100' so
    PyTorch's cross-entropy ignores them.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> from chemberta4.data import PretrainingDataset
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> ds = PretrainingDataset(["CC", "CCO"], tokenizer, max_len=16)
    >>> sample = ds[0]
    >>> list(sample.keys())
    ['input_ids', 'attention_mask', 'labels']
    >>> sample["input_ids"].shape
    torch.Size([16])
    >>> (sample["labels"] == -100).any().item()
    True
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 256,
        prefix: str = "SMILES: ",
    ) -> None:
        """Initialise PretrainingDataset.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings.
        tokenizer : PreTrainedTokenizerBase
            HuggingFace tokenizer.
        max_len : int
            Maximum token sequence length for truncation/padding.
        prefix : str
            String prepended to each SMILES (default: 'SMILES: ').
        """

        texts = [f"{prefix}{s}" for s in smiles_list]

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        # Labels for causal LM: same as input_ids, mask padding
        self.labels = self.encodings["input_ids"].clone()
        self.labels[self.labels == tokenizer.pad_token_id] = -100

        self.num_samples = len(smiles_list)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single tokenized sample.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dict with 'input_ids', 'attention_mask', and 'labels'.

        Examples
        --------
        >>> from transformers import AutoTokenizer
        >>> from chemberta4.data import PretrainingDataset
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> ds = PretrainingDataset(["CC", "CCO"], tokenizer, max_len=16)
        >>> sample = ds[0]
        >>> list(sample.keys())
        ['input_ids', 'attention_mask', 'labels']
        >>> sample["input_ids"].shape
        torch.Size([16])
        >>> (sample["labels"] == -100).any().item()
        True
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class InstructionDataset(Dataset):
    """
    This class wraps instruction/input/output tuples as a PyTorch dataset for causal LM instruction tuning.

    It formats each sample as an instruction/input/output tuple for next-token prediction training. Each
    sample is formatted as '"Instruction: ...\nInput: ...\nOutput: ..."'
    and tokenised on-the-fly (lazy tokenisation). Padding tokens in 'labels'
    are masked to '-100'.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> from chemberta4.data import InstructionDataset
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> data = [{"instruction": "Predict product.", "input": "CC + O", "output": "CCO"}]
    >>> ds = InstructionDataset(data, tokenizer, max_len=32)
    >>> sample = ds[0]
    >>> list(sample.keys())
    ['input_ids', 'attention_mask', 'labels']
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 512,
    ) -> None:
        """Initialise InstructionDataset.

        Parameters
        ----------
        data : List[Dict]
            List of dicts with 'instruction', 'input', and 'output' keys.
        tokenizer : PreTrainedTokenizerBase
            HuggingFace tokenizer.
        max_len : int
            Maximum token sequence length for truncation/padding.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Materialize if streaming
        self.data = list(data)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single tokenized instruction sample.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dict with 'input_ids', 'attention_mask', and 'labels'.

        Examples
        --------
        >>> from transformers import AutoTokenizer
        >>> from chemberta4.data import InstructionDataset
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> data = [{"instruction": "Predict product.", "input": "CC + O", "output": "CCO"}]
        >>> ds = InstructionDataset(data, tokenizer, max_len=32)
        >>> sample = ds[0]
        >>> list(sample.keys())
        ['input_ids', 'attention_mask', 'labels']
        """
        item = self.data[idx]

        # Build instruction format
        prompt = (
            f"Instruction: {item['instruction']}\n"
            f"Input: {item['input']}\n"
            f"Output: {item['output']}"
        )

        encodings = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Labels: mask padding tokens
        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def make_collate_fn(tokenizer: PreTrainedTokenizerBase, max_len: int):
    """Return a collate function that tokenizes and pads to the longest sequence in the batch.

    The tokenizer is called once per batch with ``padding="longest"`` and
    ``truncation=True``, so each batch is padded only to its own longest
    sequence rather than a global maximum.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer used for encoding.
    max_len : int
        Maximum sequence length for truncation.

    Returns
    -------
    Callable
        A collate function suitable for ``torch.utils.data.DataLoader``.
    """
    def collate_fn(samples: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [s["text"] for s in samples]
        enc = tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = torch.stack([s["labels"] for s in samples])
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "label_mask" in samples[0]:
            batch["label_mask"] = torch.stack([s["label_mask"] for s in samples])
        return batch

    return collate_fn
