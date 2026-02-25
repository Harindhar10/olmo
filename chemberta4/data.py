"""
Dataset classes for molecular property prediction.

Provides unified dataset interfaces for classification, regression,
pretraining, and instruction tuning tasks.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Any
from transformers import PreTrainedTokenizerBase


class MoleculeNetDataset(Dataset):
    """
    Dataset class for MoleculeNet datasets.

    Handles single-task, multi-task classification and regression.
    Supports both classification head and LM head prompt formats.

    The class handles three distinct label regimes: binary 'single_task'
    (CrossEntropy-compatible 'long' labels), 'multi_task' with NaN masking
    (for datasets like Tox21 where some tasks may be missing for a molecule),
    and 'regression' with z-score normalisation computed from the training
    set and applied at inference via 'label_stats'. When 'use_lm_head=True'
    the prompt is reformatted as an 'Answer:' completion so that the LM head
    can score 'Yes'/'No' token logits instead of a linear classifier.

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
    ['input_ids', 'attention_mask', 'labels']
    >>> sample["labels"].item()
    0
    >>> sample["input_ids"].shape
    torch.Size([32])
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
        use_lm_head: bool = False,
        label_stats: Optional[Dict[str, float]] = None,
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
        use_lm_head : bool
            If 'True', format prompts for Yes/No LM-head prediction.
        label_stats : Dict[str, float], optional
            Dict with 'mean' and 'std' keys for regression normalization.
            Pass training-set stats when creating val/test datasets.
        smiles_column : str
            Name of the column containing SMILES strings.
        """

        self.task_type = task_type
        self.experiment_type = experiment_type
        self.num_tasks = len(task_columns)
        self.label_mean = 0.0
        self.label_std = 1.0

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

            if label_stats is None:
                # Compute normalization stats from this data (training set)
                self.label_mean = float(labels.mean())
                self.label_std = float(labels.std())
            else:
                # Use provided stats (for val/test sets)
                self.label_mean = label_stats["mean"]
                self.label_std = label_stats["std"]

            normalized = (labels - self.label_mean) / self.label_std
            self.labels = torch.tensor(normalized, dtype=torch.float32)
            self.label_mask = None

        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

        # Build prompts
        if use_lm_head:
            texts = [
                f"Molecule: {s}\nQuestion: {prompt}\nAnswer:"
                for s in df[smiles_column]
            ]
        else:
            texts = [f"Molecule: {s}\n{prompt}" for s in df[smiles_column]]

        # Tokenize all at once
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.num_samples = len(df)

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
            Dict with 'input_ids', 'attention_mask', 'labels', and
            optionally 'label_mask' (for multi_task tasks).

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
        ['input_ids', 'attention_mask', 'labels']
        >>> sample["labels"].item()
        0
        >>> sample["input_ids"].shape
        torch.Size([32])
        """
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
        if self.label_mask is not None:
            item["label_mask"] = self.label_mask[idx]
        return item

    def get_label_stats(self) -> Optional[Dict[str, float]]:
        """Return label normalization statistics for regression tasks.

        Intended to be called on the training split and the result passed as
        'label_stats' to validation and test splits so that all splits are
        normalised with the same statistics. Returns 'None' for
        classification tasks where no normalisation is applied.

        Returns
        -------
        Dict[str, float] or None
            Dict with 'mean' and 'std' if task is regression,
            otherwise None.

        Examples
        --------
        >>> import pandas as pd
        >>> from transformers import AutoTokenizer
        >>> from chemberta4.data import MoleculeNetDataset
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> df = pd.DataFrame({"smiles": ["CC", "CCO", "CCC"], "value": [1.0, 2.0, 3.0]})
        >>> ds = MoleculeNetDataset(
        ...     df, tokenizer, ["value"], "Predict logP.",
        ...     "regression", "regression", max_len=32)
        >>> stats = ds.get_label_stats()
        >>> list(stats.keys())
        ['mean', 'std']
        >>> round(stats["mean"], 1)
        2.0
        """
        if self.experiment_type == "regression":
            return {"mean": self.label_mean, "std": self.label_std}
        return None


class PretrainingDataset(Dataset):
    """
    Dataset for causal language modeling pretraining on SMILES.

    Simply formats SMILES strings for next-token prediction. Each SMILES is
    prefixed with 'prefix' (default '"SMILES: "') and tokenised to a fixed
    length with right-padding. The 'labels' tensor is identical to
    'input_ids' except that padding positions are set to '-100' so
    PyTorch's cross-entropy ignores them. A 'num_bytes' tensor records the
    UTF-8 byte length of each formatted string, which is used by the
    validation loop to compute bits-per-byte (BPB).

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> from chemberta4.data import PretrainingDataset
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token
    >>> ds = PretrainingDataset(["CC", "CCO"], tokenizer, max_len=16)
    >>> sample = ds[0]
    >>> list(sample.keys())
    ['input_ids', 'attention_mask', 'labels', 'num_bytes']
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

        # Byte counts for BPB calculation
        self.num_bytes = torch.tensor(
            [len(t.encode("utf-8")) for t in texts], dtype=torch.long
        )

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
            Dict with 'input_ids', 'attention_mask', 'labels', and
            'num_bytes' (byte count used for BPB calculation).

        Examples
        --------
        >>> from transformers import AutoTokenizer
        >>> from chemberta4.data import PretrainingDataset
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> ds = PretrainingDataset(["CC", "CCO"], tokenizer, max_len=16)
        >>> sample = ds[0]
        >>> list(sample.keys())
        ['input_ids', 'attention_mask', 'labels', 'num_bytes']
        >>> sample["input_ids"].shape
        torch.Size([16])
        >>> (sample["labels"] == -100).any().item()
        True
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
            "num_bytes": self.num_bytes[idx],
        }


class InstructionDataset(Dataset):
    """
    Dataset for instruction tuning (USPTO-style).

    Formats instruction/input/output tuples for causal LM training. Each
    sample is formatted as '"Instruction: ...\nInput: ...\nOutput: ..."'
    and tokenised on-the-fly (lazy tokenisation). Padding tokens in 'labels'
    are masked to '-100'. A 'num_bytes' field for BPB tracking is precomputed
    at '__init__' time to avoid repeated string encoding during training.

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
    ['input_ids', 'attention_mask', 'labels', 'num_bytes']
    >>> sample["num_bytes"].item() > 0
    True
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

        # Precompute byte counts for BPB calculation
        self.num_bytes = [
            len(
                f"Instruction: {item['instruction']}\n"
                f"Input: {item['input']}\n"
                f"Output: {item['output']}".encode("utf-8")
            )
            for item in self.data
        ]

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
            Dict with 'input_ids', 'attention_mask', 'labels', and
            'num_bytes' (byte count used for BPB calculation).

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
        ['input_ids', 'attention_mask', 'labels', 'num_bytes']
        >>> sample["num_bytes"].item() > 0
        True
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
            "num_bytes": torch.tensor(self.num_bytes[idx], dtype=torch.long),
        }

