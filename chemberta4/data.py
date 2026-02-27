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
    TODO: Update docstring to reflect recent change.
    This class wraps MoleculeNet CSV splits into a PyTorch dataset for classification and regression tasks.

    It handles single-task and multi-task classification as well as regression.
    It supports both classification head and LM head prompt formats.

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
            If 'True', format prompts for Yes/No LM-head prediction
            (classification) or embed the answer in the text for teacher-forced
            causal LM regression.
        smiles_column : str
            Name of the column containing SMILES strings.
        """

        self.task_type = task_type
        self.experiment_type = experiment_type
        self.num_tasks = len(task_columns)
        self.use_lm_head = use_lm_head

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

        elif use_lm_head and experiment_type == "regression":
            # CLM regression: embed the answer in the text; tokenize lazily
            # per sample in __getitem__ using the reference's split-separator
            # approach to avoid BPE tokenization boundary issues.
            _SEPARATOR = "### Response:\n"
            df = df.dropna(subset=task_columns).copy()
            labels = df[task_columns[0]].values.astype(np.float32)

            self._clm_texts = [
                f"Molecule: {s}\nQuestion: {prompt}\n{_SEPARATOR}{v:.5f}{tokenizer.eos_token}"
                for s, v in zip(df[smiles_column], labels)
            ]
            self._clm_separator = _SEPARATOR
            self._tokenizer = tokenizer
            self._max_len = max_len
            self.label_values = torch.tensor(labels, dtype=torch.float32)
            self.label_mask = None
            self.num_samples = len(df)
            return  # __getitem__ handles tokenization for CLM regression

        elif experiment_type == "regression":
            df = df.dropna(subset=task_columns).copy()
            labels = df[task_columns[0]].values.astype(np.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
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
        """
        Retrieve a single dataset sample formatted for transformer training.

        This method supports two modes of operation:

        1. Standard encoder-style training (classification or regression)
        Returns pre-tokenized inputs stored in ``self.encodings`` along with
        their corresponding labels. Optionally includes a ``label_mask`` for
        multi-task setups with missing labels.

        2. Causal language modeling (CLM) regression mode
        Triggered when ``self.use_lm_head`` is True and
        ``self.experiment_type == "regression"``.
        In this case:
            - The full prompt + target text is tokenized.
            - ``labels`` are initialized as a clone of ``input_ids``.
            - Tokens corresponding to the prompt portion (before
            ``self._clm_separator``) are masked with -100 so that loss is
            computed only on the target portion.
            - Padding tokens are also masked with -100.
            - The original scalar regression value is returned separately as
            ``label_values``.

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
        if self.use_lm_head and self.experiment_type == "regression":
            text = self._clm_texts[idx]
            enc = self._tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self._max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            parts = text.split(self._clm_separator)
            if len(parts) >= 2:
                prompt_text = parts[0] + self._clm_separator
                prompt_enc = self._tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=self._max_len,
                    return_tensors="pt",
                )
                prompt_len = prompt_enc["input_ids"].shape[1]
                if prompt_len < len(labels):
                    labels[:prompt_len] = -100

            labels[labels == self._tokenizer.pad_token_id] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "label_values": self.label_values[idx],
            }

        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
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
        }

