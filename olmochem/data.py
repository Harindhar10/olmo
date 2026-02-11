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


class MoleculeDataset(Dataset):
    """
    Dataset class for MoleculeNet datasets.

    Handles binary, multilabel, multitask classification and regression.
    Supports both classification head and LM head prompt formats.

    Args:
        df: DataFrame with SMILES and target columns
        tokenizer: HuggingFace tokenizer
        task_columns: List of column names containing labels
        prompt: Task-specific prompt text
        task_type: 'binary', 'multilabel', 'multitask', or 'regression'
        max_len: Maximum sequence length for tokenization
        use_lm_head: If True, format prompts for Yes/No prediction
        label_stats: Dict with 'mean' and 'std' for regression (use training stats)
        smiles_column: Name of column containing SMILES strings
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        task_columns: List[str],
        prompt: str,
        task_type: str,
        max_len: int = 128,
        use_lm_head: bool = False,
        label_stats: Optional[Dict[str, float]] = None,
        smiles_column: str = "smiles",
    ):
        self.task_type = task_type
        self.num_tasks = len(task_columns)
        self.label_mean = 0.0
        self.label_std = 1.0

        # Process labels based on task type
        if task_type == "binary":
            df = df.dropna(subset=task_columns).copy()
            self.labels = torch.tensor(
                df[task_columns[0]].values.astype(int), dtype=torch.long
            )
            self.label_mask = None

        elif task_type in ("multilabel", "multitask"):
            df = df.copy()
            labels_array = df[task_columns].values.astype(np.float32)
            # Mask for missing labels (NaN values)
            self.label_mask = torch.tensor(~np.isnan(labels_array), dtype=torch.bool)
            # Replace NaN with 0 for computation
            labels_array = np.nan_to_num(labels_array, nan=0.0)
            self.labels = torch.tensor(labels_array, dtype=torch.float32)

        elif task_type == "regression":
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
            raise ValueError(f"Unknown task_type: {task_type}")

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
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
        if self.label_mask is not None:
            item["label_mask"] = self.label_mask[idx]
        return item

    def get_label_stats(self) -> Optional[Dict[str, float]]:
        """Return label normalization stats for regression tasks."""
        if self.task_type == "regression":
            return {"mean": self.label_mean, "std": self.label_std}
        return None


class PretrainingDataset(Dataset):
    """
    Dataset for causal language modeling pretraining on SMILES.

    Simply formats SMILES strings for next-token prediction.

    Args:
        smiles_list: List of SMILES strings
        tokenizer: HuggingFace tokenizer
        max_len: Maximum sequence length
        prefix: Prefix before each SMILES (default: "SMILES: ")
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer,
        max_len: int = 256,
        prefix: str = "SMILES: ",
    ):
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
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class InstructionDataset(Dataset):
    """
    Dataset for instruction tuning (USPTO-style).

    Formats instruction/input/output tuples for causal LM training.

    Args:
        data: Iterable of dicts with 'instruction', 'input', 'output' keys
        tokenizer: HuggingFace tokenizer
        max_len: Maximum sequence length
    """

    def __init__(
        self,
        data,
        tokenizer,
        max_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Materialize if streaming
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


def create_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Optional[Dataset] = None,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Create dataloaders with standard settings.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Optional test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of data loader workers

    Returns:
        Dict with 'train', 'val', and optionally 'test' dataloaders
    """
    from torch.utils.data import DataLoader

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    loaders = {
        "train": DataLoader(train_ds, shuffle=True, **common_kwargs),
        "val": DataLoader(val_ds, shuffle=False, **common_kwargs),
    }

    if test_ds is not None:
        loaders["test"] = DataLoader(test_ds, shuffle=False, **common_kwargs)

    return loaders
