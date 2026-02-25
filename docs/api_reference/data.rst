chemberta4.data
===============

Dataset classes for classification, regression, pretraining, and instruction tuning.
All datasets return dictionaries of ``torch.Tensor`` compatible with PyTorch DataLoader.

.. class:: MoleculeNetDataset(Dataset)

   Unified dataset for MoleculeNet classification and regression tasks.
   Handles single-task, multi-task, and LM-head prompt formatting.
   Supports optional z-score normalization for regression labels.

   .. method:: __init__(df, tokenizer, task_columns, prompt, task_type, experiment_type, max_len=128, use_lm_head=False, label_stats=None, smiles_column="smiles")

      Initializes the dataset from a DataFrame with SMILES strings and target columns.

   .. method:: __len__()

      Returns the number of samples in the dataset.

   .. method:: __getitem__(idx)

      Returns a dict of tensors with ``input_ids``, ``attention_mask``, ``labels``,
      and optionally ``label_mask`` for multi-task datasets.

   .. method:: get_label_stats()

      Returns ``{'mean': float, 'std': float}`` for regression datasets, or ``None``
      for classification datasets.

.. class:: PretrainingDataset(Dataset)

   Dataset for causal language model pretraining on SMILES strings.
   Formats each SMILES as ``"{prefix}{smiles}"`` for next-token prediction.

   .. method:: __init__(smiles_list, tokenizer, max_len=256, prefix="SMILES: ")

      Initializes the dataset from a list of SMILES strings.

   .. method:: __len__()

      Returns the number of SMILES strings.

   .. method:: __getitem__(idx)

      Returns a dict with ``input_ids``, ``attention_mask``, ``labels``,
      and ``num_bytes`` for bits-per-byte (BPB) calculation.

.. class:: InstructionDataset(Dataset)

   Dataset for instruction tuning on reaction or chemistry tasks (e.g., USPTO).
   Formats each sample as ``Instruction / Input / Output`` and masks input tokens
   from the loss so only output tokens are trained on.

   .. method:: __init__(data, tokenizer, max_len=512)

      Initializes the dataset from a list of instruction/input/output dicts.

   .. method:: __len__()

      Returns the number of instruction samples.

   .. method:: __getitem__(idx)

      Returns a dict with ``input_ids``, ``attention_mask``, ``labels``,
      and ``num_bytes`` for BPB calculation.
