chemberta4.training
-------------------

End-to-end experiment runners that assemble datasets, models, callbacks, and
PyTorch Lightning ``Trainer`` for each task type.

.. function:: chemberta4.training.pretrain.load_smiles_data(args: SimpleNamespace) -> List[str]

   Loads a SMILES dataset from HuggingFace Hub based on ``args.dataset``.
   Supports ``'zinc20'``, ``'pubchem'``, and ``'custom'`` datasets.

.. function:: chemberta4.training.pretrain.run_pretraining_experiment(args: SimpleNamespace, task_name: str) -> None

   Runs a full causal LM pretraining experiment on a SMILES dataset.

   1. Loads SMILES data and splits into train/validation sets.
   2. Creates ``PretrainingDataset`` instances and ``OLMoPretrainer``.
   3. Runs ``pl.Trainer`` with DDP support.
   4. Optionally merges LoRA adapter and pushes to HuggingFace Hub.

.. function:: chemberta4.training.train_classification.run_classification_experiment(args: SimpleNamespace, task_name: str) -> None

   Trains and evaluates an OLMo classifier on a MoleculeNet classification task.

   1. Loads task config, tokenizer, and train/val/test CSV files.
   2. Creates ``MoleculeNetDataset`` instances and ``OLMoClassifier``.
   3. Configures ``EarlyStopping``, ``ModelCheckpoint``, and optional W&B logging.
   4. Trains via ``pl.Trainer`` and evaluates on the test set.

.. function:: chemberta4.training.train_regression.run_regression_experiment(args: SimpleNamespace, task_name: str) -> None

   Trains and evaluates an OLMo regressor on a MoleculeNet regression task.

   1. Loads task config and computes label normalization statistics from training data.
   2. Creates ``MoleculeNetDataset`` instances with shared stats and ``OLMoRegressor``.
   3. Trains via ``pl.Trainer`` and evaluates on the test set.

.. function:: chemberta4.training.train_instruction.run_instruction_experiment(args: SimpleNamespace, task_name: str) -> None

   Runs instruction tuning on a chemistry instruction dataset (e.g., USPTO).

   1. Loads a streaming HuggingFace dataset and splits into train/validation sets.
   2. Creates ``InstructionDataset`` instances and ``OLMoPretrainer``.
   3. Runs ``pl.Trainer`` with DDP support.
   4. Optionally merges LoRA adapter and pushes to HuggingFace Hub.
