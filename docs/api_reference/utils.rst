chemberta4.utils
================

Distributed training utilities for rank-aware operations. Follows nanochat patterns
for explicit, simple multi-GPU support.

.. function:: log0(msg: str) -> None

   Logs a message only on rank 0 to avoid duplicate output in DDP training.

.. function:: get_rank() -> int

   Returns the current process rank in a distributed training setup.
   Returns ``0`` on single-GPU or CPU setups.

.. function:: is_main_process() -> bool

   Checks whether the current process is the main process (rank 0).

.. function:: set_seed(seed: int = 42) -> None

   Sets random seeds across Python, NumPy, PyTorch, and CUDA for reproducibility.

.. function:: get_device_map(device: torch.device) -> Dict[str, Any]

   Returns a device map for model loading compatible with DDP.
   Returns ``{"": device.index}`` for CUDA, or ``{"": "cpu"}`` for CPU.

.. function:: load_config(config_path: str) -> SimpleNamespace

   Loads a YAML configuration file and returns it as a ``SimpleNamespace``
   so fields are accessible as attributes (e.g., ``config.lr``).

.. function:: get_task(name: str, tasks_path: Optional[str] = None) -> SimpleNamespace

   Loads a task definition from ``configs/tasks.yaml`` by dataset name.
   Returns a ``SimpleNamespace`` with task metadata such as ``experiment_type``,
   ``task_columns``, and ``prompt``.

.. function:: prepare_config(cli_args: SimpleNamespace, task: SimpleNamespace) -> SimpleNamespace

   Merges YAML default hyperparameters with CLI argument overrides.
   Loads the appropriate YAML file based on ``task.experiment_type``
   (e.g., ``classification.yaml``, ``regression.yaml``).
   CLI values take precedence; the ``datasets`` key is skipped as it is routing information only.

----

chemberta4.callbacks
--------------------

Custom PyTorch Lightning callbacks for experiment tracking.

.. class:: WandbCallback(Callback)

   Logs training, validation, and test metrics to Weights & Biases.
   Only logs on rank 0 (``trainer.is_global_zero``) to avoid duplicates in DDP.

   .. method:: on_train_epoch_end(trainer, pl_module)

      Logs all metrics whose keys contain ``'train'`` at the end of each training epoch.

   .. method:: on_validation_epoch_end(trainer, pl_module)

      Logs all metrics whose keys contain ``'val'`` plus the current learning rate.

   .. method:: on_test_epoch_end(trainer, pl_module)

      Logs all metrics whose keys contain ``'test'``, prefixed with ``'final_'``.
