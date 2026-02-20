"""
Distributed training utilities following nanochat patterns.

Simple, explicit utilities for rank-aware operations.
"""

import logging
import os
import random
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


def log0(msg: str) -> None:
    """Log a message only on rank 0 to avoid duplicate output in DDP.

    Parameters
    ----------
    msg : str
        Message to log via :func:`logging.Logger.info`.
    """
    if is_main_process():
        logger.info(msg)


def get_rank() -> int:
    """Return the current process rank in distributed training.

    Returns
    -------
    int
        Local rank of the current process (0 on single-GPU / CPU).
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check whether this process is the main process (rank 0).

    Returns
    -------
    bool
        'True' if rank is 0, 'False' otherwise.
    """
    return get_rank() == 0


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all random sources.

    Parameters
    ----------
    seed : int
        Seed value to use for Python, NumPy, and PyTorch RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_map(device: torch.device) -> Dict[str, Any]:
    """Return a device map for model loading compatible with DDP.

    Parameters
    ----------
    device : torch.device
        The torch device obtained from 'self.device' inside a Lightning module.

    Returns
    -------
    Dict[str, Any]
        Device map dict suitable for 'from_pretrained(device_map=...)'.
    """
    if device.type == "cuda":
        return {"": device.index if device.index is not None else 0}
    return {"": "cpu"}


def load_config(config_path: str) -> SimpleNamespace:
    """Load a YAML config file and return it as a :class:`SimpleNamespace`.

    Parameters
    ----------
    config_path : str
        Absolute or relative path to the YAML configuration file.

    Returns
    -------
    SimpleNamespace
        Config fields accessible as attributes.
    """
    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)


def get_task(name: str, tasks_path: Optional[str] = None) -> SimpleNamespace:
    """Load a task definition from 'configs/tasks.yaml'.

    Parameters
    ----------
    name : str
        Dataset / task name (must be a key in 'tasks.yaml').
    tasks_path : str, optional
        Path to 'tasks.yaml'. Defaults to 'configs/tasks.yaml' relative
        to the package root.

    Returns
    -------
    SimpleNamespace
        Task metadata (e.g. 'experiment_type', 'task_columns', 'prompt').
    """
    import yaml
    from pathlib import Path
    if tasks_path is None:
        tasks_path = str(Path(__file__).resolve().parent.parent / "configs" / "tasks.yaml")
    with open(tasks_path) as f:
        all_tasks = yaml.safe_load(f)
    if name not in all_tasks:
        raise ValueError(f"Unknown task '{name}'. Available: {list(all_tasks.keys())}")
    return SimpleNamespace(name=name, **all_tasks[name])


def prepare_config(cli_args: SimpleNamespace, task: SimpleNamespace) -> SimpleNamespace:
    """Merge YAML defaults with CLI overrides for a given task.

    Loads the type-appropriate YAML config (based on 'task.experiment_type'),
    then overrides any field where the CLI arg is non-None. The 'datasets'
    key is skipped as it is routing information, not a config field.

    Parameters
    ----------
    cli_args : SimpleNamespace
        Parsed CLI arguments from :func:`argparse.ArgumentParser.parse_args`.
    task : SimpleNamespace
        Task metadata returned by :func:`get_task`.

    Returns
    -------
    SimpleNamespace
        Merged configuration ready for use by experiment scripts.
    """
    from pathlib import Path

    config_dir = Path(__file__).resolve().parent.parent / "configs"
    yaml_map = {
        "classification": "classification.yaml",
        "regression": "regression.yaml",
        "pretraining": "pretrain.yaml",
        "instruction": "instruction.yaml",
    }

    yaml_file = config_dir / yaml_map[task.experiment_type]
    config = load_config(str(yaml_file))

    # Override with CLI args that were explicitly set (non-None)
    for key, value in vars(cli_args).items():
        if key == "datasets":
            continue
        if value is not None:
            setattr(config, key, value)

    return config


def format_params(num_params: int) -> str:
    """Format a raw parameter count into a human-readable string.

    Parameters
    ----------
    num_params : int
        Total number of parameters.

    Returns
    -------
    str
        Formatted string such as '7.00B', '350.00M', or '512K'.
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)
