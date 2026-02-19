"""
Distributed training utilities following nanochat patterns.

Simple, explicit utilities for rank-aware operations.
"""

import os
import random
import numpy as np
import torch


def get_rank() -> int:
    """Get the current process rank in distributed training."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def print0(*args, **kwargs):
    """Print only on rank 0. Use this instead of print() in DDP code."""
    if is_main_process():
        print(*args, **kwargs)


def set_seed(seed: int = 42):
    """Set seed for reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_map(device: torch.device) -> dict:
    """
    Get device map for model loading compatible with DDP.

    Args:
        device: The torch device (from self.device in Lightning module)

    Returns:
        Device map dict for from_pretrained()
    """
    if device.type == "cuda":
        return {"": device.index if device.index is not None else 0}
    return {"": "cpu"}


def load_config(config_path: str):
    """Load YAML config file and return as SimpleNamespace."""
    import yaml
    from types import SimpleNamespace
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)


def get_task(name: str, tasks_path: str = None):
    """Load a task definition from configs/tasks.yaml as a SimpleNamespace."""
    import yaml
    from types import SimpleNamespace
    from pathlib import Path
    if tasks_path is None:
        tasks_path = str(Path(__file__).resolve().parent.parent / "configs" / "tasks.yaml")
    with open(tasks_path) as f:
        all_tasks = yaml.safe_load(f)
    if name not in all_tasks:
        raise ValueError(f"Unknown task '{name}'. Available: {list(all_tasks.keys())}")
    return SimpleNamespace(name=name, **all_tasks[name])


def format_params(num_params: int) -> str:
    """Format parameter count for display."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)
