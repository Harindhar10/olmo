"""
Base task interface and registry for molecular property prediction.

This module provides a simple registry pattern for defining tasks.
Contributors can add new datasets by registering a TaskConfig.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class TaskConfig:
    """
    Configuration for a molecular property prediction task.

    Args:
        name: Unique task identifier (e.g., 'bbbp', 'hiv')
        task_columns: Column names in the CSV containing labels
        prompt: Natural language prompt for the task
        task_type: One of 'binary', 'multilabel', 'multitask', 'regression', 'generation'
        target_column: For regression, the specific column name (defaults to task_columns[0])
        normalize: Whether to normalize regression labels
        monitor_metric: Metric to monitor for early stopping/checkpointing
        monitor_mode: 'max' or 'min' for the monitored metric
        smiles_column: Column containing SMILES strings (default: 'smiles')
    """
    name: str
    task_columns: List[str]
    prompt: str
    task_type: Literal["binary", "multilabel", "multitask", "regression", "generation"]

    # Regression specific
    target_column: Optional[str] = None
    normalize: bool = True

    # Monitoring
    monitor_metric: str = "val/roc_auc"
    monitor_mode: str = "max"

    # Data columns
    smiles_column: str = "smiles"

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.target_column is None and self.task_type == "regression":
            self.target_column = self.task_columns[0]

        # Set appropriate defaults based on task type
        if self.task_type == "regression":
            if self.monitor_metric == "val/roc_auc":
                self.monitor_metric = "val/rmse"
                self.monitor_mode = "min"

    @property
    def num_tasks(self) -> int:
        """Number of tasks/labels."""
        return len(self.task_columns)


# Global registry
TASK_REGISTRY: dict[str, TaskConfig] = {}


def register_task(config: TaskConfig) -> TaskConfig:
    """
    Register a task configuration.

    Args:
        config: TaskConfig instance to register

    Returns:
        The same config (for chaining)
    """
    if config.name in TASK_REGISTRY:
        raise ValueError(f"Task '{config.name}' is already registered")
    TASK_REGISTRY[config.name] = config
    return config


def get_task(name: str) -> TaskConfig:
    """
    Get task configuration by name.

    Args:
        name: Task identifier

    Returns:
        TaskConfig for the requested task

    Raises:
        ValueError: If task is not found
    """
    if name not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{name}'. Available tasks: {available}")
    return TASK_REGISTRY[name]


def list_tasks() -> List[str]:
    """
    List all registered task names.

    Returns:
        List of task identifiers
    """
    return list(TASK_REGISTRY.keys())
