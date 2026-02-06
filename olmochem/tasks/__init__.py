"""
Task registry for molecular property prediction.

Adding a new task is simple:
    from olmochem.tasks import register_task, TaskConfig

    register_task(TaskConfig(
        name="my_dataset",
        task_columns=["activity"],
        prompt="Predict activity for this molecule",
        task_type="binary",
    ))
"""

from .base import TaskConfig, register_task, get_task, list_tasks, TASK_REGISTRY

# Import task modules to register tasks
from . import classification
from . import regression
from . import generation

__all__ = [
    "TaskConfig",
    "register_task",
    "get_task",
    "list_tasks",
    "TASK_REGISTRY",
]
