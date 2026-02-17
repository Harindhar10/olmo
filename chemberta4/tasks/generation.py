"""
Generation task configurations for pretraining and instruction tuning.

These tasks use causal language modeling (next token prediction).
"""

from .base import TaskConfig, register_task


# ============================================================================
# Pretraining Tasks (Causal LM on SMILES)
# ============================================================================

register_task(TaskConfig(
    name="zinc20",
    task_columns=[],  # No labels for pretraining
    prompt="SMILES:",  # Prefix for SMILES strings
    task_type="generation",
    monitor_metric="train/loss",
    monitor_mode="min",
))

register_task(TaskConfig(
    name="pubchem",
    task_columns=[],
    prompt="SMILES:",
    task_type="generation",
    monitor_metric="train/loss",
    monitor_mode="min",
))


# ============================================================================
# Instruction Tuning Tasks
# ============================================================================

register_task(TaskConfig(
    name="uspto",
    task_columns=["instruction", "input", "output"],  # Instruction format columns
    prompt="",  # Prompt is built from instruction/input/output
    task_type="generation",
    monitor_metric="train/loss",
    monitor_mode="min",
))
