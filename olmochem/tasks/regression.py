"""
MoleculeNet regression task configurations.

Regression tasks for predicting continuous molecular properties.
"""

from .base import TaskConfig, register_task


# ============================================================================
# ADMET Regression Tasks
# ============================================================================

register_task(TaskConfig(
    name="clearance",
    task_columns=["target"],
    prompt="Predict intrinsic hepatic clearance from SMILES",
    task_type="regression",
    target_column="target",
    normalize=True,
    monitor_metric="val/rmse",
    monitor_mode="min",
))


# ============================================================================
# Physicochemical Property Tasks
# ============================================================================

register_task(TaskConfig(
    name="esol",
    task_columns=["measured log solubility in mols per litre"],
    prompt="Predict aqueous solubility from SMILES",
    task_type="regression",
    target_column="measured log solubility in mols per litre",
    normalize=True,
    monitor_metric="val/rmse",
    monitor_mode="min",
))

register_task(TaskConfig(
    name="freesolv",
    task_columns=["expt"],
    prompt="Predict hydration free energy from SMILES",
    task_type="regression",
    target_column="expt",
    normalize=True,
    monitor_metric="val/rmse",
    monitor_mode="min",
))

register_task(TaskConfig(
    name="lipophilicity",
    task_columns=["exp"],
    prompt="Predict lipophilicity from SMILES",
    task_type="regression",
    target_column="exp",
    normalize=True,
    monitor_metric="val/rmse",
    monitor_mode="min",
))
