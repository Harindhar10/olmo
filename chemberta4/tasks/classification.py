"""
MoleculeNet classification task configurations.

Binary, multilabel, and multitask classification tasks for molecular property prediction.
"""

from .base import TaskConfig, register_task


# ============================================================================
# Binary Classification Tasks
# ============================================================================

register_task(TaskConfig(
    name="bbbp",
    task_columns=["p_np"],
    prompt="Does this molecule permeate the blood-brain barrier?",
    task_type="binary",
    monitor_metric="val/roc_auc",
    monitor_mode="max",
))

register_task(TaskConfig(
    name="bace_classification",
    task_columns=["Class"],
    prompt="Is this molecule a BACE-1 inhibitor?",
    task_type="binary",
    monitor_metric="val/roc_auc",
    monitor_mode="max",
))

register_task(TaskConfig(
    name="hiv",
    task_columns=["HIV_active"],
    prompt="Does this molecule inhibit HIV replication?",
    task_type="binary",
    monitor_metric="val/roc_auc",
    monitor_mode="max",
))

register_task(TaskConfig(
    name="clintox",
    task_columns=["CT_TOX"],
    prompt="Is this molecule clinically toxic?",
    task_type="binary",
    monitor_metric="val/roc_auc",
    monitor_mode="max",
))


# ============================================================================
# Multilabel Classification Tasks
# ============================================================================

register_task(TaskConfig(
    name="sider",
    task_columns=[
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    prompt="What side effects does this drug cause?",
    task_type="multilabel",
    monitor_metric="val/roc_auc",
    monitor_mode="max",
))


# ============================================================================
# Multitask Classification Tasks
# ============================================================================

register_task(TaskConfig(
    name="tox21",
    task_columns=[
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
    prompt="Predict toxicity across multiple assays for this molecule.",
    task_type="multitask",
    monitor_metric="val/roc_auc",
    monitor_mode="max",
))
