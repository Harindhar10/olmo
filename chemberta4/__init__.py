"""
chemberta4 - A minimal library for molecular property prediction with OLMo

Following the nanochat philosophy: explicit over implicit, minimal abstraction.
"""

from .utils import is_main_process, get_rank, set_seed, get_task, log0
from .model import ClassificationHead, CausalLMClassificationHead, RegressionHead
from .data import MoleculeNetDataset, PretrainingDataset, InstructionDataset
from .trainer import OLMoClassifier, OLMoRegressor, OLMoPretrainer
from .callbacks import WandbCallback