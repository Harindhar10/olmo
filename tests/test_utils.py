"""Unit tests for chemberta4.utils."""

import os
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chemberta4.utils import (
    get_device_map,
    get_rank,
    get_task,
    is_main_process,
    load_config,
    prepare_config,
    set_seed,
)

# ---------------------------------------------------------------------------
# Path to the real configs directory
# ---------------------------------------------------------------------------
_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
_TASKS_YAML = os.path.join(_CONFIGS_DIR, "tasks.yaml")


# ---------------------------------------------------------------------------
# get_rank
# ---------------------------------------------------------------------------

class TestGetRank:
    def test_default_is_zero(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert get_rank() == 0

    def test_reads_local_rank_env(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert get_rank() == 2


# ---------------------------------------------------------------------------
# is_main_process
# ---------------------------------------------------------------------------

class TestIsMainProcess:
    def test_true_when_rank_zero(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert is_main_process() is True

    def test_true_when_local_rank_zero(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert is_main_process() is True

    def test_false_when_rank_nonzero(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert is_main_process() is False


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

class TestSetSeed:
    def test_determinism(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_different_seeds_differ(self):
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b

    def test_numpy_determinism(self):
        set_seed(99)
        a = np.random.rand()
        set_seed(99)
        b = np.random.rand()
        assert a == b

    def test_torch_determinism(self):
        set_seed(7)
        a = torch.rand(1).item()
        set_seed(7)
        b = torch.rand(1).item()
        assert a == b


# ---------------------------------------------------------------------------
# get_device_map
# ---------------------------------------------------------------------------

class TestGetDeviceMap:
    def test_cpu(self):
        result = get_device_map(torch.device("cpu"))
        assert result == {"": "cpu"}

    def test_cuda_with_index(self):
        result = get_device_map(torch.device("cuda:0"))
        assert result == {"": 0}

    def test_cuda_without_index(self):
        # torch.device("cuda") has index None; should default to 0
        result = get_device_map(torch.device("cuda"))
        assert result == {"": 0}

    def test_cuda_index_1(self):
        result = get_device_map(torch.device("cuda:1"))
        assert result == {"": 1}


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_yaml_as_namespace(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("lr: 0.001\nbatch_size: 4\nmodel: olmo\n")
        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, SimpleNamespace)
        assert cfg.lr == 0.001
        assert cfg.batch_size == 4
        assert cfg.model == "olmo"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------

class TestGetTask:
    def test_known_task_bbbp(self):
        task = get_task("bbbp", tasks_path=_TASKS_YAML)
        assert isinstance(task, SimpleNamespace)
        assert task.name == "bbbp"
        assert hasattr(task, "experiment_type")

    def test_known_task_delaney(self):
        task = get_task("delaney", tasks_path=_TASKS_YAML)
        assert task.name == "delaney"
        assert task.experiment_type == "regression"

    def test_unknown_task_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task_xyz", tasks_path=_TASKS_YAML)


# ---------------------------------------------------------------------------
# prepare_config
# ---------------------------------------------------------------------------

class TestPrepareConfig:
    def _make_cli_args(self, **kwargs):
        """Build a minimal SimpleNamespace representing parsed CLI args."""
        defaults = {
            "datasets": ["bbbp"],
            "lr": None,
            "batch_size": None,
            "epochs": None,
            "finetune_strategy": None,
            "wandb": None,
            "wandb_project": None,
            "output_dir": None,
            "seed": None,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_defaults_not_overridden_by_none(self):
        task = get_task("bbbp", tasks_path=_TASKS_YAML)
        cli = self._make_cli_args()  # all None
        cfg = prepare_config(cli, task)
        # The YAML default lr for classification is 2e-4
        assert cfg.lr == 2e-4

    def test_cli_overrides_yaml_default(self):
        task = get_task("bbbp", tasks_path=_TASKS_YAML)
        cli = self._make_cli_args(lr=1e-3)
        cfg = prepare_config(cli, task)
        assert cfg.lr == 1e-3

    def test_datasets_key_not_written_to_config(self):
        task = get_task("bbbp", tasks_path=_TASKS_YAML)
        cli = self._make_cli_args(datasets=["bbbp", "hiv"])
        cfg = prepare_config(cli, task)
        assert not hasattr(cfg, "datasets")

    def test_regression_task_loads_regression_yaml(self):
        task = get_task("delaney", tasks_path=_TASKS_YAML)
        cli = self._make_cli_args()
        cfg = prepare_config(cli, task)
        # regression.yaml has epochs=30, classification.yaml has 15
        assert cfg.epochs == 30
