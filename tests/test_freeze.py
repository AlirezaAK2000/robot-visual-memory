from pathlib import Path

import pytest

from rvm.evaluation import freeze_development, verify_frozen
from rvm.io import load_config


def test_freeze_sanitizes_paths_and_detects_setting_changes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.example.yaml")
    config["dataset"]["input_path"] = "/kaggle/input/private/data.tgz"
    path = tmp_path / "freeze.json"
    freeze_development(config, path, development_environment="hm3d-v0/dev")
    assert "/kaggle/input" not in path.read_text()
    assert verify_frozen(config, path)
    config["dataset"]["input_path"] = None  # local handoff has no original dataset location
    assert verify_frozen(config, path)
    config["retrieval"]["min_graph_score"] = 3.0
    with pytest.raises(ValueError, match="changed"):
        verify_frozen(config, path)
