"""save_artifacts writes the 4 artifacts to output_dir; default '.' == cwd."""
import json
import os

import joblib

from train_model import save_artifacts, MODEL_PATH, RIDGE_MODEL_PATH, SCALER_PATH, ENSEMBLE_WEIGHTS_PATH
from train_model import build_weights_config

ARTIFACT_NAMES = [MODEL_PATH, RIDGE_MODEL_PATH, SCALER_PATH, ENSEMBLE_WEIGHTS_PATH]
WEIGHTS = {"xgb_weight": 0.7, "ridge_weight": 0.3, "test_accuracy": 0.66, "test_brier": 0.2, "train_date": "x"}


def test_save_artifacts_writes_all_four_to_output_dir(tmp_path):
    save_artifacts({"m": 1}, {"m": 2}, {"m": 3}, WEIGHTS, output_dir=str(tmp_path))
    for name in ARTIFACT_NAMES:
        assert (tmp_path / name).exists(), f"missing {name}"
    with open(tmp_path / ENSEMBLE_WEIGHTS_PATH) as f:
        assert json.load(f) == WEIGHTS
    assert joblib.load(tmp_path / MODEL_PATH) == {"m": 1}


def test_save_artifacts_default_writes_to_cwd(tmp_path, monkeypatch):
    # Regression guard: no-arg behavior must stay identical to today.
    monkeypatch.chdir(tmp_path)
    save_artifacts({"m": 1}, {"m": 2}, {"m": 3}, WEIGHTS)
    for name in ARTIFACT_NAMES:
        assert os.path.exists(name), f"{name} not written to cwd"


def test_build_weights_config_includes_numeric_test_brier():
    cfg = build_weights_config(0.6, 0.4, 0.66, 0.21, train_date="2026-07-17")
    assert cfg["xgb_weight"] == 0.6
    assert cfg["ridge_weight"] == 0.4
    assert cfg["test_accuracy"] == 0.66
    assert cfg["test_brier"] == 0.21
    assert isinstance(cfg["test_accuracy"], float)
    assert isinstance(cfg["test_brier"], float)
    assert cfg["train_date"] == "2026-07-17"


def test_build_weights_config_coerces_numpy_scalars_to_float():
    import numpy as np
    cfg = build_weights_config(0.5, 0.5, np.float64(0.66), np.float64(0.19), train_date="x")
    assert type(cfg["test_accuracy"]) is float
    assert type(cfg["test_brier"]) is float


def test_build_weights_config_defaults_train_date_to_now():
    cfg = build_weights_config(0.5, 0.5, 0.66, 0.2)
    # A non-empty ISO-ish timestamp string; exact value is time-dependent.
    assert isinstance(cfg["train_date"], str) and len(cfg["train_date"]) > 0
