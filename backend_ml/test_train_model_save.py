"""save_artifacts writes the 4 artifacts to output_dir; default '.' == cwd."""
import json
import os

import joblib

from train_model import save_artifacts, MODEL_PATH, RIDGE_MODEL_PATH, SCALER_PATH, ENSEMBLE_WEIGHTS_PATH

ARTIFACT_NAMES = [MODEL_PATH, RIDGE_MODEL_PATH, SCALER_PATH, ENSEMBLE_WEIGHTS_PATH]
WEIGHTS = {"xgb_weight": 0.7, "ridge_weight": 0.3, "test_accuracy": 0.66, "train_date": "x"}


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
