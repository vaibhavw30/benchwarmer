"""Layer 3 integration tests for the walk-forward half-life sweep.

Pure-function tests (make_folds, select_h_star) are deterministic. The
run_sweep tests use small synthetic frames: one with a planted regime shift
(recency should help) and one stationary (uniform should win).
"""
import numpy as np
import pandas as pd
import pytest

import halflife_sweep as hs
from signal_research.dataset import FEATURES

TARGET = "HOME_WIN"


def _base_frame(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["GAME_DATE_H"] = pd.date_range("2015-10-01", periods=n, freq="D")
    df["GAME_ID"] = np.arange(n)
    df["TEAM_ID_H"] = 1
    df["TEAM_ID_A"] = 2
    return df, rng


def _stationary_frame(n=1000, seed=3):
    df, rng = _base_frame(n, seed)
    signal = df["ELO_H"].values + 0.2 * rng.normal(size=n)
    df[TARGET] = (signal > 0).astype(int)
    return df


def _regime_shift_frame(n=1000, seed=3):
    # First half: home wins when ELO_H>0. Second half: mapping flips.
    df, rng = _base_frame(n, seed)
    driver = df["ELO_H"].values + 0.15 * rng.normal(size=n)
    half = n // 2
    label = np.empty(n, dtype=int)
    label[:half] = (driver[:half] > 0).astype(int)
    label[half:] = (driver[half:] < 0).astype(int)
    df[TARGET] = label
    return df


# ---- pure-function tests -------------------------------------------------

def test_make_folds_expanding_window_has_no_leakage():
    folds = hs.make_folds(n_rows=1000, n_folds=4, eval_games=400)
    assert len(folds) == 4
    prev_end = 1000 - 400
    for train_end, fold_start, fold_end in folds:
        assert train_end == fold_start          # training is strictly [0, fold_start)
        assert fold_start == prev_end            # folds are contiguous
        assert fold_end > fold_start
        prev_end = fold_end
    assert folds[-1][2] == 1000                  # last fold reaches the end


def test_make_folds_all_training_indices_precede_fold():
    folds = hs.make_folds(n_rows=1000, n_folds=4, eval_games=400)
    for train_end, fold_start, fold_end in folds:
        # every training index < every fold index
        assert train_end <= fold_start


def test_select_h_star_picks_finite_when_it_beats_margin():
    per_h = {
        "None": {"mean_brier": 0.250, "mean_accuracy": 0.60, "folds": []},
        "800": {"mean_brier": 0.240, "mean_accuracy": 0.62, "folds": []},
        "1500": {"mean_brier": 0.246, "mean_accuracy": 0.61, "folds": []},
    }
    assert hs.select_h_star(per_h, margin=0.002) == 800


def test_select_h_star_returns_none_inside_margin():
    per_h = {
        "None": {"mean_brier": 0.250, "mean_accuracy": 0.60, "folds": []},
        "800": {"mean_brier": 0.2495, "mean_accuracy": 0.61, "folds": []},  # only 0.0005 better
    }
    assert hs.select_h_star(per_h, margin=0.002) is None


def test_select_h_star_returns_none_when_uniform_best():
    per_h = {
        "None": {"mean_brier": 0.240, "mean_accuracy": 0.62, "folds": []},
        "800": {"mean_brier": 0.250, "mean_accuracy": 0.60, "folds": []},
    }
    assert hs.select_h_star(per_h, margin=0.002) is None


# ---- integration tests over run_sweep ------------------------------------

FROZEN = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1}


def test_infinity_always_in_grid():
    df = _stationary_frame()
    res = hs.run_sweep(df, h_grid=[400, None], n_folds=3, eval_games=300,
                       frozen_params=FROZEN)
    assert "None" in res["per_h"]                # uniform baseline always scored
    for stats in res["per_h"].values():
        assert np.isfinite(stats["mean_brier"])
        assert len(stats["folds"]) == 3


def test_planted_regime_shift_favors_finite_H():
    df = _regime_shift_frame(n=1000, seed=3)
    # eval region sits deep in the flipped regime, so training contains both
    # regimes and recency can up-weight the recent (correct) one.
    res = hs.run_sweep(df, h_grid=[150, None], n_folds=3, eval_games=300,
                       frozen_params=FROZEN)
    finite_brier = res["per_h"]["150"]["mean_brier"]
    uniform_brier = res["per_h"]["None"]["mean_brier"]
    assert finite_brier < uniform_brier          # recency genuinely helps here


def test_stable_data_favors_uniform():
    df = _stationary_frame(n=1000, seed=5)
    res = hs.run_sweep(df, h_grid=[150, 400, None], n_folds=3, eval_games=300,
                       frozen_params=FROZEN)
    # On a stationary mapping no finite H should clear the acceptance margin.
    assert res["h_star"] is None


def test_report_written_with_required_fields(tmp_path):
    df = _stationary_frame()
    res = hs.run_sweep(df, h_grid=[400, None], n_folds=3, eval_games=300,
                       frozen_params=FROZEN)
    paths = hs.write_report(res, out_dir=str(tmp_path))
    import json
    with open(paths["json"]) as f:
        payload = json.load(f)
    assert "per_h" in payload and "None" in payload["per_h"]
    assert "frozen_params" in payload
    assert "h_star" in payload
    assert (tmp_path / "halflife_sweep.md").exists()
    # per-fold breakdown present
    assert isinstance(payload["per_h"]["None"]["folds"], list)


def test_pick_frozen_params_returns_hyperparam_dict():
    df = _stationary_frame(n=300, seed=1)
    params = hs.pick_frozen_params(df[FEATURES], df[TARGET])
    assert isinstance(params, dict)
    for key in ("n_estimators", "max_depth", "learning_rate"):
        assert key in params
