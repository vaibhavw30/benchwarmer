import ast
import pathlib
import numpy as np
import pandas as pd
from backend_ml.signal import dataset as ds


class _FakeXGB:
    # returns P(home win) = 0.7 for every row via predict_proba[:,1]
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


class _FakeRidge:
    def decision_function(self, X):
        return np.zeros(len(X))          # sigmoid(0) = 0.5


class _FakeScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def _games(n=3):
    row = {f: 1.0 for f in ds.FEATURES}
    df = pd.DataFrame([row] * n)
    df["GAME_ID"] = [f"g{i}" for i in range(n)]
    df["GAME_DATE_H"] = pd.to_datetime("2026-01-01")
    df["TEAM_ID_H"] = 100
    df["TEAM_ID_A"] = 200
    df["HOME_WIN"] = [1, 0, 1][:n]
    return df


def test_ensemble_probs_matches_weighted_average():
    df = _games(2)
    p = ds.ensemble_probs(df[ds.FEATURES], _FakeXGB(), _FakeRidge(), _FakeScaler(),
                          xgb_weight=0.7, ridge_weight=0.3)
    # 0.7*0.7 + 0.3*0.5 = 0.64
    assert np.allclose(p, 0.64)


def test_build_recompute_dataset_schema_and_values():
    out = ds.build_recompute_dataset(
        _games(3), _FakeXGB(), _FakeRidge(), _FakeScaler(),
        weights={"xgb_weight": 0.7, "ridge_weight": 0.3})
    assert list(out.columns) == ["game_id", "date", "home_team_id",
                                 "away_team_id", "p_model", "outcome"]
    assert len(out) == 3
    assert np.allclose(out["p_model"], 0.64)
    assert out["outcome"].tolist() == [1, 0, 1]


def test_build_served_dataset_derives_outcome_from_scores():
    rows = [
        {"game_id": "g1", "date": "2026-01-01", "home_team_id": 1,
         "away_team_id": 2, "home_win_probability": 0.6,
         "home_score": 110, "away_score": 100},
        {"game_id": "g2", "date": "2026-01-01", "home_team_id": 3,
         "away_team_id": 4, "home_win_probability": 0.4,
         "home_score": 95, "away_score": 120},
    ]
    out = ds.build_served_dataset(rows)
    assert out["p_model"].tolist() == [0.6, 0.4]
    assert out["outcome"].tolist() == [1, 0]


def test_served_dataset_skips_unfinished_games():
    rows = [{"game_id": "g1", "date": "2026-01-01", "home_team_id": 1,
             "away_team_id": 2, "home_win_probability": 0.6,
             "home_score": None, "away_score": None}]
    assert ds.build_served_dataset(rows).empty


def test_no_leakage_predict_games_not_imported():
    # Guard: the recompute path must never pull in predict.predict_games
    src = pathlib.Path(ds.__file__).read_text()
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names += [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            names += [a.name for a in node.names]
    assert not any("predict_games" in n for n in names)
    assert "predict" not in [n.split(".")[-1] for n in names]
