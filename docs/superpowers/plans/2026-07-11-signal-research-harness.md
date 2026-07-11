# Signal-Research Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline Python harness that measures the NBA ensemble's probability calibration, fits a recalibration map that feeds back into the fair values the C++ Kalshi engine trades on, and captures market prices forward to compute Closing Line Value (CLV) after fees.

**Architecture:** New `backend_ml/signal_research/` package of seven deep modules unified around one labeled dataset schema `(game_id, date, home_team_id, away_team_id, p_model, [market_price], outcome)`. Pure metric/transform modules are fully unit-tested on synthetic data; I/O and API modules are tested with mocks. Recalibration wires into `publish_fair_values.py` behind a `RECALIBRATE=1` flag so raw vs. recalibrated is A/B-comparable.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn (IsotonicRegression / LogisticRegression), pytest. No new heavyweight deps. Reuses `backend_ml/data_engine.py`, `ensemble_weights.json`, the gitignored `.pkl` models, and `trading_engine/config/engine.json`.

## Global Constraints

- **Package location:** all new code under `backend_ml/signal_research/`; tests under `backend_ml/signal_research/tests/`.
- **No leakage:** the historical recompute path uses ONLY `data_engine.build_training_dataset()`'s as-of features. `backend_ml/signal_research/dataset.py` must NOT import or call `predict.predict_games` (it uses today's stats → future leak).
- **Ensemble math must match `backtest.py`:** `p_model = xgb_weight * p_xgb + ridge_weight * p_ridge`; weights from `ensemble_weights.json` (current: `xgb_weight=0.7`, `ridge_weight=0.3`), fallback `0.5/0.5`. Ridge probability is `1/(1+exp(-decision_function))`.
- **Feature order (18, verbatim from `backtest.py`):** `ELO_H, ELO_A, REB_MISMATCH, TOV_MISMATCH, SHOOTING_GAP, EFG_PCT_EWMA_H, TOV_PCT_EWMA_H, ORB_PCT_EWMA_H, FT_RATE_EWMA_H, FATIGUE_SCORE_H, MOMENTUM_H, HOME_ALTITUDE, EFG_PCT_EWMA_A, TOV_PCT_EWMA_A, ORB_PCT_EWMA_A, FT_RATE_EWMA_A, FATIGUE_SCORE_A, MOMENTUM_A`.
- **Fee is single-sourced:** read `fee_cents_per_contract` (currently `1`) from `trading_engine/config/engine.json`; never hard-code a second copy.
- **Recalibration reported out-of-sample only:** train/test split; never report an in-sample Brier drop as the headline.
- **CLV is forward-accruing:** zero on day one; refuse to summarize below a minimum captured-snapshot count.
- **Secrets in env only:** `ODDS_API_KEY`, any Kalshi read key. `backend_ml/signal_research/artifacts/` and the market-snapshots file are gitignored.
- **Run tests from the repo root** with `python -m pytest`. Model `.pkl` files and `.csv` caches are gitignored and MUST NOT appear in any test.

---

### Task 1: Calibration metrics (pure TDD core)

**Files:**
- Create: `backend_ml/signal_research/__init__.py`
- Create: `backend_ml/signal_research/calibration.py`
- Create: `backend_ml/signal_research/tests/__init__.py`
- Test: `backend_ml/signal_research/tests/test_calibration.py`

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces:
  - `brier_score(p: np.ndarray, y: np.ndarray) -> float`
  - `log_loss(p: np.ndarray, y: np.ndarray) -> float`
  - `reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame` with columns `bin_lo, bin_hi, count, mean_pred, mean_obs`
  - `expected_calibration_error(p, y, n_bins: int = 10) -> float`
  - `calibration_report(p, y, n_bins: int = 10) -> dict` with keys `brier, log_loss, ece, n, reliability` (reliability = list of row dicts)

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_calibration.py
import numpy as np
import pandas as pd
from backend_ml.signal_research import calibration as cal


def test_brier_half_p_half_win():
    p = np.full(100, 0.5)
    y = np.array([1] * 50 + [0] * 50)
    assert cal.brier_score(p, y) == 0.25


def test_brier_perfect():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1, 0, 1, 0])
    assert cal.brier_score(p, y) == 0.0


def test_log_loss_is_clipped_and_finite():
    p = np.array([0.0, 1.0])          # extreme, would be inf unclipped
    y = np.array([1, 0])              # both "wrong"
    assert np.isfinite(cal.log_loss(p, y))


def test_reliability_table_bins_and_counts():
    p = np.array([0.05, 0.15, 0.95, 0.95])
    y = np.array([0, 0, 1, 1])
    tbl = cal.reliability_table(p, y, n_bins=10)
    assert list(tbl.columns) == ["bin_lo", "bin_hi", "count", "mean_pred", "mean_obs"]
    # two points land in the top bin, both win -> mean_obs == 1.0
    top = tbl[tbl["bin_lo"] == 0.9].iloc[0]
    assert top["count"] == 2
    assert top["mean_obs"] == 1.0


def test_ece_zero_for_perfectly_calibrated():
    # 1000 points where observed rate matches predicted per bin
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert cal.expected_calibration_error(p, y, n_bins=10) < 0.03


def test_ece_large_for_overconfident():
    p = np.full(1000, 0.9)
    y = np.array([1] * 500 + [0] * 500)   # only 50% actually win
    assert cal.expected_calibration_error(p, y, n_bins=10) > 0.35


def test_calibration_report_keys():
    p = np.array([0.5, 0.5])
    y = np.array([1, 0])
    rep = cal.calibration_report(p, y)
    assert set(rep) == {"brier", "log_loss", "ece", "n", "reliability"}
    assert rep["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research'`

- [ ] **Step 3: Write minimal implementation**

Create empty `backend_ml/signal_research/__init__.py` and `backend_ml/signal_research/tests/__init__.py`, then:

```python
# backend_ml/signal_research/calibration.py
"""Pure calibration metrics on (predicted probability, binary outcome).

No I/O, no model loading. Every function takes array-likes and returns a
number or a tidy DataFrame/dict. This is the harness's TDD core.
"""
import numpy as np
import pandas as pd

_EPS = 1e-15


def _arrays(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"p and y shape mismatch: {p.shape} vs {y.shape}")
    return p, y


def brier_score(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p, y = _arrays(p, y)
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    p, y = _arrays(p, y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False; clip so p==1.0 lands in the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        rows.append({
            "bin_lo": round(float(edges[b]), 4),
            "bin_hi": round(float(edges[b + 1]), 4),
            "count": count,
            "mean_pred": float(p[mask].mean()) if count else float("nan"),
            "mean_obs": float(y[mask].mean()) if count else float("nan"),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    p, y = _arrays(p, y)
    tbl = reliability_table(p, y, n_bins)
    tbl = tbl[tbl["count"] > 0]
    weights = tbl["count"] / len(p)
    gaps = (tbl["mean_pred"] - tbl["mean_obs"]).abs()
    return float((weights * gaps).sum())


def calibration_report(p, y, n_bins: int = 10) -> dict:
    p, y = _arrays(p, y)
    tbl = reliability_table(p, y, n_bins)
    return {
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "ece": expected_calibration_error(p, y, n_bins),
        "n": int(p.size),
        "reliability": tbl.to_dict(orient="records"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_calibration.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/__init__.py backend_ml/signal_research/calibration.py backend_ml/signal_research/tests/
git commit -m "feat(signal): pure calibration metrics (brier, log-loss, reliability, ece)"
```

---

### Task 2: Recalibration map (fit + out-of-sample eval + JSON artifact)

**Files:**
- Create: `backend_ml/signal_research/recalibration.py`
- Test: `backend_ml/signal_research/tests/test_recalibration.py`

**Interfaces:**
- Consumes: `calibration.brier_score`.
- Produces:
  - `class Recalibrator` with `.method: str`, `.transform(p: np.ndarray) -> np.ndarray`, `.to_dict() -> dict`, classmethod `.from_dict(d) -> Recalibrator`, `.save(path)`, classmethod `.load(path) -> Recalibrator`.
  - `fit_recalibrator(p, y, method: str = "isotonic") -> Recalibrator`
  - `evaluate_recalibration(p, y, method="isotonic", test_size=0.3, seed=0) -> dict` with keys `method, n_train, n_test, brier_raw, brier_recal, brier_delta` (all computed on the TEST split).

Isotonic is serialized as JSON-native `{"method":"isotonic","x":[...],"y":[...]}` and applied with `np.interp` (pure at load time — no sklearn needed to transform). Platt is `{"method":"platt","a":..,"b":..}`, applied as `sigmoid(a*logit(p)+b)`.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_recalibration.py
import numpy as np
import pytest
from backend_ml.signal_research import recalibration as rc
from backend_ml.signal_research import calibration as cal


def _miscalibrated(n=4000, seed=1):
    # true outcome prob is p_raw**2 -> model is systematically overconfident
    rng = np.random.default_rng(seed)
    p_raw = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p_raw ** 2).astype(int)
    return p_raw, y


def test_isotonic_reduces_out_of_sample_brier():
    p, y = _miscalibrated()
    res = rc.evaluate_recalibration(p, y, method="isotonic", test_size=0.3, seed=0)
    assert res["brier_recal"] < res["brier_raw"]
    assert res["brier_delta"] < 0            # delta = recal - raw, improvement is negative
    assert res["n_test"] > 0


def test_transform_is_monotonic_and_bounded():
    p, y = _miscalibrated()
    r = rc.fit_recalibrator(p, y, method="isotonic")
    grid = np.linspace(0, 1, 50)
    out = r.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)     # non-decreasing
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_save_load_roundtrip(tmp_path):
    p, y = _miscalibrated()
    r = rc.fit_recalibrator(p, y, method="isotonic")
    path = tmp_path / "recal.json"
    r.save(path)
    r2 = rc.Recalibrator.load(path)
    grid = np.linspace(0, 1, 20)
    assert np.allclose(r.transform(grid), r2.transform(grid))


def test_platt_also_reduces_brier():
    p, y = _miscalibrated()
    res = rc.evaluate_recalibration(p, y, method="platt", test_size=0.3, seed=0)
    assert res["brier_recal"] <= res["brier_raw"] + 1e-6


def test_unknown_method_raises():
    p, y = _miscalibrated()
    with pytest.raises(ValueError):
        rc.fit_recalibrator(p, y, method="banana")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_recalibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research.recalibration'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend_ml/signal_research/recalibration.py
"""Fit a monotone recalibration map and evaluate it out-of-sample.

Isotonic (default) is stored JSON-native as knot points and applied with
np.interp, so loading and transforming need no sklearn. Platt stores two
scalars. Improvement is always measured on a held-out test split.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from backend_ml.signal_research.calibration import brier_score

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Recalibrator:
    def __init__(self, method, params):
        self.method = method
        self._params = params

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            x = np.asarray(self._params["x"], dtype=float)
            y = np.asarray(self._params["y"], dtype=float)
            return np.clip(np.interp(p, x, y), 0.0, 1.0)
        if self.method == "platt":
            a, b = self._params["a"], self._params["b"]
            return _sigmoid(a * _logit(p) + b)
        raise ValueError(f"unknown method {self.method!r}")

    def to_dict(self):
        return {"method": self.method, **self._params}

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        method = d.pop("method")
        return cls(method, d)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_recalibrator(p, y, method: str = "isotonic") -> Recalibrator:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        # knot points sufficient to reproduce the step function via interp
        xs = np.unique(np.clip(p, 0, 1))
        ys = iso.predict(xs)
        return Recalibrator("isotonic", {"x": xs.tolist(), "y": ys.tolist()})
    if method == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(_logit(p).reshape(-1, 1), y)
        return Recalibrator("platt", {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])})
    raise ValueError(f"unknown method {method!r}")


def evaluate_recalibration(p, y, method="isotonic", test_size=0.3, seed=0) -> dict:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p_tr, p_te, y_tr, y_te = train_test_split(p, y, test_size=test_size, random_state=seed)
    r = fit_recalibrator(p_tr, y_tr, method=method)
    brier_raw = brier_score(p_te, y_te)
    brier_recal = brier_score(r.transform(p_te), y_te)
    return {
        "method": method,
        "n_train": int(p_tr.size),
        "n_test": int(p_te.size),
        "brier_raw": brier_raw,
        "brier_recal": brier_recal,
        "brier_delta": brier_recal - brier_raw,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_recalibration.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/recalibration.py backend_ml/signal_research/tests/test_recalibration.py
git commit -m "feat(signal): isotonic/platt recalibration with out-of-sample eval + JSON artifact"
```

---

### Task 3: Labeled dataset builder (recompute + served, leakage-guarded)

**Files:**
- Create: `backend_ml/signal_research/dataset.py`
- Test: `backend_ml/signal_research/tests/test_dataset.py`

**Interfaces:**
- Consumes: `data_engine.build_training_dataset` (real, at runtime), joblib models (real, at runtime). Tests inject fakes.
- Produces:
  - `FEATURES: list[str]` (the 18 columns, Global Constraints order).
  - `ensemble_probs(features_df, xgb_model, ridge_model, scaler, xgb_weight, ridge_weight) -> np.ndarray` — pure given models; matches `backtest.py`.
  - `build_recompute_dataset(games_df, xgb_model, ridge_model, scaler, weights: dict) -> pd.DataFrame` with columns `game_id, date, home_team_id, away_team_id, p_model, outcome`.
  - `build_served_dataset(rows: list[dict]) -> pd.DataFrame` — same schema, from Supabase `game_predictions ⋈ games` records (each row has `game_id, date, home_team_id, away_team_id, home_win_probability, home_score, away_score`).

`build_recompute_dataset` takes already-loaded models/frames (not paths) so tests need no `.pkl`. A thin `load_*` convenience wrapper that reads real artifacts lives in the CLI (Task 6), not here.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_dataset.py
import ast
import pathlib
import numpy as np
import pandas as pd
from backend_ml.signal_research import dataset as ds


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research.dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend_ml/signal_research/dataset.py
"""Assemble the labeled evaluation dataset: (p_model, outcome) per game.

Two builders share one output schema:
  game_id, date, home_team_id, away_team_id, p_model, outcome

- build_recompute_dataset: as-of features (from data_engine.build_training_dataset)
  scored by the ensemble. Models/frames are passed in, so tests need no .pkl.
- build_served_dataset: probabilities the model actually emitted (Supabase),
  outcome derived from final scores.

LEAKAGE GUARD: this module must never import predict.predict_games, which
builds features from *today's* stats and would leak the future into the past.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "ELO_H", "ELO_A",
    "REB_MISMATCH", "TOV_MISMATCH", "SHOOTING_GAP",
    "EFG_PCT_EWMA_H", "TOV_PCT_EWMA_H", "ORB_PCT_EWMA_H", "FT_RATE_EWMA_H",
    "FATIGUE_SCORE_H", "MOMENTUM_H", "HOME_ALTITUDE",
    "EFG_PCT_EWMA_A", "TOV_PCT_EWMA_A", "ORB_PCT_EWMA_A", "FT_RATE_EWMA_A",
    "FATIGUE_SCORE_A", "MOMENTUM_A",
]

_OUT_COLS = ["game_id", "date", "home_team_id", "away_team_id", "p_model", "outcome"]


def ensemble_probs(features_df, xgb_model, ridge_model, scaler,
                   xgb_weight, ridge_weight) -> np.ndarray:
    """P(home win) per row, matching backtest.py exactly."""
    X = features_df[FEATURES]
    xgb_p = xgb_model.predict_proba(X)[:, 1]
    ridge_decision = ridge_model.decision_function(scaler.transform(X))
    ridge_p = 1.0 / (1.0 + np.exp(-ridge_decision))
    return xgb_weight * xgb_p + ridge_weight * ridge_p


def build_recompute_dataset(games_df, xgb_model, ridge_model, scaler,
                            weights: dict) -> pd.DataFrame:
    xgb_w = weights.get("xgb_weight", 0.5)
    ridge_w = weights.get("ridge_weight", 0.5)
    p = ensemble_probs(games_df, xgb_model, ridge_model, scaler, xgb_w, ridge_w)
    return pd.DataFrame({
        "game_id": games_df["GAME_ID"].values,
        "date": pd.to_datetime(games_df["GAME_DATE_H"]).values,
        "home_team_id": games_df["TEAM_ID_H"].values,
        "away_team_id": games_df["TEAM_ID_A"].values,
        "p_model": p,
        "outcome": games_df["HOME_WIN"].astype(int).values,
    })[_OUT_COLS]


def build_served_dataset(rows) -> pd.DataFrame:
    out = []
    for r in rows:
        hs, as_ = r.get("home_score"), r.get("away_score")
        if hs is None or as_ is None:
            continue                      # unfinished game: no label
        out.append({
            "game_id": r["game_id"],
            "date": pd.to_datetime(r["date"]),
            "home_team_id": r["home_team_id"],
            "away_team_id": r["away_team_id"],
            "p_model": float(r["home_win_probability"]),
            "outcome": int(hs > as_),
        })
    return pd.DataFrame(out, columns=_OUT_COLS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_dataset.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/dataset.py backend_ml/signal_research/tests/test_dataset.py
git commit -m "feat(signal): labeled dataset builders (recompute + served) with leakage guard"
```

---

### Task 4: De-vig + market snapshot capture (forward infra)

**Files:**
- Create: `backend_ml/signal_research/market_capture.py`
- Test: `backend_ml/signal_research/tests/test_market_capture.py`

**Interfaces:**
- Consumes: nothing pure; `fetch_two_way_odds` and `fetch_kalshi_price` are injected in tests, real at runtime.
- Produces:
  - `american_to_prob(odds: float) -> float`
  - `devig(home_american: float, away_american: float) -> tuple[float, float]` — returns `(p_home, p_away)` summing to 1.
  - `build_snapshot_rows(watchlist, moment, kalshi_prices: dict, book_odds: dict) -> list[dict]` — pure row assembly; each row `{ticker, game_id, moment, kalshi_p, book_p, asof}`.
  - `append_snapshots(rows, path)` — append JSONL.
  - `capture(watchlist, moment, asof, fetch_kalshi_price, fetch_two_way_odds, path)` — orchestrator wiring the fetchers into `build_snapshot_rows` + `append_snapshots`.

`asof` is passed in (never call `datetime.now()` inside pure/testable functions). Kalshi price is cents [1,99]; `kalshi_p = cents/100`.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_market_capture.py
import json
import numpy as np
from backend_ml.signal_research import market_capture as mc


def test_american_to_prob_even_and_favorite():
    assert abs(mc.american_to_prob(100) - 0.5) < 1e-9
    assert abs(mc.american_to_prob(-110) - (110 / 210)) < 1e-9
    assert abs(mc.american_to_prob(150) - (100 / 250)) < 1e-9


def test_devig_even_market_is_half():
    ph, pa = mc.devig(-110, -110)
    assert abs(ph - 0.5) < 1e-9 and abs(pa - 0.5) < 1e-9


def test_devig_sums_to_one_and_orders_favorite():
    ph, pa = mc.devig(-200, +170)     # home favorite
    assert abs((ph + pa) - 1.0) < 1e-9
    assert ph > pa


def test_build_snapshot_rows_joins_and_shapes():
    watchlist = [{"ticker": "KXNBA-LAL-BOS", "game_id": "g1",
                  "home_team_id": 1, "away_team_id": 2, "game_date": "2026-01-01"}]
    rows = mc.build_snapshot_rows(
        watchlist, moment="tipoff",
        kalshi_prices={"KXNBA-LAL-BOS": 55},          # cents
        book_odds={"KXNBA-LAL-BOS": (-110, -110)},
        asof="2026-01-01T00:00:00Z")
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "KXNBA-LAL-BOS"
    assert r["game_id"] == "g1"
    assert r["moment"] == "tipoff"
    assert abs(r["kalshi_p"] - 0.55) < 1e-9
    assert abs(r["book_p"] - 0.5) < 1e-9
    assert r["asof"] == "2026-01-01T00:00:00Z"


def test_build_snapshot_rows_skips_missing_market_data():
    watchlist = [{"ticker": "T1", "game_id": "g1", "home_team_id": 1,
                  "away_team_id": 2, "game_date": "2026-01-01"}]
    # no kalshi price for T1 -> row skipped (fail-closed)
    rows = mc.build_snapshot_rows(watchlist, "tipoff", {}, {}, "2026-01-01T00:00:00Z")
    assert rows == []


def test_append_snapshots_writes_jsonl(tmp_path):
    path = tmp_path / "snaps.jsonl"
    mc.append_snapshots([{"ticker": "T1", "kalshi_p": 0.5}], path)
    mc.append_snapshots([{"ticker": "T2", "kalshi_p": 0.6}], path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["ticker"] == "T2"


def test_capture_orchestrates_fetchers(tmp_path):
    watchlist = [{"ticker": "T1", "game_id": "g1", "home_team_id": 1,
                  "away_team_id": 2, "game_date": "2026-01-01"}]
    path = tmp_path / "snaps.jsonl"
    n = mc.capture(
        watchlist, moment="t-60", asof="2026-01-01T00:00:00Z",
        fetch_kalshi_price=lambda ticker: 60,
        fetch_two_way_odds=lambda w: (-150, +130),
        path=path)
    assert n == 1
    row = json.loads(path.read_text().strip())
    assert abs(row["kalshi_p"] - 0.60) < 1e-9
    assert 0 < row["book_p"] < 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_market_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research.market_capture'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend_ml/signal_research/market_capture.py
"""Forward market-price capture for CLV.

Pure pieces (american_to_prob, devig, build_snapshot_rows) are fully tested.
The live fetchers (Kalshi REST GET, sportsbook odds) are injected so tests
never touch the network; the real fetchers are wired in the CLI (Task 6) and
live-verified by the user.

Snapshots are appended as JSONL. Path is gitignored.
"""
import json
from pathlib import Path


def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def devig(home_american: float, away_american: float):
    ph = american_to_prob(home_american)
    pa = american_to_prob(away_american)
    total = ph + pa
    return ph / total, pa / total


def build_snapshot_rows(watchlist, moment, kalshi_prices, book_odds, asof):
    rows = []
    for w in watchlist:
        ticker = w["ticker"]
        cents = kalshi_prices.get(ticker)
        odds = book_odds.get(ticker)
        if cents is None or odds is None:
            continue                      # fail-closed: incomplete market data
        book_p, _ = devig(odds[0], odds[1])
        rows.append({
            "ticker": ticker,
            "game_id": w["game_id"],
            "moment": moment,
            "kalshi_p": float(cents) / 100.0,
            "book_p": book_p,
            "asof": asof,
        })
    return rows


def append_snapshots(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def capture(watchlist, moment, asof, fetch_kalshi_price, fetch_two_way_odds, path):
    kalshi_prices, book_odds = {}, {}
    for w in watchlist:
        ticker = w["ticker"]
        try:
            kalshi_prices[ticker] = fetch_kalshi_price(ticker)
            book_odds[ticker] = fetch_two_way_odds(w)
        except Exception:
            continue                      # skip this ticker, keep the rest
    rows = build_snapshot_rows(watchlist, moment, kalshi_prices, book_odds, asof)
    append_snapshots(rows, path)
    return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_market_capture.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/market_capture.py backend_ml/signal_research/tests/test_market_capture.py
git commit -m "feat(signal): de-vig + forward market snapshot capture (fetchers injected)"
```

---

### Task 5: CLV + edge-vs-close (pure, fee-aware)

**Files:**
- Create: `backend_ml/signal_research/clv.py`
- Test: `backend_ml/signal_research/tests/test_clv.py`

**Interfaces:**
- Consumes: nothing pure.
- Produces:
  - `clv_cents(entry_price_cents, closing_price_cents, side: str) -> float` — `side` in `{"YES","NO"}`; YES = `closing - entry`, NO = `entry - closing`.
  - `edge_vs_close_cents(p_model, market_p, fee_cents) -> float` — `abs(p_model - market_p)*100 - fee_cents`.
  - `clv_report(snapshots_by_game, fee_cents, min_samples: int) -> dict` — pairs `t-60` (entry) and `tipoff` (closing) snapshots per game; returns `{"n": int, "insufficient": bool, "mean_clv_cents": float|None, "mean_edge_cents": float|None}`. Below `min_samples`, `insufficient=True` and the means are `None`.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_clv.py
from backend_ml.signal_research import clv


def test_clv_yes_positive_when_price_rises():
    assert clv.clv_cents(40, 46, "YES") == 6
    assert clv.clv_cents(40, 46, "NO") == -6


def test_edge_after_fee_can_flip_negative():
    # raw edge 3c, fee 1c -> +2c
    assert abs(clv.edge_vs_close_cents(0.55, 0.52, fee_cents=1) - 2.0) < 1e-9
    # raw edge 0.5c, fee 1c -> negative (not real edge)
    assert clv.edge_vs_close_cents(0.505, 0.50, fee_cents=1) < 0


def test_clv_report_insufficient_below_min():
    snaps = {"g1": {"t-60": {"kalshi_p": 0.40}, "tipoff": {"kalshi_p": 0.46}}}
    rep = clv.clv_report(snaps, fee_cents=1, min_samples=5)
    assert rep["insufficient"] is True
    assert rep["mean_clv_cents"] is None
    assert rep["n"] == 1


def test_clv_report_computes_means_when_enough():
    snaps = {
        f"g{i}": {"t-60": {"kalshi_p": 0.40, "book_p": 0.50},
                  "tipoff": {"kalshi_p": 0.46, "book_p": 0.50}}
        for i in range(5)
    }
    rep = clv.clv_report(snaps, fee_cents=1, min_samples=5)
    assert rep["insufficient"] is False
    assert rep["n"] == 5
    assert abs(rep["mean_clv_cents"] - 6.0) < 1e-9   # YES side, 40 -> 46


def test_clv_report_skips_games_missing_a_leg():
    snaps = {
        "g1": {"t-60": {"kalshi_p": 0.40}, "tipoff": {"kalshi_p": 0.46}},
        "g2": {"t-60": {"kalshi_p": 0.40}},           # no closing leg -> skipped
    }
    rep = clv.clv_report(snaps, fee_cents=1, min_samples=1)
    assert rep["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_clv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research.clv'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend_ml/signal_research/clv.py
"""Closing Line Value and fee-aware edge-vs-close.

Pure. Operates on captured snapshots. CLV pairs the t-60 (entry) and tipoff
(closing) Kalshi prices per game. Forward-accruing: below min_samples the
report refuses to summarize, so an empty result reads as 'not enough captured
slates yet', never 'broken'.

The position side is inferred once per game from the entry snapshot: if the
market implies YES is underpriced vs the book consensus we'd be long YES,
else NO. Absent a book leg we default to YES (the report still measures raw
price drift; side only sets the sign convention).
"""


def clv_cents(entry_price_cents, closing_price_cents, side: str) -> float:
    delta = closing_price_cents - entry_price_cents
    if side == "YES":
        return float(delta)
    if side == "NO":
        return float(-delta)
    raise ValueError(f"side must be YES or NO, got {side!r}")


def edge_vs_close_cents(p_model, market_p, fee_cents) -> float:
    return abs(p_model - market_p) * 100.0 - fee_cents


def _side_for_game(entry) -> str:
    book_p = entry.get("book_p")
    if book_p is None:
        return "YES"
    # long YES when Kalshi's implied YES prob is below the book consensus
    return "YES" if entry["kalshi_p"] < book_p else "NO"


def clv_report(snapshots_by_game, fee_cents, min_samples: int) -> dict:
    clvs, edges = [], []
    for _game, legs in snapshots_by_game.items():
        entry = legs.get("t-60")
        closing = legs.get("tipoff")
        if entry is None or closing is None:
            continue
        side = _side_for_game(entry)
        clvs.append(clv_cents(entry["kalshi_p"] * 100.0,
                              closing["kalshi_p"] * 100.0, side))
        if entry.get("book_p") is not None:
            edges.append(edge_vs_close_cents(entry["book_p"], entry["kalshi_p"], fee_cents))
    n = len(clvs)
    if n < min_samples:
        return {"n": n, "insufficient": True,
                "mean_clv_cents": None, "mean_edge_cents": None}
    return {
        "n": n,
        "insufficient": False,
        "mean_clv_cents": sum(clvs) / n,
        "mean_edge_cents": (sum(edges) / len(edges)) if edges else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_clv.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/clv.py backend_ml/signal_research/tests/test_clv.py
git commit -m "feat(signal): pure CLV + fee-aware edge-vs-close with min-sample gate"
```

---

### Task 6: CLI entrypoints (evaluate / capture / clv_report)

**Files:**
- Create: `backend_ml/signal_research/report.py`
- Create: `backend_ml/signal_research/config.py`
- Test: `backend_ml/signal_research/tests/test_report.py`

**Interfaces:**
- Consumes: all prior modules.
- Produces:
  - `config.load_fee_cents(engine_json_path=...) -> int` — reads `fee_cents_per_contract` from `trading_engine/config/engine.json` (single source of truth), fallback `1`.
  - `report.run_evaluate(dataset_df, method="isotonic", artifact_path=..., min_n=200) -> dict` — calibration report + out-of-sample recalibration eval; saves the fitted `Recalibrator`; returns a JSON-able dict.
  - `report.load_snapshots(path) -> dict` — reads the JSONL snapshot store into `{game_id: {moment: row}}`.
  - `report.main(argv)` — subcommands `evaluate`, `capture`, `clv-report`. The heavy real-artifact loading (`.pkl`, `data_engine`, live fetchers) lives ONLY in `main`, not in the unit-tested functions.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_report.py
import json
import numpy as np
import pandas as pd
from backend_ml.signal_research import report, config


def test_load_fee_cents_reads_engine_json(tmp_path):
    ej = tmp_path / "engine.json"
    ej.write_text(json.dumps({"fee_cents_per_contract": 3}))
    assert config.load_fee_cents(ej) == 3


def test_load_fee_cents_fallback(tmp_path):
    assert config.load_fee_cents(tmp_path / "missing.json") == 1


def test_run_evaluate_reports_and_saves_artifact(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p ** 2).astype(int)   # miscalibrated
    df = pd.DataFrame({"p_model": p, "outcome": y})
    art = tmp_path / "recal.json"
    out = report.run_evaluate(df, method="isotonic", artifact_path=art, min_n=200)
    assert out["calibration"]["n"] == 2000
    assert out["recalibration"]["brier_recal"] < out["recalibration"]["brier_raw"]
    assert art.exists()


def test_run_evaluate_refuses_small_sample(tmp_path):
    df = pd.DataFrame({"p_model": [0.5, 0.6], "outcome": [1, 0]})
    out = report.run_evaluate(df, artifact_path=tmp_path / "r.json", min_n=200)
    assert out["insufficient"] is True
    assert not (tmp_path / "r.json").exists()


def test_load_snapshots_groups_by_game(tmp_path):
    path = tmp_path / "snaps.jsonl"
    path.write_text(
        json.dumps({"game_id": "g1", "moment": "t-60", "kalshi_p": 0.4}) + "\n" +
        json.dumps({"game_id": "g1", "moment": "tipoff", "kalshi_p": 0.46}) + "\n")
    snaps = report.load_snapshots(path)
    assert set(snaps["g1"]) == {"t-60", "tipoff"}
    assert snaps["g1"]["tipoff"]["kalshi_p"] == 0.46
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend_ml/signal_research/config.py
"""Single-source config reads for the signal harness."""
import json
from pathlib import Path

DEFAULT_ENGINE_JSON = "trading_engine/config/engine.json"


def load_fee_cents(engine_json_path=DEFAULT_ENGINE_JSON) -> int:
    try:
        data = json.loads(Path(engine_json_path).read_text())
        return int(data["fee_cents_per_contract"])
    except Exception:
        return 1
```

```python
# backend_ml/signal_research/report.py
"""CLI + orchestration for the signal harness.

Testable functions (run_evaluate, load_snapshots) take plain data. Only main()
touches real .pkl models, data_engine, Supabase, and the live market fetchers.
"""
import argparse
import json
import os
from pathlib import Path

from backend_ml.signal_research import calibration as cal
from backend_ml.signal_research import recalibration as rc
from backend_ml.signal_research import clv as clv_mod
from backend_ml.signal_research import config

ARTIFACT_PATH = "backend_ml/signal_research/artifacts/recalibrator.json"
SNAPSHOTS_PATH = os.getenv("MARKET_SNAPSHOTS_PATH",
                           "backend_ml/signal_research/artifacts/market_snapshots.jsonl")


def run_evaluate(dataset_df, method="isotonic", artifact_path=ARTIFACT_PATH, min_n=200) -> dict:
    p = dataset_df["p_model"].to_numpy(dtype=float)
    y = dataset_df["outcome"].to_numpy(dtype=float)
    if len(p) < min_n:
        return {"insufficient": True, "n": int(len(p)),
                "message": f"need >= {min_n} labeled games, have {len(p)}"}
    calibration = cal.calibration_report(p, y)
    recal_eval = rc.evaluate_recalibration(p, y, method=method)
    rc.fit_recalibrator(p, y, method=method).save(artifact_path)
    return {"insufficient": False, "calibration": calibration,
            "recalibration": recal_eval, "artifact": str(artifact_path)}


def load_snapshots(path=SNAPSHOTS_PATH) -> dict:
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.setdefault(row["game_id"], {})[row["moment"]] = row
    return out


def _cmd_evaluate(args):
    # Heavy, real-artifact path lives here only.
    import joblib
    import sys
    sys.path.insert(0, "backend_ml")            # data_engine script-style imports
    from data_engine import build_training_dataset
    from backend_ml.signal_research import dataset as ds

    games = build_training_dataset()
    xgb = joblib.load("backend_ml/xgboost_nba_model.pkl")
    ridge = joblib.load("backend_ml/ridge_nba_model.pkl")
    scaler = joblib.load("backend_ml/feature_scaler.pkl")
    weights = json.loads(Path("backend_ml/ensemble_weights.json").read_text())
    df = ds.build_recompute_dataset(games, xgb, ridge, scaler, weights)
    out = run_evaluate(df, method=args.method)
    print(json.dumps(out, indent=2, default=str))


def _cmd_clv_report(args):
    fee = config.load_fee_cents()
    snaps = load_snapshots()
    out = clv_mod.clv_report(snaps, fee_cents=fee, min_samples=args.min_samples)
    print(json.dumps(out, indent=2))


def _cmd_capture(args):
    # Live fetchers wired here; deferred to user's live verification.
    raise SystemExit("capture requires live Kalshi/odds credentials; run manually "
                     "with fetchers wired per before-live-checklist.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="signal")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("evaluate")
    pe.add_argument("--method", default="isotonic", choices=["isotonic", "platt"])
    pe.set_defaults(func=_cmd_evaluate)

    pc = sub.add_parser("clv-report")
    pc.add_argument("--min-samples", type=int, default=30)
    pc.set_defaults(func=_cmd_clv_report)

    pcap = sub.add_parser("capture")
    pcap.set_defaults(func=_cmd_capture)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_report.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/report.py backend_ml/signal_research/config.py backend_ml/signal_research/tests/test_report.py
git commit -m "feat(signal): CLI (evaluate/capture/clv-report) with single-sourced fee"
```

---

### Task 7: Wire recalibration into publish_fair_values (flag-gated)

**Files:**
- Modify: `backend_ml/publish_fair_values.py`
- Test: `backend_ml/signal_research/tests/test_publish_recal.py`

**Interfaces:**
- Consumes: `recalibration.Recalibrator`.
- Produces: `build_fair_values(predictions, watchlist, recalibrator=None)` — when `recalibrator` is `None` the output is byte-identical to today; when provided, `p_yes` is recalibrated and `confidence` is recomputed as `max(p, 1-p)`.

The `RECALIBRATE=1` env flag (read in `main()`) loads `backend_ml/signal_research/artifacts/recalibrator.json` and passes it in. Default (unset) preserves current behavior exactly.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_publish_recal.py
from backend_ml.publish_fair_values import build_fair_values
from backend_ml.signal_research.recalibration import Recalibrator


PRED = [{"home_team_id": 1, "away_team_id": 2, "date": "2026-01-01",
         "home_win_probability": 0.80, "confidence_score": 0.80, "game_id": "g1"}]
WL = [{"home_team_id": 1, "away_team_id": 2, "game_date": "2026-01-01",
       "ticker": "T1"}]


def test_flag_off_is_unchanged():
    rows = build_fair_values(PRED, WL)                 # no recalibrator
    assert rows[0]["p_yes"] == 0.80
    assert rows[0]["confidence"] == 0.80


def test_recalibrator_transforms_and_recomputes_confidence():
    # identity-ish isotonic that maps 0.80 -> 0.60
    r = Recalibrator("isotonic", {"x": [0.0, 0.8, 1.0], "y": [0.0, 0.60, 1.0]})
    rows = build_fair_values(PRED, WL, recalibrator=r)
    assert abs(rows[0]["p_yes"] - 0.60) < 1e-9
    assert abs(rows[0]["confidence"] - 0.60) < 1e-9    # max(0.6, 0.4)


def test_recalibrated_confidence_uses_max_of_p_and_complement():
    r = Recalibrator("isotonic", {"x": [0.0, 0.8, 1.0], "y": [0.0, 0.30, 1.0]})
    rows = build_fair_values(PRED, WL, recalibrator=r)
    assert abs(rows[0]["p_yes"] - 0.30) < 1e-9
    assert abs(rows[0]["confidence"] - 0.70) < 1e-9    # max(0.3, 0.7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_publish_recal.py -v`
Expected: FAIL — `TypeError: build_fair_values() got an unexpected keyword argument 'recalibrator'`

- [ ] **Step 3: Write minimal implementation**

Modify `backend_ml/publish_fair_values.py`. Change the `build_fair_values` signature and body, and wire the flag in `main()`:

```python
def build_fair_values(predictions, watchlist, recalibrator=None):
    """Pure mapping: (model predictions, ticker watchlist) -> fair-value rows.

    Joins on (home_team_id, away_team_id, game_date). Unmapped games are
    skipped (fail-closed): a wrong ticker means trading the wrong game.

    When `recalibrator` is provided, p_yes is passed through the recalibration
    map and confidence is recomputed as max(p, 1-p). When None, behavior is
    byte-identical to the pre-recalibration publisher.
    """
    index = {(w["home_team_id"], w["away_team_id"], w["game_date"]): w["ticker"]
             for w in watchlist}
    rows = []
    for p in predictions:
        key = (p["home_team_id"], p["away_team_id"], p["date"])
        ticker = index.get(key)
        if ticker is None:
            continue
        p_yes = max(0.0, min(1.0, float(p["home_win_probability"])))
        confidence = max(0.0, min(1.0, float(p["confidence_score"])))
        if recalibrator is not None:
            p_yes = float(recalibrator.transform([p_yes])[0])
            confidence = max(p_yes, 1.0 - p_yes)
        rows.append({
            "ticker": ticker,
            "p_yes": p_yes,               # YES = home wins
            "confidence": confidence,
            "asof": datetime.datetime.utcnow().isoformat() + "Z",
            "game_id": p["game_id"],
        })
    return rows
```

Then in `main()`, after loading `watchlist` and before `build_fair_values(...)`:

```python
    recalibrator = None
    if os.getenv("RECALIBRATE") == "1":
        from backend_ml.signal_research.recalibration import Recalibrator
        recalibrator = Recalibrator.load("backend_ml/signal_research/artifacts/recalibrator.json")
    rows = build_fair_values(predictions, watchlist, recalibrator=recalibrator)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_publish_recal.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full signal suite to confirm no regressions**

Run: `python -m pytest backend_ml/signal_research/ backend_ml/test_player_impact.py -v`
Expected: PASS (all tasks' tests green)

- [ ] **Step 6: Commit**

```bash
git add backend_ml/publish_fair_values.py backend_ml/signal_research/tests/test_publish_recal.py
git commit -m "feat(signal): wire recalibration into publish_fair_values behind RECALIBRATE flag"
```

---

### Task 8: gitignore artifacts + before-live-checklist additions

**Files:**
- Modify: `backend_ml/.gitignore`
- Modify: `docs/superpowers/before-live-checklist.md`

**Interfaces:** none (docs + ignore rules).

- [ ] **Step 1: Ignore harness artifacts and secrets**

Append to `backend_ml/.gitignore`:

```
# Signal-research harness artifacts (fitted maps, captured market snapshots)
signal/artifacts/
```

Verify nothing tracked slipped in:

Run: `git status --porcelain backend_ml/signal_research/artifacts/ 2>/dev/null; echo "ok"`
Expected: no artifact files listed (only `ok`).

- [ ] **Step 2: Add the harness's deferred live items to the checklist**

Append a new section to `docs/superpowers/before-live-checklist.md`:

```markdown
## F. Signal-research harness (forward market capture — needs live creds)

These were built to run on mocked fetchers; the live paths need network + keys:

- [ ] **Live market capture.** Wire `fetch_kalshi_price(ticker)` (Kalshi REST GET
  mid/close, cents) and `fetch_two_way_odds(watchlist_row)` (two-way American
  odds for de-vig) into `signal/report.py:_cmd_capture`. Run `capture` at T-60min
  and tip-off across a real slate; confirm rows land in the snapshots JSONL.
- [ ] **CLV accrual.** CLV is zero until enough slates are captured. After N
  captured slates, `signal clv-report` should stop reporting `insufficient` and
  produce mean CLV / edge-vs-close. Confirm the fee (`engine.json`) is applied.
- [ ] **Recalibration go/no-go.** Run `signal evaluate`; only enable
  `RECALIBRATE=1` in the publisher once the out-of-sample Brier delta is a
  genuine improvement AND the served-prediction cross-check agrees. An in-sample
  gain is not sufficient.
- [ ] **Served cross-check volume.** `build_served_dataset` needs enough Supabase
  `game_predictions` with settled results to be meaningful; until then the
  recompute path stands alone.
```

- [ ] **Step 3: Commit**

```bash
git add backend_ml/.gitignore docs/superpowers/before-live-checklist.md
git commit -m "chore(signal): gitignore artifacts; add harness items to before-live checklist"
```

---

## Definition of Done

- All 8 tasks' tests pass: `python -m pytest backend_ml/signal_research/ -v` (green).
- `backend_ml/test_player_impact.py` still passes (no regression).
- `signal evaluate` produces a calibration + out-of-sample recalibration report and writes `recalibrator.json` (run locally with real models).
- `RECALIBRATE` unset → `publish_fair_values` output unchanged; set → recalibrated `p_yes` + recomputed `confidence`.
- No `.pkl`, `.csv`, artifact, or secret is committed.
- Deferred live-capture items are recorded in the before-live checklist.
