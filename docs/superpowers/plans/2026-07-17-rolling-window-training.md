# Rolling-Window (Recency-Weighted) Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add empirically-tuned exponential recency weighting to NBA model training so recent games dominate the learned feature→outcome mapping, shipped end-to-end (build sweep → run it → pick `H*` → wire into nightly retrain) with a guardrail that it can never regress the model.

**Architecture:** One pure weight function (`recency.py`), a reusable weighted training core extracted from the monolithic trainer (`fit_ensemble` in `train_model.py`), an offline walk-forward sweep that picks the half-life `H*` against a uniform incumbent (`halflife_sweep.py`), and a single production constant wiring `H*` into the existing nightly `GridSearchCV` path. Both the sweep and production call the *same* `fit_ensemble` and `recency_weights`, so the sweep measures exactly what production runs.

**Tech Stack:** Python 3, XGBoost 3.2, scikit-learn 1.7 (`GridSearchCV`, `TimeSeriesSplit`, `RidgeClassifierCV`, `StandardScaler`), pandas, NumPy, pytest. Design spec: `docs/superpowers/specs/2026-07-17-rolling-window-training-design.md`.

## Global Constraints

- **Never commit** `ensemble_weights.json`, any `.pkl`, any `.csv`, or any sweep artifact under `backend_ml/signal_research/artifacts/`. None of these are git-ignored — the *only* protection is exact-path `git add`. Use exact-path `git add <file> <file>` at every commit; **never** `git add -A`, `git add .`, or `git add <dir>`.
- **No new pip dependencies.** `hypothesis` is not installed — "property-based" tests use parametrized grids, not hypothesis. Everything uses the already-installed stack above.
- **The 18-feature vector and `HOME_WIN` target are untouched.** No feature is added, removed, or changed. Features (exact order) live in `backend_ml/signal_research/dataset.py::FEATURES` and are duplicated inline in `train_model.py`; do not diverge them.
- **`train_and_optimize_model(output_dir=".")` keeps its exact public signature** and, with uniform weighting, must produce the same artifacts it does today (regression-guarded in Task 2).
- **Uniform weighting is always the incumbent.** `∞` (represented as `None`) is always in the sweep grid and always scored. Recency ships only if a finite `H` beats uniform on **mean held-out Brier by ≥ 0.002**; otherwise `TRAIN_HALFLIFE_GAMES` stays `None`. A "no finite H wins" result is a valid, documented outcome — not a failure.
- **Leakage guard:** `halflife_sweep.py` must **never** import `predict.predict_games` (same rule as `signal_research/dataset.py`). Training for fold *k* uses only games strictly before fold *k*'s first game.
- **Free-tier safe:** the sweep reads only the local `nba_training_cache.csv` — no `nba_api`, no Odds API, no network.
- **Test placement:** tests are sibling `test_*.py` under `backend_ml/` (relies on `backend_ml/conftest.py` inserting that dir on `sys.path` and `pytest.ini`'s `consider_namespace_packages = true`). Run pytest from the repo root. **No `.pkl` loading in unit/integration tests** — fit tiny real models on tiny synthetic frames.
- **No edits to `scheduled_retrain.py`.** The #4 drift gate stays compatible because the test split remains unweighted (both sides of its Brier comparison stay unweighted).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend_ml/recency.py` | **New.** Pure `recency_weights(n, half_life_games)` → weight vector. No I/O. |
| `backend_ml/train_model.py` | **Modify.** Add `FitResult` dataclass + `fit_ensemble(...)`; add `TRAIN_HALFLIFE_GAMES` constant; call `fit_ensemble` from `train_and_optimize_model`, weighting the training split. |
| `backend_ml/halflife_sweep.py` | **New.** Walk-forward sweep: fold construction, one up-front frozen-param search, per-(fold,H) fit+score, aggregation, `H*` selection, report writer, `main()`. |
| `backend_ml/test_recency.py` | **New.** Layer 1 unit tests for `recency_weights`. |
| `backend_ml/test_fit_ensemble.py` | **New.** Layer 2 unit tests + Layer 4 regression-equivalence guard. |
| `backend_ml/test_halflife_sweep.py` | **New.** Layer 3 integration tests on synthetic planted-regime-shift data + pure-function tests for folds/selection. |
| `backend_ml/test_train_model_save.py` | **Unchanged** — existing regression guard; must stay green. |
| `backend_ml/signal_research/artifacts/` | Sweep report output at run time (Task 5). **Never committed.** |

**Execution order (each task = one reviewer gate):** Task 1 (`recency_weights`, pure, no deps) → Task 2 (extract `fit_ensemble`, behavior-preserving) → Task 3 (`halflife_sweep.py`) → Task 4 (wire constant) → Task 5 (operational: run real sweep, set `H*`, confirmations).

---

### Task 1: `recency_weights` pure function (Component A, Layers 1)

**Files:**
- Create: `backend_ml/recency.py`
- Test: `backend_ml/test_recency.py`

**Interfaces:**
- Consumes: nothing (NumPy only).
- Produces: `recency_weights(n: int, half_life_games: float | None) -> np.ndarray`. Rows are oldest→newest (ascending `GAME_DATE_H`); newest row (index `n-1`) has weight `1.0`; `weight_i = 0.5 ** (((n-1)-i) / half_life_games)`. `half_life_games` `None` or non-finite → all-ones. Weights are **not** normalized. Later consumers: `fit_ensemble` (Task 2/4) and `halflife_sweep` (Task 3).

- [ ] **Step 1: Write the failing tests**

Create `backend_ml/test_recency.py`:

```python
"""Layer 1 (pure unit) tests for recency_weights."""
import numpy as np
import pytest

from recency import recency_weights


def test_none_halflife_returns_ones():
    w = recency_weights(5, None)
    assert np.array_equal(w, np.ones(5))


def test_inf_halflife_returns_ones():
    w = recency_weights(5, np.inf)
    assert np.array_equal(w, np.ones(5))


def test_newest_row_weight_is_one():
    w = recency_weights(10, 3.0)
    assert w[-1] == 1.0


def test_weights_strictly_increasing_with_index():
    w = recency_weights(20, 5.0)
    assert np.all(np.diff(w) > 0)


def test_one_halflife_older_row_is_half():
    # Row exactly H games older than the newest has weight 0.5.
    n, H = 11, 4.0
    w = recency_weights(n, H)
    # newest index = n-1 = 10; H games older => index 10 - 4 = 6
    assert w[6] == pytest.approx(0.5, abs=1e-12)


def test_two_halflives_older_row_is_quarter():
    n, H = 11, 4.0
    w = recency_weights(n, H)
    # 2H games older => index 10 - 8 = 2
    assert w[2] == pytest.approx(0.25, abs=1e-12)


def test_n_zero_returns_empty():
    w = recency_weights(0, 3.0)
    assert isinstance(w, np.ndarray) and w.shape == (0,)


def test_n_one_returns_single_one():
    w = recency_weights(1, 3.0)
    assert np.array_equal(w, np.array([1.0]))


@pytest.mark.parametrize("bad_H", [0, -1, -0.5, 0.0])
def test_nonpositive_halflife_raises(bad_H):
    with pytest.raises(ValueError):
        recency_weights(5, bad_H)


def test_output_dtype_length_and_range():
    w = recency_weights(50, 7.0)
    assert w.dtype == np.float64
    assert len(w) == 50
    assert np.all(np.isfinite(w))
    assert np.all(w > 0) and np.all(w <= 1.0)


@pytest.mark.parametrize("n", [2, 3, 8, 50, 500])
@pytest.mark.parametrize("H", [1.0, 10.0, 137.0, 2500.0])
def test_property_monotone_max_one_and_bounded(n, H):
    # Parametrized stand-in for property-based testing (hypothesis not installed).
    w = recency_weights(n, H)
    assert np.all(np.diff(w) >= 0)          # monotone non-decreasing
    assert w.max() == pytest.approx(1.0)    # newest is the max, == 1.0
    assert np.all(w > 0) and np.all(w <= 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_recency.py -q`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'recency'`.

- [ ] **Step 3: Write the minimal implementation**

Create `backend_ml/recency.py`:

```python
"""Exponential recency weights for chronologically-sorted training rows.

Pure, no I/O — the TDD anchor of the recency-weighting feature (mirrors the
role of signal_research/calibration.py). See
docs/superpowers/specs/2026-07-17-rolling-window-training-design.md.
"""
import numpy as np


def recency_weights(n: int, half_life_games: float | None) -> np.ndarray:
    """Exponential recency weights for `n` rows sorted oldest->newest.

    The NEWEST row (index n-1) gets weight 1.0; older rows decay by games-ago:

        games_ago = (n - 1) - i          # newest row -> 0
        weight_i  = 0.5 ** (games_ago / half_life_games)

    half_life_games is measured in league-games (one row = one game).

    If half_life_games is None or not finite (np.inf), returns all-ones —
    uniform weighting, identical to pre-#5 behavior.

    Weights are intentionally NOT normalized: scikit-learn / XGBoost treat
    sample_weight as relative, and normalization would obscure the
    "newest == 1.0" invariant. Do not "fix" this.
    """
    if n == 0:
        return np.empty(0, dtype=float)
    if half_life_games is None or not np.isfinite(half_life_games):
        return np.ones(n, dtype=float)
    if half_life_games <= 0:
        raise ValueError(f"half_life_games must be > 0, got {half_life_games!r}")
    idx = np.arange(n, dtype=float)
    games_ago = (n - 1) - idx
    return 0.5 ** (games_ago / half_life_games)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_recency.py -q`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git add backend_ml/recency.py backend_ml/test_recency.py
git commit -m "feat(ml): add recency_weights pure function for recency-weighted training"
```

---

### Task 2: Extract `fit_ensemble` core (Component B, Layers 2 + 4)

**Files:**
- Modify: `backend_ml/train_model.py` (add `FitResult` + `fit_ensemble`; replace inline fitting in `train_and_optimize_model` with a call to it — **uniform weighting only in this task**, no `TRAIN_HALFLIFE_GAMES` yet).
- Test: `backend_ml/test_fit_ensemble.py`

**Interfaces:**
- Consumes: nothing new (XGBoost/sklearn already imported in `train_model.py`).
- Produces:
  - `FitResult` dataclass with fields `xgb_model`, `ridge_model`, `scaler`.
  - `fit_ensemble(X_train, y_train, *, sample_weight=None, params=None) -> FitResult`. `params=None` → run `GridSearchCV` over the existing `param_grid` (production path); `params` a dict → fit `XGBClassifier(**params)` directly (sweep path, no grid search). `sample_weight` threads into `GridSearchCV.fit`, `XGBClassifier.fit`, and `RidgeClassifierCV.fit`. Scaler is `StandardScaler` fit on `X_train` only; Ridge is `RidgeClassifierCV` fit on scaled X in **both** paths.
  - Later consumers: `halflife_sweep.run_sweep` (Task 3) uses `params` + `sample_weight`; `train_and_optimize_model` (Task 4) uses `sample_weight` with `params=None`.

**Refactor notes (behavior-preserving):**
- Only the XGBoost hyperparameter search is "frozen" in the sweep path — Ridge stays `RidgeClassifierCV` (self-tuning `alpha`, cheap) in both paths, so its behavior is identical between production and sweep.
- `fit_ensemble` only *fits* the two base models + scaler. Ensemble-weight selection (`xgb_w`/`ridge_w` search over `[0.3..0.7]`) and all evaluation/printing stay in `train_and_optimize_model`.
- `sample_weight` passed to `GridSearchCV.fit` is routed to the per-fold estimator `fit` (subset per CV fold) but **not** to the scorer — CV model selection scoring stays unweighted `accuracy`. This is intentional; the tests below assert the fit changes, not weighted CV scoring.

- [ ] **Step 1: Write the failing tests**

Create `backend_ml/test_fit_ensemble.py`:

```python
"""Layer 2 (fit_ensemble behavior) + Layer 4 (regression equivalence) tests.

No .pkl loading: tiny real models on a tiny synthetic frame carrying the 18
feature columns + HOME_WIN.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from signal_research.dataset import FEATURES
from train_model import FitResult, fit_ensemble

TARGET = "HOME_WIN"


def _synth_frame(n=160, seed=0):
    """Separable-ish synthetic frame: label driven by ELO_H sign + noise."""
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(size=n) for f in FEATURES}
    df = pd.DataFrame(data)
    signal = df["ELO_H"].values + 0.3 * rng.normal(size=n)
    df[TARGET] = (signal > 0).astype(int)
    return df


def _Xy(df):
    return df[FEATURES], df[TARGET]


def test_returns_fitted_xgb_ridge_scaler():
    X, y = _Xy(_synth_frame())
    res = fit_ensemble(X, y, params={"n_estimators": 40, "max_depth": 3,
                                     "learning_rate": 0.1})
    assert isinstance(res, FitResult)
    # Fitted: these calls must not raise.
    assert res.xgb_model.predict_proba(X).shape == (len(X), 2)
    assert res.ridge_model.decision_function(res.scaler.transform(X)).shape == (len(X),)


def test_fixed_params_path_skips_grid_search():
    X, y = _Xy(_synth_frame())
    res = fit_ensemble(X, y, params={"n_estimators": 30, "max_depth": 2,
                                     "learning_rate": 0.1})
    # Fixed-param path returns a bare XGBClassifier, never a fitted GridSearchCV.
    assert isinstance(res.xgb_model, XGBClassifier)
    assert not isinstance(res.xgb_model, GridSearchCV)


def test_none_sample_weight_matches_unweighted():
    X, y = _Xy(_synth_frame())
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1}
    a = fit_ensemble(X, y, sample_weight=None, params=params)
    b = fit_ensemble(X, y, sample_weight=np.ones(len(X)), params=params)
    pa = a.xgb_model.predict_proba(X)[:, 1]
    pb = b.xgb_model.predict_proba(X)[:, 1]
    assert np.allclose(pa, pb, atol=1e-6)


def test_sample_weight_changes_the_fit():
    df = _synth_frame(n=200, seed=1)
    X, y = _Xy(df)
    params = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1}
    uniform = fit_ensemble(X, y, sample_weight=None, params=params)
    # Lopsided weights: heavily up-weight the most recent quarter of rows.
    w = np.ones(len(X))
    w[-len(X) // 4:] = 50.0
    weighted = fit_ensemble(X, y, sample_weight=w, params=params)
    p_uniform = uniform.xgb_model.predict_proba(X)[:, 1]
    p_weighted = weighted.xgb_model.predict_proba(X)[:, 1]
    # Threading proof: the two fits must differ materially somewhere.
    assert np.max(np.abs(p_uniform - p_weighted)) > 1e-3


def test_scaler_fit_on_train_only():
    df = _synth_frame()
    X, y = _Xy(df)
    res = fit_ensemble(X, y, params={"n_estimators": 20, "max_depth": 2,
                                     "learning_rate": 0.1})
    manual = StandardScaler().fit(X)
    assert np.allclose(res.scaler.mean_, manual.mean_)
    assert np.allclose(res.scaler.var_, manual.var_)


def _inline_reference_fit(X_train, y_train):
    """Byte-for-byte replica of train_and_optimize_model's pre-refactor fitting
    sequence (train_model.py lines ~99-138 before this task). This IS the
    pre-refactor behavior; fit_ensemble(params=None) must reproduce it."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = {
        "n_estimators": [150, 250],
        "learning_rate": [0.03, 0.05],
        "max_depth": [3, 4],
        "scale_pos_weight": [0.85, 0.9],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(xgb, param_grid, cv=tscv, scoring="accuracy",
                        verbose=0, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge = RidgeClassifierCV(alphas=alphas, cv=tscv, scoring="accuracy")
    ridge.fit(X_train_scaled, y_train)
    return best, ridge, scaler


def test_unweighted_equivalence_smoke():
    """Layer 4 regression guard: the extracted production path (params=None,
    sample_weight=None) reproduces the inline pre-refactor fit exactly."""
    df = _synth_frame(n=180, seed=7)
    X, y = _Xy(df)
    ref_xgb, ref_ridge, ref_scaler = _inline_reference_fit(X, y)
    res = fit_ensemble(X, y, sample_weight=None, params=None)
    # XGBoost probabilities match.
    assert np.allclose(ref_xgb.predict_proba(X)[:, 1],
                       res.xgb_model.predict_proba(X)[:, 1], atol=1e-6)
    # Ridge decisions match (scaler is equivalent -> transform equivalent).
    assert np.allclose(ref_ridge.decision_function(ref_scaler.transform(X)),
                       res.ridge_model.decision_function(res.scaler.transform(X)),
                       atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_fit_ensemble.py -q`
Expected: FAIL / collection error — `ImportError: cannot import name 'FitResult' from 'train_model'`.

- [ ] **Step 3: Add `FitResult` + `fit_ensemble` to `train_model.py`**

At the top of `backend_ml/train_model.py`, add `from dataclasses import dataclass` alongside the existing imports, and add these module-level definitions immediately **after** the `ENSEMBLE_WEIGHTS_PATH = "ensemble_weights.json"` line (line 16):

```python
from dataclasses import dataclass


@dataclass
class FitResult:
    """Fitted base models + scaler produced by fit_ensemble."""
    xgb_model: object
    ridge_model: object
    scaler: object


def fit_ensemble(X_train, y_train, *, sample_weight=None, params=None):
    """Fit XGBoost + scaled Ridge on X_train/y_train; return a FitResult.

    - sample_weight threads into GridSearchCV.fit / XGBClassifier.fit and
      RidgeClassifierCV.fit. It is applied to the per-fold estimator fit but
      not the CV scorer (model-selection scoring stays unweighted accuracy).
    - params is None  -> production path: GridSearchCV over the param grid.
    - params is a dict -> sweep path: XGBClassifier(**params) fit directly
      (no grid search), so every half-life is compared on identical params.

    The scaler is StandardScaler fit on X_train only. Ridge is
    RidgeClassifierCV (self-tuning alpha) in both paths.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    tscv = TimeSeriesSplit(n_splits=3)

    if params is None:
        param_grid = {
            'n_estimators': [150, 250],
            'learning_rate': [0.03, 0.05],
            'max_depth': [3, 4],
            'scale_pos_weight': [0.85, 0.9],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
        }
        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
        grid_search = GridSearchCV(xgb, param_grid, cv=tscv, scoring='accuracy',
                                   verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train, sample_weight=sample_weight)
        xgb_model = grid_search.best_estimator_
    else:
        xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
        xgb_model.fit(X_train, y_train, sample_weight=sample_weight)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge = RidgeClassifierCV(alphas=alphas, cv=tscv, scoring='accuracy')
    ridge.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    return FitResult(xgb_model=xgb_model, ridge_model=ridge, scaler=scaler)
```

- [ ] **Step 4: Rewire `train_and_optimize_model` to call `fit_ensemble` (uniform, no behavior change)**

In `backend_ml/train_model.py`, replace the inline fitting block (from the `# 3B. SCALE FEATURES FOR RIDGE` comment through the `print(f"🌟 Best Ridge Alpha: {ridge.alpha_}")` line — currently lines ~99-138) with:

```python
    # 3B/4. FIT ENSEMBLE (scaler + XGBoost grid search + Ridge) via shared core.
    print("🔎 Grid Search Tuning...")
    fit = fit_ensemble(X_train, y_train, sample_weight=None)
    best_model = fit.xgb_model
    ridge = fit.ridge_model
    scaler = fit.scaler
    X_test_scaled = scaler.transform(X_test)

    print(f"🌟 Best Ridge Alpha: {ridge.alpha_}")
```

Notes for the implementer:
- Delete the now-dead local `param_grid`, `xgb`, `tscv`, `grid_search`, `alphas`, `ridge = RidgeClassifierCV(...)`, `X_train_scaled`, and the standalone `X_test_scaled = scaler.transform(X_test)` further down (line ~103) — `X_test_scaled` is now defined here once. Search for `X_test_scaled` and ensure it is defined exactly once (above) and still used by the Ridge evaluation (lines ~160-168). `X_train_scaled` should no longer appear anywhere in `train_and_optimize_model`.
- `best_model = grid_search.best_estimator_` (old line ~145) must be removed — `best_model` now comes from `fit.xgb_model`. Everything downstream (`best_model.predict`, `best_model.feature_importances_`, `ridge.predict`, `ridge.coef_`, `ridge.decision_function`, `scaler`) keeps working unchanged.

- [ ] **Step 5: Run the new tests + the existing regression guard**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_fit_ensemble.py backend_ml/test_train_model_save.py -q`
Expected: PASS. `test_fit_ensemble.py` all green (including `test_unweighted_equivalence_smoke` and `test_sample_weight_changes_the_fit`); `test_train_model_save.py` green **unchanged**.

If `GridSearchCV.fit(..., sample_weight=...)` raises or warns about metadata routing on sklearn 1.7: it should not (routing is opt-in/disabled by default, and array fit-params are subset per fold). If a hard error appears, do **not** enable global metadata routing; instead report BLOCKED with the traceback — the controller decides.

- [ ] **Step 6: Commit**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git add backend_ml/train_model.py backend_ml/test_fit_ensemble.py
git commit -m "refactor(ml): extract weighted fit_ensemble core from train_and_optimize_model"
```

---

### Task 3: Walk-forward half-life sweep (Component C, Layer 3)

**Files:**
- Create: `backend_ml/halflife_sweep.py`
- Test: `backend_ml/test_halflife_sweep.py`

**Interfaces:**
- Consumes: `recency_weights` (Task 1); `fit_ensemble` (Task 2); `signal_research.dataset.FEATURES` + `ensemble_probs`; `signal_research.calibration.brier_score`.
- Produces (module-level, all importable by tests):
  - `H_GRID = [800, 1500, 2500, 4000, None]` (`None` == uniform/∞ incumbent, always scored).
  - `ACCEPTANCE_MARGIN = 0.002`; `SWEEP_BLEND = (0.5, 0.5)`; `TARGET = "HOME_WIN"`.
  - `make_folds(n_rows, n_folds, eval_games) -> list[tuple[int, int, int]]` — each tuple `(train_end, fold_start, fold_end)` with `train_end == fold_start` (expanding window; training is `[0, fold_start)`), folds contiguous over the last `eval_games` rows, last fold absorbing any remainder.
  - `pick_frozen_params(X, y) -> dict` — one `GridSearchCV` over the production grid on pre-evaluation data; returns the winning XGB hyperparameter dict.
  - `score_fold(df, train_end, fold_start, fold_end, half_life, frozen_params) -> dict` — fits on `[0, fold_start)` with recency weights, scores `[fold_start, fold_end)` unweighted; returns `{"brier": float, "accuracy": float, "n_train": int, "n_fold": int}`.
  - `run_sweep(df, h_grid=H_GRID, n_folds=4, eval_games=2000, frozen_params=None) -> dict` — result with keys `per_h` (`{repr(H): {"mean_brier","mean_accuracy","folds":[...]}}`), `frozen_params`, `h_star`, `n_folds`, `eval_games`.
  - `select_h_star(per_h, margin=ACCEPTANCE_MARGIN) -> float | None` — best finite H by mean Brier, only if it beats the `None` (uniform) entry by ≥ margin; else `None`.
  - `write_report(result, out_dir) -> dict` — writes `halflife_sweep.json` and `halflife_sweep.md`; returns their paths.
  - `main() -> int` — loads cache, runs sweep, writes report under `signal_research/artifacts/`, prints `H*`.

- [ ] **Step 1: Write the failing tests**

Create `backend_ml/test_halflife_sweep.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_halflife_sweep.py -q`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'halflife_sweep'`.

- [ ] **Step 3: Write the sweep module**

Create `backend_ml/halflife_sweep.py`:

```python
"""Walk-forward sweep to pick the recency half-life H* against a uniform baseline.

Expanding-window backtest over the most recent games of nba_training_cache.csv.
For each fold and each candidate half-life (plus the uniform incumbent, None),
fit the SAME ensemble via train_model.fit_ensemble with frozen hyperparameters
and score the held-out fold with signal_research.calibration.brier_score. H* is
the finite half-life that beats uniform mean Brier by >= ACCEPTANCE_MARGIN, else
None (uniform).

Reads only the local cache — no network. LEAKAGE GUARD: never import
predict.predict_games; fold k trains only on games strictly before fold k.
See docs/superpowers/specs/2026-07-17-rolling-window-training-design.md.
"""
import json
import os

import numpy as np
import pandas as pd

from recency import recency_weights
from train_model import fit_ensemble
from signal_research.dataset import FEATURES, ensemble_probs
from signal_research import calibration

TARGET = "HOME_WIN"
H_GRID = [800, 1500, 2500, 4000, None]   # None == uniform/inf incumbent (always scored)
ACCEPTANCE_MARGIN = 0.002                # finite H must beat uniform mean Brier by this
SWEEP_BLEND = (0.5, 0.5)                 # fixed (xgb, ridge) blend to isolate H's effect
DEFAULT_N_FOLDS = 4
DEFAULT_EVAL_GAMES = 2000
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "signal_research", "artifacts")


def make_folds(n_rows, n_folds, eval_games):
    """Contiguous expanding-window folds over the last `eval_games` rows.

    Returns a list of (train_end, fold_start, fold_end); training is [0, fold_start),
    the held-out block is [fold_start, fold_end). train_end == fold_start. The last
    fold absorbs any remainder so the folds exactly cover the eval region.
    """
    if eval_games > n_rows:
        raise ValueError(f"eval_games={eval_games} exceeds n_rows={n_rows}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    eval_start = n_rows - eval_games
    fold_size = eval_games // n_folds
    if fold_size < 1:
        raise ValueError("eval_games too small for n_folds (fold_size < 1)")
    folds = []
    start = eval_start
    for k in range(n_folds):
        end = start + fold_size if k < n_folds - 1 else n_rows
        folds.append((start, start, end))
        start = end
    return folds


def pick_frozen_params(X, y):
    """One GridSearchCV over the production grid; return the winning XGB params."""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from xgboost import XGBClassifier
    param_grid = {
        'n_estimators': [150, 250],
        'learning_rate': [0.03, 0.05],
        'max_depth': [3, 4],
        'scale_pos_weight': [0.85, 0.9],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=TimeSeriesSplit(n_splits=3),
                        scoring='accuracy', verbose=0, n_jobs=-1)
    grid.fit(X, y)
    return dict(grid.best_params_)


def score_fold(df, train_end, fold_start, fold_end, half_life, frozen_params):
    """Fit on [0, fold_start) with recency weights; score [fold_start, fold_end)."""
    train_df = df.iloc[:fold_start]
    fold_df = df.iloc[fold_start:fold_end]
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    w = recency_weights(len(X_train), half_life)
    fit = fit_ensemble(X_train, y_train, sample_weight=w, params=frozen_params)
    xgb_w, ridge_w = SWEEP_BLEND
    p = ensemble_probs(fold_df, fit.xgb_model, fit.ridge_model, fit.scaler,
                       xgb_w, ridge_w)
    y_true = fold_df[TARGET].astype(int).values
    brier = calibration.brier_score(p, y_true)
    accuracy = float(np.mean((p > 0.5).astype(int) == y_true))
    return {"brier": brier, "accuracy": accuracy,
            "n_train": int(len(X_train)), "n_fold": int(len(fold_df))}


def run_sweep(df, h_grid=H_GRID, n_folds=DEFAULT_N_FOLDS,
              eval_games=DEFAULT_EVAL_GAMES, frozen_params=None):
    """Expanding-window sweep. Returns a result dict (see module docstring)."""
    df = df.sort_values("GAME_DATE_H").reset_index(drop=True)
    folds = make_folds(len(df), n_folds, eval_games)
    if frozen_params is None:
        eval_start = folds[0][1]
        pre = df.iloc[:eval_start]
        frozen_params = pick_frozen_params(pre[FEATURES], pre[TARGET])

    per_h = {}
    for H in h_grid:
        fold_stats = [score_fold(df, te, fs, fe, H, frozen_params)
                      for (te, fs, fe) in folds]
        per_h[repr_h(H)] = {
            "mean_brier": float(np.mean([s["brier"] for s in fold_stats])),
            "mean_accuracy": float(np.mean([s["accuracy"] for s in fold_stats])),
            "folds": fold_stats,
        }
    h_star = select_h_star(per_h)
    return {"per_h": per_h, "frozen_params": frozen_params, "h_star": h_star,
            "n_folds": n_folds, "eval_games": eval_games}


def repr_h(H):
    """Stable dict key for a half-life value (None -> 'None')."""
    return "None" if H is None else str(int(H))


def select_h_star(per_h, margin=ACCEPTANCE_MARGIN):
    """Best finite H by mean Brier, only if it beats uniform by >= margin; else None."""
    uniform = per_h.get("None")
    if uniform is None:
        return None
    uniform_brier = uniform["mean_brier"]
    best_H, best_brier = None, None
    for key, stats in per_h.items():
        if key == "None":
            continue
        if best_brier is None or stats["mean_brier"] < best_brier:
            best_H, best_brier = int(key), stats["mean_brier"]
    if best_H is None:
        return None
    if best_brier <= uniform_brier - margin:
        return best_H
    return None


def write_report(result, out_dir=ARTIFACT_DIR):
    """Write halflife_sweep.json (machine) + halflife_sweep.md (human)."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "halflife_sweep.json")
    md_path = os.path.join(out_dir, "halflife_sweep.md")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    lines = ["# Half-life sweep report", ""]
    lines.append(f"- Folds: {result['n_folds']}  Eval games: {result['eval_games']}")
    lines.append(f"- Frozen params: `{result['frozen_params']}`")
    lines.append(f"- **Recommended H\\*: {result['h_star']}** "
                 f"(None => uniform; recency did not clear the margin)")
    lines += ["", "| H | mean Brier | mean Accuracy |", "|---|---|---|"]
    for key, stats in result["per_h"].items():
        lines.append(f"| {key} | {stats['mean_brier']:.5f} | "
                     f"{stats['mean_accuracy']:.4f} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"json": json_path, "md": md_path}


def main():
    from data_engine import load_or_build_training_dataset
    df = load_or_build_training_dataset()
    if df.empty:
        print("❌ Dataset is empty!")
        return 1
    result = run_sweep(df)
    paths = write_report(result)
    print(f"\n✅ Sweep complete. H* = {result['h_star']}")
    print(f"   Report: {paths['md']}")
    print(f"   JSON:   {paths['json']}")
    print("   (Artifacts are NOT committed — set TRAIN_HALFLIFE_GAMES from H* by hand.)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_halflife_sweep.py -q`
Expected: PASS. If `test_planted_regime_shift_favors_finite_H` is flaky, do **not** weaken the assertion — the planted shift is strong by construction (mapping fully flips at the midpoint and the eval region is entirely in the flipped regime); report BLOCKED with the observed Briers so the controller can adjust the fixture. Do not touch `test_stable_data_favors_uniform`'s `is None` assertion (the margin guarantees it).

- [ ] **Step 5: Commit**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git add backend_ml/halflife_sweep.py backend_ml/test_halflife_sweep.py
git commit -m "feat(ml): add walk-forward half-life sweep to pick H* vs uniform baseline"
```

---

### Task 4: Wire `TRAIN_HALFLIFE_GAMES` into production (Component D)

**Files:**
- Modify: `backend_ml/train_model.py` (add the constant; apply recency weights to the training split).
- Test: `backend_ml/test_fit_ensemble.py` (add one wiring test).

**Interfaces:**
- Consumes: `recency_weights` (Task 1), `fit_ensemble` (Task 2).
- Produces: module-level `TRAIN_HALFLIFE_GAMES: float | None` (default `None`). `train_and_optimize_model` now weights the training split by `recency_weights(len(X_train), TRAIN_HALFLIFE_GAMES)`; test split stays unweighted.

- [ ] **Step 1: Write the failing test**

Append to `backend_ml/test_fit_ensemble.py`:

```python
def test_train_model_exposes_halflife_constant_defaulting_none():
    import train_model
    assert hasattr(train_model, "TRAIN_HALFLIFE_GAMES")
    assert train_model.TRAIN_HALFLIFE_GAMES is None   # uniform until the sweep sets H*


def test_train_and_optimize_uses_recency_weights_on_train_split(monkeypatch):
    """train_and_optimize_model must weight the TRAIN split via recency_weights
    (len == train split size) and pass those weights into fit_ensemble; the test
    split stays unweighted."""
    import numpy as np
    import pandas as pd
    import train_model
    from signal_research.dataset import FEATURES

    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["HOME_WIN"] = (df["ELO_H"].values > 0).astype(int)
    df["GAME_DATE_H"] = pd.date_range("2015-10-01", periods=n, freq="D")

    monkeypatch.setattr(train_model, "TRAIN_HALFLIFE_GAMES", 50.0)
    # Stub data load to our synthetic frame.
    import data_engine
    monkeypatch.setattr(data_engine, "load_or_build_training_dataset", lambda: df)

    captured = {}
    real_fit = train_model.fit_ensemble

    def spy_fit(X_train, y_train, *, sample_weight=None, params=None):
        captured["sample_weight"] = sample_weight
        captured["n_train"] = len(X_train)
        return real_fit(X_train, y_train, sample_weight=sample_weight, params=params)

    monkeypatch.setattr(train_model, "fit_ensemble", spy_fit)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert train_model.train_and_optimize_model(output_dir=tmp) is True

    w = captured["sample_weight"]
    assert w is not None
    assert len(w) == captured["n_train"]              # weights sized to TRAIN split
    assert w[-1] == pytest.approx(1.0)                # newest train row weight 1.0
    assert np.all(np.diff(w) > 0)                      # recency decay applied
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_fit_ensemble.py -k "halflife_constant or recency_weights_on_train" -q`
Expected: FAIL — `AttributeError: module 'train_model' has no attribute 'TRAIN_HALFLIFE_GAMES'` (and the wiring test fails because `sample_weight` is `None`).

- [ ] **Step 3: Add the constant and apply weights**

In `backend_ml/train_model.py`:

1. Add the constant next to the other module-level constants (immediately after `ENSEMBLE_WEIGHTS_PATH = "ensemble_weights.json"`):

```python
# Half-life (league-games) for recency weighting of the training split.
# Chosen by halflife_sweep.py (docs/superpowers/specs/2026-07-17-rolling-window-training-design.md).
# None => uniform weighting (no recency), byte-identical to pre-#5 behavior.
TRAIN_HALFLIFE_GAMES: float | None = None   # set to H* after the sweep runs
```

2. Add the import near the top of the file (with the other imports):

```python
from recency import recency_weights
```

3. In `train_and_optimize_model`, change the fit call added in Task 2 from:

```python
    fit = fit_ensemble(X_train, y_train, sample_weight=None)
```

to:

```python
    train_weights = recency_weights(len(X_train), TRAIN_HALFLIFE_GAMES)
    fit = fit_ensemble(X_train, y_train, sample_weight=train_weights)
```

(The test split stays unweighted — no change to the evaluation path.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_fit_ensemble.py backend_ml/test_train_model_save.py -q`
Expected: PASS. With `TRAIN_HALFLIFE_GAMES = None` (the committed default), `recency_weights` returns all-ones, so production behavior is unchanged and `test_train_model_save.py` stays green.

- [ ] **Step 5: Commit**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git add backend_ml/train_model.py backend_ml/test_fit_ensemble.py
git commit -m "feat(ml): wire TRAIN_HALFLIFE_GAMES recency weighting into nightly retrain"
```

---

### Task 5: Run the real sweep, set `H*`, and confirm (Layer 5 — operational)

**Files:**
- Modify (only if the sweep selects a finite `H*`): `backend_ml/train_model.py` — set `TRAIN_HALFLIFE_GAMES`.
- No test file; this task is manual validation. Its deliverable is the recorded sweep result and the (possibly unchanged) constant.

**Interfaces:** consumes the finished `halflife_sweep.py` and the wired `train_model.py`. Produces the committed value of `TRAIN_HALFLIFE_GAMES`.

This task runs real models on the real 13,182-game cache. It is CPU-heavy (minutes), not a fast unit test. **Never `git add`** any `.pkl`, `.csv`, `ensemble_weights.json`, or `signal_research/artifacts/*` produced here.

- [ ] **Step 1: Confirm the full fast suite is green first**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m pytest backend_ml/test_recency.py backend_ml/test_fit_ensemble.py backend_ml/test_halflife_sweep.py backend_ml/test_train_model_save.py -q`
Expected: PASS (all layers 1–4 green).

- [ ] **Step 2: Run the real sweep**

Run: `cd /Users/vaibhav.wudaru/benchwarmer-nba && python3 -m backend_ml.halflife_sweep` — or, if module invocation fails due to the namespace-package layout, `cd backend_ml && python3 halflife_sweep.py`.
Expected: prints `H* = <value>` and writes `signal_research/artifacts/halflife_sweep.{json,md}`. Read the report and note the per-H mean Brier table and the recommended `H*`.

- [ ] **Step 3: Set the constant to `H*`**

- If `H*` is a finite number, edit `TRAIN_HALFLIFE_GAMES` in `backend_ml/train_model.py` to that number (e.g. `TRAIN_HALFLIFE_GAMES: float | None = 2500`).
- If `H*` is `None` (no finite H cleared the 0.002 margin), **leave the constant as `None`** — this is the valid "stable regime" outcome; production stays byte-identical to today. Record the finding in the commit message.

- [ ] **Step 4: Weighted-vs-uniform confirmation (only if `H*` is finite)**

Confirm production wiring reproduces the sweep's finding. From `backend_ml/`:

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba/backend_ml
# Uniform baseline into a temp dir:
python3 -c "import train_model as t; t.TRAIN_HALFLIFE_GAMES=None; t.train_and_optimize_model(output_dir='/tmp/nba_uniform')"
# Weighted (H*) into a second temp dir:
python3 -c "import train_model as t; t.TRAIN_HALFLIFE_GAMES=<H*>; t.train_and_optimize_model(output_dir='/tmp/nba_weighted')"
# Compare held-out test_brier (lower is better for the weighted run):
python3 -c "import json; u=json.load(open('/tmp/nba_uniform/ensemble_weights.json'))['test_brier']; w=json.load(open('/tmp/nba_weighted/ensemble_weights.json'))['test_brier']; print('uniform',u,'weighted',w,'OK' if w<=u else 'REGRESSION')"
```

Expected: `weighted <= uniform` (`OK`). If it prints `REGRESSION`, the wiring or `H*` is wrong — report BLOCKED with both Brier values; do not commit a finite `H*` that regresses held-out Brier. (`/tmp/nba_*` are scratch dirs — never committed.)

- [ ] **Step 5: Drift-gate smoke (recompute path still loads a weighted artifact set)**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba/backend_ml
python3 -c "
import data_engine, scheduled_retrain as s
df = data_engine.load_or_build_training_dataset()
b, n = s.measure_drift(df, live_dir='/tmp/nba_weighted')
print('recompute brier', b, 'n', n)
assert b is not None and n > 0, 'measure_drift failed on weighted artifacts'
print('drift-gate smoke OK')
"
```

Expected: prints a finite Brier and `drift-gate smoke OK`, proving `scheduled_retrain.measure_drift` scores a weighted-model artifact set without error (recompute path unchanged). If `H*` was `None`, run this against `/tmp/nba_uniform` (train it first) to smoke the path anyway.

- [ ] **Step 6: Commit the constant (train_model.py only)**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git status --short   # verify ONLY train_model.py is staged-worthy; no .pkl/.csv/.json/artifacts
git add backend_ml/train_model.py
git commit -m "feat(ml): set TRAIN_HALFLIFE_GAMES=<H* or None> from real half-life sweep"
```

Before committing, run `git status --short` and confirm no `*.pkl`, `*.csv`, `ensemble_weights.json`, or `signal_research/artifacts/*` is being added. If `H*` was `None` and `train_model.py` is therefore unchanged, there is nothing to commit — record the sweep finding in the final review notes instead.

---

## Self-Review

**1. Spec coverage:**
- §4 Component A → Task 1 (all §4.1 edge cases mapped to `test_recency.py`, incl. `n=0`, `n=1`, non-positive raise, dtype/range, "0.5 at H games back", parametrized monotonicity property). ✅
- §5 Component B → Task 2 (`FitResult`, `fit_ensemble`, both param paths, sample_weight threading subtlety asserted via `test_sample_weight_changes_the_fit`, scaler-on-train-only, `sigmoid(decision_function)` preserved through `ensemble_probs`). ✅
- §5 behavior preservation / §11.4 regression guard → Task 2 `test_unweighted_equivalence_smoke` (inline replica) + `test_train_model_save.py` stays green. ✅
- §6 Component C → Task 3 (`make_folds` expanding window + no-leakage, `pick_frozen_params` one up-front grid search, per-(fold,H) fit via shared `fit_ensemble` + frozen params, `brier_score` primary + accuracy secondary, unweighted fold scoring, aggregation, report writer). ✅
- §6 H grid `{800,1500,2500,4000,∞}` → `H_GRID = [800,1500,2500,4000,None]`; ∞ always scored → `test_infinity_always_in_grid`. ✅
- §7 Component D → Task 4 (`TRAIN_HALFLIFE_GAMES` constant, weights on train split only, test split unweighted). ✅
- §8 guardrails → `ACCEPTANCE_MARGIN=0.002`, `select_h_star` margin logic (`test_select_h_star_*`), null result → `None` (`test_stable_data_favors_uniform`). ✅
- §9 #4 drift interaction → no `scheduled_retrain.py` edits; Task 5 Step 5 drift-gate smoke. ✅
- §11.5 Layer 5 → Task 5 (real sweep, weighted-vs-uniform, drift smoke). ✅
- §11.3 planted-regime-shift + stable-data + margin + report tests → Task 3. ✅

**2. Placeholder scan:** No "TBD"/"TODO"/"add error handling"/"similar to Task N" — every code step contains complete code. The only intentional deferred value is `TRAIN_HALFLIFE_GAMES = None` (set in Task 5 from the sweep output; this is a runtime result, not a plan placeholder). ✅

**3. Type consistency:** `recency_weights(n, half_life_games)` signature identical across Tasks 1/3/4. `fit_ensemble(X_train, y_train, *, sample_weight=None, params=None) -> FitResult` identical across Tasks 2/3/4. `FitResult` fields `xgb_model`/`ridge_model`/`scaler` used consistently by `score_fold` via `ensemble_probs(..., fit.xgb_model, fit.ridge_model, fit.scaler, ...)`. `select_h_star`/`make_folds`/`run_sweep`/`write_report`/`score_fold`/`repr_h` signatures match their test call sites. `per_h` keys are `repr_h(H)` (`"None"`/`"800"`…) consistently in `run_sweep`, `select_h_star`, `write_report`, and tests. ✅

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-17-rolling-window-training.md`.**
