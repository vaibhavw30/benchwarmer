# Rolling-Window (Recency-Weighted) Training — Design Spec

- **Date:** 2026-07-17
- **Feature:** #5 — Rolling-window training
- **Status:** Approved design, ready for implementation planning
- **Predecessor:** #4 drift-gated retraining (merged, commit `356df33`)
- **Execution model:** TDD, subagent-driven. Every component is specified test-first with an explicit testing-pyramid layer (see §11).

---

## 1. Context & Motivation

The model is an XGBoost + Ridge ensemble predicting `HOME_WIN` for NBA games,
trained in [`backend_ml/train_model.py`](../../../backend_ml/train_model.py)
(`train_and_optimize_model`). It currently trains on **all** history with a
chronological 85/15 split and **no per-game weighting** — every game from
2015 counts exactly as much as a game from last week when the model learns the
feature→outcome relationship.

The dataset (`nba_training_cache.csv`) is one row per game: **13,182 games,
2015-10-29 → 2026-04-12, 11 seasons**, sorted by `GAME_DATE_H`.

The league is not stationary. Three-point volume, pace, and the home-court
edge have all drifted materially over 11 seasons. A relationship fit with
equal weight on 2015 and 2026 games is a blend of regimes that no longer
exist. **Recency-weighted training** lets recent games dominate what the model
learns, so the fitted relationship tracks the current regime.

### 1.1 What this is NOT (scope boundary)

This is deliberately **distinct** from per-game *features*. Injuries, hot
streaks, and short-term team form are already captured by the EWMA, MOMENTUM,
and FATIGUE features (`data_engine.calculate_rolling_features`, `ewm(span=10)`).
Those describe *this matchup's inputs*.

Recency weighting changes something different: **how much each historical game
teaches the model the mapping from features to outcome** — i.e. the learned
model weights, capturing *league-level regime shift*, not team form. We are not
adding, removing, or changing any feature. The 18-feature vector and the
`HOME_WIN` target are untouched.

---

## 2. Goals / Non-Goals

### Goals
1. Add exponential recency weighting to model training, controlled by a single
   half-life parameter `H` (in league-games).
2. Choose `H` **empirically** via a new walk-forward backtest sweep, not by
   guessing.
3. Ship end-to-end in one cycle: build the sweep, run it, pick `H*`, and wire
   `H*` into the nightly (#4-gated) retrain path.
4. Guarantee the change **cannot regress** the model: uniform weighting is
   always the incumbent baseline, and recency weighting ships only if it beats
   uniform on held-out Brier.
5. Preserve clean interaction with the #4 drift gate.

### Non-Goals
- No feature changes.
- No change to the ensemble structure (XGBoost + Ridge blend) or the 85/15
  split.
- No dropping/truncating of history (we keep all 13,182 games; weighting, not
  windowing).
- No live-trading changes; this touches model training only.
- No new hyperparameter search strategy in production — production keeps its
  nightly `GridSearchCV`, now merely weighted.

---

## 3. Design Overview

Four components, each an independent, separately testable unit:

| # | Component | File | Purpose |
|---|-----------|------|---------|
| A | `recency_weights(...)` | new: `backend_ml/recency.py` | Pure function: games-ago → sample weight vector. |
| B | `fit_ensemble(...)` | refactor in `train_model.py` | Reusable weighted training core shared by production and the sweep. |
| C | Walk-forward sweep | new: `backend_ml/halflife_sweep.py` | Pick `H*` by expanding-window backtest vs uniform baseline. |
| D | Production wiring | `train_model.py` | Apply `H*` in the nightly retrain via a single constant. |

Data flow:

```
data_engine.load_or_build_training_dataset()  ->  df (13,182 games, sorted)
                                                    |
   ┌────────────────────────────────────────────────┴───────────────┐
   │ SWEEP (offline, run once to pick H*)          PRODUCTION (nightly)│
   │ halflife_sweep.py                             train_model.py       │
   │   for each fold boundary:                       sort, 85/15 split  │
   │     for each H in grid ∪ {∞}:                    w = recency_weights│
   │       w = recency_weights(n_train, H)             (n_train, H*)     │
   │       fit_ensemble(train, w, frozen_params)     fit_ensemble(train, │
   │       score fold with calibration.brier_score     w, grid-searched) │
   │   rank H by mean held-out Brier -> H*           evaluate on test    │
   │   write report                                  save_artifacts      │
   └──────────────────────────────────────────────────────────────────┘
```

Both paths call the **same** `fit_ensemble` and the **same** `recency_weights`,
so the sweep measures exactly what production will run.

---

## 4. Component A — `recency_weights` (pure core)

New module `backend_ml/recency.py`. Single pure function, no I/O, the TDD
anchor of the whole feature (mirrors the role of `signal_research/calibration.py`).

### API
```python
def recency_weights(n: int, half_life_games: float | None) -> np.ndarray:
    """Exponential recency weights for `n` chronologically-sorted training rows.

    Rows are assumed sorted oldest→newest (ascending GAME_DATE_H), matching
    train_model's `df.sort_values('GAME_DATE_H')`. The NEWEST row (index n-1)
    gets weight 1.0; older rows decay by games-ago:

        games_ago = (n - 1) - i          # newest row -> 0
        weight_i  = 0.5 ** (games_ago / half_life_games)

    half_life_games is measured in league-games (one row = one game).

    If half_life_games is None or not finite (np.inf), returns all-ones —
    i.e. uniform weighting, identical to today's behavior.
    """
```

### Contract / edge cases (each is a unit test, §11.1)
- `half_life_games=None` → `np.ones(n)`.
- `half_life_games=np.inf` → `np.ones(n)`.
- Newest row weight is exactly `1.0`; weights strictly increase with index.
- A row exactly `H` games older than the newest has weight `0.5`.
- `n=0` → empty array (no crash). `n=1` → `[1.0]`.
- `half_life_games <= 0` → `ValueError` (nonsensical; fail loud).
- Output dtype float, length `n`, all finite, all in `(0, 1]`.

### Design notes
- Weights are **not** normalized to sum to 1. scikit-learn/XGBoost treat
  `sample_weight` as relative; normalization is unnecessary and would only
  obscure the "newest = 1.0" invariant. Documented explicitly so a reviewer
  doesn't "fix" it.
- Age is measured from the newest **training** row, which sits at the
  train/test boundary — so the rows nearest the prediction horizon carry the
  most weight. Correct: we want the fit to favor the regime closest to what
  we'll predict next.

---

## 5. Component B — `fit_ensemble` refactor

`train_and_optimize_model` is currently monolithic (loads data, splits, grid
searches, evaluates, prints, saves — all in one ~230-line function). The sweep
needs to train the *same* ensemble many times with different weights. Rather
than duplicate training logic (which would let production and the sweep drift
apart — a correctness hazard), extract a reusable core.

### Extracted API
```python
def fit_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    sample_weight: np.ndarray | None = None,
    params: dict | None = None,      # None -> run GridSearchCV (production path)
) -> FitResult:
    """Fit XGBoost + scaled Ridge on X_train/y_train.

    - sample_weight threads into GridSearchCV.fit(..., sample_weight=w),
      XGBClassifier.fit, and RidgeClassifierCV.fit(..., sample_weight=w).
    - params is None  -> production: GridSearchCV over the existing param_grid.
    - params is a dict -> sweep: fit XGBoost directly with fixed hyperparams
      (no grid search), so every H is compared on identical params and each
      fit is cheap.

    Returns FitResult(xgb_model, ridge_model, scaler).
    """
```

`FitResult` is a small dataclass (`xgb_model`, `ridge_model`, `scaler`).

### Refactor constraints (behavior-preserving)
- `train_and_optimize_model(output_dir=".")` keeps its exact public signature
  and, **with `sample_weight=None`, produces the same artifacts it does today**
  (regression-guarded — §11.4).
- The scaler is fit on `X_train` only (unchanged). Ridge probability transform
  stays `sigmoid(decision_function)` (matches `dataset.ensemble_probs` exactly,
  so the drift gate's recompute path keeps matching the trained model).
- Ensemble weight selection (the xgb_w/ridge_w search over `[0.3..0.7]`) and
  evaluation stay in `train_and_optimize_model`; `fit_ensemble` only *fits* the
  two base models + scaler. Keeps the unit small and single-purpose.

### `sample_weight` threading subtlety (documented for reviewer)
`sample_weight` passed to `GridSearchCV.fit` is routed to the estimator's
`fit` in each CV fold, but **not** to the scorer — CV model selection scoring
stays unweighted `accuracy`, which is what we want (we care about raw accuracy).
This is intentional, not a bug. A test asserts weighting changes the fit
(§11.2) but does not assert weighted CV scoring.

---

## 6. Component C — Walk-forward half-life sweep

New module `backend_ml/halflife_sweep.py`. This is genuinely new machinery:
`backtest.py` evaluates the *already-deployed* `.pkl`s on recent days and does
**not** retrain — there is no walk-forward retraining harness anywhere in the
repo today.

### Structure
- **Evaluation region:** the most recent ~2 seasons (~2,000 games) of the
  cache, split into **4 sequential, non-overlapping folds** (~500 games each).
- **Expanding window:** for fold *k*, the training set is *every game before
  fold k's first game*; the fold itself is the held-out block.
- **Frozen hyperparameters:** run `GridSearchCV` **once** up front on the
  pre-evaluation data to pick one hyperparameter set, then **freeze it** for
  every (fold × H) fit. This isolates H's effect (you're not conflating it with
  re-tuned hyperparameters) and makes each fit cheap.
- **H grid:** `{800, 1500, 2500, 4000, ∞}` league-games. `∞` (uniform) is the
  incumbent baseline and is **always** evaluated. (For scale: ~1,230 games/
  season, so 2500 ≈ 2 seasons.)
- **Per (fold, H):** `w = recency_weights(n_train, H)`;
  `fit_ensemble(train, sample_weight=w, params=frozen)`; score the fold with
  `signal_research.calibration.brier_score` (primary) and accuracy (secondary),
  scoring the held-out fold **unweighted**.
- **Aggregate:** mean held-out Brier and mean accuracy per H across the 4 folds.
  Rank by mean Brier (lower = better).

### Output
- A written report (Markdown + a machine-readable JSON/CSV) under
  `backend_ml/signal_research/artifacts/` (the existing artifacts home):
  per-H mean Brier, mean accuracy, per-fold breakdown, the frozen
  hyperparameters used, and the recommended `H*`.
- `H*` selection rule: the finite H with the lowest mean Brier, **but only if**
  it beats `∞` by ≥ the acceptance margin (§8). Otherwise `H* = None` (uniform).

### Compute cost
4 folds × 5 H = 20 cheap fixed-param fits + 1 up-front grid search. Reads only
the local `nba_training_cache.csv` — **no nba_api, no Odds API, no network**.
A few minutes total.

### Leakage safety
The sweep reuses `signal_research.dataset` conventions (features passed in,
models passed in). It must **never** import `predict.predict_games` (same
leakage guard as `dataset.py`). Training for fold *k* uses only games strictly
before fold *k*.

---

## 7. Component D — Production wiring

In `train_model.py`, add a single module-level constant:

```python
# Half-life (league-games) for recency weighting of the training split.
# Chosen by halflife_sweep.py (see docs/superpowers/specs/2026-07-17-...).
# None => uniform weighting (no recency), identical to pre-#5 behavior.
TRAIN_HALFLIFE_GAMES: float | None = None   # set to H* after the sweep runs
```

In `train_and_optimize_model`, after the 85/15 split, compute
`w = recency_weights(len(X_train), TRAIN_HALFLIFE_GAMES)` and pass it into
`fit_ensemble(..., sample_weight=w)`. The **test split stays unweighted** for
evaluation.

The sweep's winning `H*` is committed as the constant's value. If the sweep's
winner is uniform, the constant stays `None` and production behavior is
byte-identical to today.

---

## 8. Guardrails & Acceptance Criteria

1. **Uniform is always the incumbent.** `∞` is always in the sweep grid.
   Recency weighting ships only if some finite H beats uniform on **mean
   held-out Brier by ≥ 0.002** (a margin sibling to the drift gate's
   `DRIFT_MARGIN = 0.02`, but tighter because this is an averaged offline
   metric). A null result → keep `TRAIN_HALFLIFE_GAMES = None`. **#5 cannot
   make the model worse.**
2. **A "no finite H wins" outcome is a valid, documented result** — it means
   the feature→outcome regime is stable, which is itself worth knowing. The
   reusable walk-forward harness is delivered either way.
3. **Free-tier safe:** the sweep touches no paid/rate-limited API.

---

## 9. Interaction with #4 Drift Gate

The #4 gate stores the deployed model's held-out `test_brier` in
`ensemble_weights.json` and, each night, compares it against the recompute
Brier of the most-recent `DRIFT_WINDOW = 150` games
(`scheduled_retrain.measure_drift` → `should_retrain`, margin `0.02`).

Recency weighting is compatible **without changes to #4**:
- `test_brier` is still computed on the **unweighted** recent test split
  (§5 keeps the test split unweighted). `measure_drift` also scores recent
  games **unweighted** via `dataset.build_recompute_dataset`. Both sides of the
  comparison stay unweighted → the drift baseline remains apples-to-apples.
- Weighting changes *the model*, not *the yardstick*. When a weighted model
  deploys, its `test_brier` self-heals on that deploy exactly as before.
- `dataset.ensemble_probs` (the recompute path) is untouched and still matches
  the trained model's probability construction (XGB proba + sigmoid(Ridge
  decision), blended) — §5 preserves this exactly.

No edits to `scheduled_retrain.py` are required.

---

## 10. File-by-File Change Map

| File | Change |
|------|--------|
| `backend_ml/recency.py` | **New.** `recency_weights` pure function. |
| `backend_ml/halflife_sweep.py` | **New.** Walk-forward sweep + report writer + `H*` selection. |
| `backend_ml/train_model.py` | Extract `fit_ensemble`; add `TRAIN_HALFLIFE_GAMES` constant; apply weights to training split. |
| `backend_ml/test_recency.py` | **New.** Unit tests for `recency_weights`. |
| `backend_ml/test_fit_ensemble.py` | **New.** Unit/integration tests for the extracted core + regression guard. |
| `backend_ml/test_halflife_sweep.py` | **New.** Sweep harness tests on synthetic data. |
| `backend_ml/signal_research/artifacts/` | Sweep report output (git-ignored artifacts, per existing convention). |

**Do NOT commit:** `ensemble_weights.json`, any `.pkl`, any `.csv`, or the
sweep's artifact outputs. Exact-path `git add` only.

---

## 11. Testing Strategy — Testing Pyramid

Execution is TDD subagent-driven, so every component below is written
**test-first**. The pyramid is wide at the base (many fast pure-unit tests),
narrow at the top (a few slow, real-data integration checks run manually).
Tests live as sibling `test_*.py` under `backend_ml/` (matching `conftest.py`'s
`sys.path` insertion and `pytest.ini`'s `consider_namespace_packages = true`).

Test-double policy: **no `.pkl` loading in unit/integration tests.** Fit tiny
real models on tiny synthetic frames, or hand-roll fakes exposing
`predict_proba` / `decision_function` — the same discipline `signal_research`
already follows.

### 11.1 Layer 1 — Pure unit (base, fast, most numerous): `recency_weights`
`test_recency.py`. No I/O, deterministic, microsecond-fast:
- `test_none_halflife_returns_ones`
- `test_inf_halflife_returns_ones`
- `test_newest_row_weight_is_one`
- `test_weights_strictly_increasing_with_index`
- `test_one_halflife_older_row_is_half` (row `H` games back → 0.5, within tol)
- `test_two_halflives_older_row_is_quarter`
- `test_n_zero_returns_empty` / `test_n_one_returns_single_one`
- `test_nonpositive_halflife_raises`
- `test_output_dtype_length_and_range` (float, len n, all in (0,1], all finite)
- **Property-based** (`hypothesis` if available, else parametrized grid):
  for random `n` and `H>0`, weights are monotone non-decreasing, max == 1.0,
  and all in (0,1].

### 11.2 Layer 2 — Unit: `fit_ensemble` behavior
`test_fit_ensemble.py`, on a small synthetic separable frame with the 18
feature columns + `HOME_WIN`:
- `test_returns_fitted_xgb_ridge_scaler` (FitResult shape; models are fitted —
  `predict_proba` / `decision_function` callable).
- `test_fixed_params_path_skips_grid_search` (params dict → no `GridSearchCV`;
  fast).
- `test_sample_weight_changes_the_fit`: fit uniform vs a lopsided weight vector
  (all mass on one class's recent rows) → resulting `predict_proba` differ
  beyond tolerance. Proves weights actually thread through.
- `test_none_sample_weight_matches_unweighted` (None ≡ all-ones vector).
- `test_scaler_fit_on_train_only` (scaler mean/var equal manual fit on X_train).

### 11.3 Layer 3 — Integration: sweep harness on synthetic data
`test_halflife_sweep.py`. Small synthetic multi-season frame (a few hundred
rows with a **planted regime shift**: the feature→label mapping flips partway
through), fast:
- `test_infinity_always_in_grid` (uniform baseline always scored).
- `test_expanding_window_has_no_leakage`: assert fold *k*'s training indices are
  all strictly before fold *k*'s first index (guards the leakage invariant).
- `test_planted_regime_shift_favors_finite_H`: on data engineered with a late
  regime change, a finite H must beat `∞` on mean Brier — proves the sweep can
  detect recency value when it genuinely exists.
- `test_stable_data_favors_uniform`: on data with a stationary mapping, `∞`
  wins (or no finite H clears the margin) → `H* is None`. Proves the guardrail.
- `test_report_written_with_required_fields` (per-H Brier/accuracy, per-fold
  breakdown, frozen params, recommended `H*`) — write to `tmp_path`.
- `test_selection_respects_acceptance_margin` (a finite H better than `∞` but
  inside the margin does **not** win).

### 11.4 Layer 4 — Regression guard (behavior preservation)
The refactor must not change today's production behavior:
- Existing `test_train_model_save.py` (`save_artifacts`, `build_weights_config`)
  must stay green **unchanged**.
- `test_fit_ensemble.py::test_unweighted_equivalence_smoke`: on a fixed small
  synthetic frame with a fixed seed, `fit_ensemble(..., sample_weight=None)`
  and the pre-refactor inline training path produce equal `predict_proba`
  (captured as a golden array in the test). Guards the extraction.

### 11.5 Layer 5 — End-to-end (top, few, manual/slow, real data)
Run by hand during implementation validation, not in the fast CI unit run
(mark `@pytest.mark.slow` or a separate invocation):
- **Real sweep run:** execute `halflife_sweep.py` on the actual
  `nba_training_cache.csv`; record the report and the chosen `H*`. This is the
  step that produces the value committed to `TRAIN_HALFLIFE_GAMES`.
- **Weighted-vs-uniform confirmation:** run `train_and_optimize_model` once with
  `TRAIN_HALFLIFE_GAMES = H*` and once with `None`, into two temp dirs; confirm
  the weighted run's held-out `test_brier` is ≤ the uniform run's (sanity check
  that production wiring reproduces the sweep's finding).
- **Drift-gate smoke:** confirm `scheduled_retrain.measure_drift` still loads
  and scores a weighted-model artifact set without error (recompute path
  unchanged).

### 11.6 What we deliberately do NOT test
- We don't assert a specific numeric `H*` — that's data-dependent and would be a
  brittle test. `H*` is an *output*, validated by the guardrail, not pinned.
- We don't re-test scikit-learn/XGBoost internals (that `sample_weight` works);
  we test *our threading of it* (§11.2).

---

## 12. Risks & Open Questions

- **Risk: no finite H wins.** Fully anticipated and handled — ships uniform,
  documented as a stable-regime finding. Not a failure.
- **Risk: refactor changes production behavior.** Mitigated by §11.4 golden-array
  regression guard + keeping the public signature and artifact outputs identical.
- **Risk: sweep compute creep.** Bounded by frozen hyperparameters (one grid
  search, then fixed) and a 5-value H grid over 4 folds.
- **Open (deferred, not blocking):** whether to later let the sweep re-run on a
  cadence and auto-update `H*`. Out of scope for #5 — `H*` is chosen once and
  committed. Revisit only if a future drift signal suggests the optimal H has
  moved.

---

## 13. Execution Order (for the implementation plan)

1. `recency_weights` + its unit/property tests (Layer 1). Pure, no deps.
2. Extract `fit_ensemble` + its unit tests and the regression golden (Layers 2, 4).
3. `halflife_sweep.py` + synthetic-data integration tests (Layer 3).
4. Wire `TRAIN_HALFLIFE_GAMES` into `train_model.py`.
5. Run the real sweep (Layer 5), record `H*`, set the constant, run the
   weighted-vs-uniform confirmation, drift-gate smoke.

Each step is a self-contained task with its tests written first.
