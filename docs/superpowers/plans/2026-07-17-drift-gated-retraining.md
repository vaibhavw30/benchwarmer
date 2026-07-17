# Drift-Gated Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the nightly retrain job spend a full GridSearch train only when the deployed model has measurably drifted, instead of every in-season night.

**Architecture:** A gate inside `scheduled_retrain.main()` sits *in front of* the existing (unchanged) train→validate→swap flow. Each night it refreshes the training cache once, scores the deployed model's recent-window Brier via the existing `signal_research` recompute path, compares it to a baseline Brier stored in `ensemble_weights.json`, and only enters the retrain flow if the model drifted (or if the signal can't be trusted — every uncertain case falls back to retraining, so the gate can only ever *reduce* retrains, never regress below today's always-retrain behavior).

**Tech Stack:** Python 3, pytest, pandas, numpy, joblib, scikit-learn (`brier_score_loss`), existing in-repo modules `data_engine`, `train_model`, `signal_research.dataset`, `signal_research.calibration`.

**Spec:** [docs/superpowers/specs/2026-07-17-drift-gated-retraining-design.md](../specs/2026-07-17-drift-gated-retraining-design.md)

## Global Constraints

- **Working directory for all commands:** `backend_ml/`. The venv interpreter is `../.venv/bin/python` (i.e. repo-root `.venv`). Tests run from repo root as `.venv/bin/python -m pytest backend_ml/<file> -q` OR from `backend_ml/` as `../.venv/bin/python -m pytest <file> -q`. This plan uses the `backend_ml/`-relative form.
- **`signal_research` import form inside `scheduled_retrain.py`:** `from signal_research import dataset, calibration` (verified importable with `backend_ml/` on `sys.path`; `conftest.py` guarantees that path in tests, and launchd sets `WorkingDirectory` to `backend_ml/`). Do NOT use `from backend_ml.signal_research import ...` in production code — that form is only for the `signal_research/tests/` suite which runs from repo root.
- **Tuning constants (named module constants in `scheduled_retrain.py`), exact values:**
  - `DRIFT_WINDOW = 150`
  - `MIN_RECENT = 100`
  - `DRIFT_MARGIN = 0.02`
- **Deployed artifact filenames (exact), loaded from the live dir by `measure_drift`:** `xgboost_nba_model.pkl`, `ridge_nba_model.pkl`, `feature_scaler.pkl`, `ensemble_weights.json`. These already match `scheduled_retrain.ARTIFACTS`.
- **`measure_drift` must never raise:** any load/score error returns `(None, 0)` so the caller hits the retrain-anyway fallback. A broken/missing artifact must never become a silent skip.
- **Do NOT change the validate-before-swap logic** (`should_deploy`, `deploy_artifacts`, `read_test_accuracy`), the plist, or the launchd entry point. The gate only decides whether to enter the existing flow.
- **Never `git add -A` / `git add .`.** Stage only the exact files named in each Commit step. In particular, NEVER stage `backend_ml/ensemble_weights.json` (it carries an unrelated uncommitted local re-tune), any `*.pkl`, or any `*.csv`.
- **Backward compatibility:** existing tests in `test_scheduled_retrain.py` and `test_train_model_save.py` must still pass. `log_run`'s existing single-line output format for `deployed` / `rejected` / `skipped: offseason` must be preserved (new Brier fields are appended only when provided).

## File Structure

- **`backend_ml/train_model.py`** (modify) — add a pure `build_weights_config(...)` helper that assembles the `weights_config` dict including a new numeric `test_brier` field; compute `ensemble_brier` beside the existing ensemble evaluation and route the dict through the helper. This is the drift baseline written on every deploy.
- **`backend_ml/scheduled_retrain.py`** (modify) — add tuning constants; add pure `read_baseline_brier(...)` and `should_retrain(...)`; add `measure_drift(...)`; extend `log_run(...)` with optional Brier context; insert the gate into `main()`.
- **`backend_ml/test_train_model_save.py`** (modify) — add unit tests for `build_weights_config`.
- **`backend_ml/test_scheduled_retrain.py`** (modify) — add unit tests for the new pure functions and `measure_drift`; update the `fake_trainer` fixture and add `main()` gate tests.

No new files. All changes live beside the code they touch, matching the existing single-directory layout of `backend_ml/`.

---

### Task 1: Record the drift baseline (`test_brier`) in `train_model.py`

Introduce a pure `build_weights_config` helper (unit-testable without running GridSearch or touching nba_api), add `test_brier`, and route the existing training code through it.

**Files:**
- Modify: `backend_ml/train_model.py` (imports near line 9; the `weights_config` assembly near lines 244-250; add `ensemble_brier` beside the ensemble evaluation near lines 189-190)
- Test: `backend_ml/test_train_model_save.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `build_weights_config(xgb_weight, ridge_weight, test_accuracy, test_brier, train_date=None) -> dict` — returns `{"xgb_weight", "ridge_weight", "test_accuracy" (float), "test_brier" (float), "train_date" (str)}`. When `train_date is None`, it is set to `str(pd.Timestamp.now())`.
  - Every deployed `ensemble_weights.json` now contains a numeric `test_brier` (consumed by Task 2's `read_baseline_brier`).

- [ ] **Step 1: Write the failing tests**

Add to `backend_ml/test_train_model_save.py` (keep the existing imports; extend the `from train_model import ...` line):

```python
from train_model import build_weights_config


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
```

Update the existing module-level `WEIGHTS` constant so the round-trip test still reflects a realistic config (add `test_brier`):

```python
WEIGHTS = {"xgb_weight": 0.7, "ridge_weight": 0.3, "test_accuracy": 0.66, "test_brier": 0.2, "train_date": "x"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_train_model_save.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_weights_config' from 'train_model'`.

- [ ] **Step 3: Add the helper and wire it in**

In `backend_ml/train_model.py`, extend the sklearn.metrics import (line 9) to include `brier_score_loss`:

```python
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score, brier_score_loss
```

Add the pure helper directly below `save_artifacts` (after line 24):

```python
def build_weights_config(xgb_weight, ridge_weight, test_accuracy, test_brier, train_date=None):
    """Assemble the ensemble_weights.json payload.

    test_brier is the held-out ensemble Brier score; it is the baseline the
    nightly drift gate compares against (see scheduled_retrain.measure_drift).
    """
    return {
        "xgb_weight": xgb_weight,
        "ridge_weight": ridge_weight,
        "test_accuracy": float(test_accuracy),
        "test_brier": float(test_brier),
        "train_date": str(train_date if train_date is not None else pd.Timestamp.now()),
    }
```

In `train_and_optimize_model`, beside the existing ensemble evaluation (right after `ensemble_logloss = log_loss(...)` near line 190), compute the Brier:

```python
    ensemble_brier = brier_score_loss(y_test, ensemble_probs_best)
```

Then replace the inline `weights_config = {...}` dict (lines 244-249) with:

```python
    weights_config = build_weights_config(xgb_w, ridge_w, best_ensemble_acc, ensemble_brier)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_train_model_save.py -q`
Expected: PASS (all tests, including the two pre-existing `save_artifacts` tests).

- [ ] **Step 5: Commit**

```bash
cd backend_ml
git add train_model.py test_train_model_save.py
git commit -m "feat: record held-out test_brier baseline in ensemble weights"
```

---

### Task 2: `read_baseline_brier` — read the baseline back

A mirror of the existing `read_test_accuracy`, tolerant of a missing/legacy `test_brier` (returns `None`, which Task 3 treats as "no baseline → retrain").

**Files:**
- Modify: `backend_ml/scheduled_retrain.py` (add the function beside `read_test_accuracy`, ~line 45-51)
- Test: `backend_ml/test_scheduled_retrain.py`

**Interfaces:**
- Consumes: the `test_brier` field written by Task 1.
- Produces: `read_baseline_brier(weights_path) -> float | None` — the `test_brier` from an `ensemble_weights.json`, or `None` if the file is absent/unreadable/missing-the-key/non-numeric. Consumed by `main()` in Task 5.

- [ ] **Step 1: Write the failing tests**

Add to `backend_ml/test_scheduled_retrain.py` (below the existing `read_test_accuracy` tests):

```python
# --- read_baseline_brier ------------------------------------------------------

def test_read_baseline_brier_happy_path(tmp_path):
    p = tmp_path / "w.json"
    p.write_text('{"test_accuracy": 0.66, "test_brier": 0.21}')
    assert sr.read_baseline_brier(str(p)) == 0.21


@pytest.mark.parametrize("content", [
    None,                                          # file absent
    "not json",                                    # unparseable
    '{"test_accuracy": 0.66}',                     # key missing (legacy model)
    '{"test_accuracy": 0.66, "test_brier": null}', # non-numeric
])
def test_read_baseline_brier_returns_none_on_problems(tmp_path, content):
    p = tmp_path / "w.json"
    if content is not None:
        p.write_text(content)
    assert sr.read_baseline_brier(str(p)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_scheduled_retrain.py -k read_baseline_brier -q`
Expected: FAIL — `AttributeError: module 'scheduled_retrain' has no attribute 'read_baseline_brier'`.

- [ ] **Step 3: Implement**

In `backend_ml/scheduled_retrain.py`, add directly below `read_test_accuracy` (after line 51):

```python
def read_baseline_brier(weights_path):
    """test_brier from an ensemble_weights.json; None if absent/unreadable/non-numeric."""
    try:
        return float(json.load(open(weights_path))["test_brier"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_scheduled_retrain.py -k read_baseline_brier -q`
Expected: PASS (5 cases: 1 happy + 4 parametrized).

- [ ] **Step 5: Commit**

```bash
cd backend_ml
git add scheduled_retrain.py test_scheduled_retrain.py
git commit -m "feat: read test_brier baseline from ensemble weights"
```

---

### Task 3: `should_retrain` — the pure gate decision

The tuning constants plus a pure boolean: retrain on any uncertainty, skip only on confident no-drift.

**Files:**
- Modify: `backend_ml/scheduled_retrain.py` (add constants near the other module constants ~line 17-19; add the function after `should_deploy` ~line 42)
- Test: `backend_ml/test_scheduled_retrain.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - Module constants `DRIFT_WINDOW = 150`, `MIN_RECENT = 100`, `DRIFT_MARGIN = 0.02`.
  - `should_retrain(recent_brier, baseline_brier, n_recent, min_recent=MIN_RECENT, margin=DRIFT_MARGIN) -> bool` — `True` (retrain) if ANY of: `n_recent < min_recent`, `recent_brier is None`, `baseline_brier is None`, or `recent_brier > baseline_brier + margin`. `False` (skip) only on confident no-drift. Consumed by `main()` in Task 5.

- [ ] **Step 1: Write the failing tests**

Add to `backend_ml/test_scheduled_retrain.py`:

```python
# --- should_retrain -----------------------------------------------------------

@pytest.mark.parametrize("recent,baseline,n,expected", [
    (0.30, 0.20, 200, True),    # drift: recent worse than baseline+margin(0.02)
    (0.20, 0.20, 200, False),   # no drift: equal, within margin
    (0.215, 0.20, 200, False),  # no drift: worse but within margin (0.215 <= 0.22)
    (0.221, 0.20, 200, True),   # drift: just beyond margin (0.221 > 0.22)
    (0.10, 0.20, 50, True),     # low data: n_recent < MIN_RECENT overrides no-drift
    (None, 0.20, 200, True),    # measurement error: recent is None -> retrain
    (0.20, None, 200, True),    # no baseline yet -> retrain
])
def test_should_retrain(recent, baseline, n, expected):
    assert sr.should_retrain(recent, baseline, n) is expected


def test_should_retrain_boundary_exactly_at_margin_is_no_drift():
    # recent == baseline + margin exactly -> not strictly greater -> skip
    assert sr.should_retrain(0.22, 0.20, 200) is False


def test_drift_constants_have_expected_values():
    assert sr.DRIFT_WINDOW == 150
    assert sr.MIN_RECENT == 100
    assert sr.DRIFT_MARGIN == 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_scheduled_retrain.py -k should_retrain -q`
Expected: FAIL — `AttributeError: module 'scheduled_retrain' has no attribute 'should_retrain'` (and no `DRIFT_WINDOW`).

- [ ] **Step 3: Implement**

In `backend_ml/scheduled_retrain.py`, add the constants beside `TOLERANCE`/`SEASON_START` (after line 19):

```python
DRIFT_WINDOW = 150            # most-recent games scored to measure current drift
MIN_RECENT = 100             # below this many recent games -> retrain-anyway fallback
DRIFT_MARGIN = 0.02          # recent Brier worse than baseline by more than this -> drift
```

Add the function after `should_deploy` (after line 42):

```python
def should_retrain(recent_brier, baseline_brier, n_recent,
                   min_recent=MIN_RECENT, margin=DRIFT_MARGIN):
    """Retrain on any uncertainty; skip only on a confident no-drift signal.

    True (retrain) if too little recent data, an unmeasurable recent Brier, no
    stored baseline, or recent Brier worse than baseline by more than `margin`.
    """
    if n_recent < min_recent:
        return True
    if recent_brier is None or baseline_brier is None:
        return True
    return recent_brier > baseline_brier + margin
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_scheduled_retrain.py -k should_retrain -q`
Expected: PASS (9 `should_retrain` cases + boundary + constants).

- [ ] **Step 5: Commit**

```bash
cd backend_ml
git add scheduled_retrain.py test_scheduled_retrain.py
git commit -m "feat: add should_retrain drift gate decision and tuning constants"
```

---

### Task 4: `measure_drift` — score the deployed model's recent Brier

Load the deployed artifacts, score the tail window via the existing leakage-safe recompute path, return `(recent_brier, n_recent)` — or `(None, 0)` on any error.

**Files:**
- Modify: `backend_ml/scheduled_retrain.py` (add `measure_drift` after `should_retrain`)
- Test: `backend_ml/test_scheduled_retrain.py`

**Interfaces:**
- Consumes: `DRIFT_WINDOW` (Task 3); `signal_research.dataset.build_recompute_dataset` and `signal_research.calibration.brier_score` (existing).
- Produces: `measure_drift(df, live_dir=".", window=DRIFT_WINDOW) -> tuple[float | None, int]` — loads `xgboost_nba_model.pkl`, `ridge_nba_model.pkl`, `feature_scaler.pkl`, `ensemble_weights.json` from `live_dir`; scores the most-recent `window` rows of `df` (sorted by `GAME_DATE_H`); returns `(brier, n_recent)`. Any exception → `(None, 0)`. Consumed by `main()` in Task 5.

- [ ] **Step 1: Write the failing tests**

Add to `backend_ml/test_scheduled_retrain.py`. Put these imports/fakes near the top of the file (after the existing imports); the fake classes MUST be module-level so `joblib.dump` can pickle them:

```python
import joblib
import numpy as np
import pandas as pd
from signal_research import dataset as _ds


class _FakeXGB:
    """predict_proba -> P(home win) = 0.7 for every row."""
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


class _FakeRidge:
    def decision_function(self, X):
        return np.zeros(len(X))            # sigmoid(0) = 0.5


class _FakeScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def _drift_games(n, all_home_wins=True):
    """A minimal recompute-ready frame: FEATURES + id/date/outcome columns."""
    row = {f: 1.0 for f in _ds.FEATURES}
    df = pd.DataFrame([row] * n)
    df["GAME_ID"] = [f"g{i}" for i in range(n)]
    df["GAME_DATE_H"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(range(n), unit="D")
    df["TEAM_ID_H"] = 100
    df["TEAM_ID_A"] = 200
    df["HOME_WIN"] = 1 if all_home_wins else 0
    return df


def _write_live_artifacts(live_dir, xgb_w=0.5, ridge_w=0.5):
    joblib.dump(_FakeXGB(), str(live_dir / "xgboost_nba_model.pkl"))
    joblib.dump(_FakeRidge(), str(live_dir / "ridge_nba_model.pkl"))
    joblib.dump(_FakeScaler(), str(live_dir / "feature_scaler.pkl"))
    (live_dir / "ensemble_weights.json").write_text(
        json.dumps({"xgb_weight": xgb_w, "ridge_weight": ridge_w,
                    "test_accuracy": 0.66, "test_brier": 0.2}))


# --- measure_drift ------------------------------------------------------------

def test_measure_drift_returns_brier_and_count(tmp_path):
    _write_live_artifacts(tmp_path)
    df = _drift_games(200, all_home_wins=True)
    brier, n = sr.measure_drift(df, live_dir=str(tmp_path), window=150)
    # p = 0.5*0.7 + 0.5*0.5 = 0.6; outcome = 1 -> brier = (0.6-1)^2 = 0.16
    assert n == 150
    assert brier == pytest.approx(0.16, abs=1e-9)


def test_measure_drift_window_larger_than_rows_uses_all(tmp_path):
    _write_live_artifacts(tmp_path)
    df = _drift_games(5)
    brier, n = sr.measure_drift(df, live_dir=str(tmp_path), window=150)
    assert n == 5
    assert brier == pytest.approx(0.16, abs=1e-9)


def test_measure_drift_missing_artifacts_returns_none(tmp_path):
    # empty live_dir -> joblib.load raises -> caught -> (None, 0)
    df = _drift_games(10)
    assert sr.measure_drift(df, live_dir=str(tmp_path), window=150) == (None, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_scheduled_retrain.py -k measure_drift -q`
Expected: FAIL — `AttributeError: module 'scheduled_retrain' has no attribute 'measure_drift'`.

- [ ] **Step 3: Implement**

In `backend_ml/scheduled_retrain.py`, add after `should_retrain`:

```python
def measure_drift(df, live_dir=".", window=DRIFT_WINDOW):
    """Recent-window Brier of the deployed model; (None, 0) on any error.

    Loads the deployed artifacts from live_dir, scores the most-recent
    `window` games in df with the leakage-safe recompute path, and returns
    (recent_brier, n_recent). Any load/score failure yields (None, 0) so the
    caller falls back to retraining rather than silently skipping.
    """
    try:
        import joblib
        from signal_research import dataset, calibration
        xgb = joblib.load(os.path.join(live_dir, "xgboost_nba_model.pkl"))
        ridge = joblib.load(os.path.join(live_dir, "ridge_nba_model.pkl"))
        scaler = joblib.load(os.path.join(live_dir, "feature_scaler.pkl"))
        with open(os.path.join(live_dir, "ensemble_weights.json")) as f:
            weights = json.load(f)
        recent = df.sort_values("GAME_DATE_H").tail(window)
        scored = dataset.build_recompute_dataset(recent, xgb, ridge, scaler, weights)
        brier = calibration.brier_score(scored["p_model"], scored["outcome"])
        return brier, len(scored)
    except Exception:
        return None, 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_scheduled_retrain.py -k measure_drift -q`
Expected: PASS (3 cases).

- [ ] **Step 5: Commit**

```bash
cd backend_ml
git add scheduled_retrain.py test_scheduled_retrain.py
git commit -m "feat: measure deployed-model recent-window Brier for drift gating"
```

---

### Task 5: Wire the gate into `main()` and extend `log_run`

Insert the gate in front of the existing train→validate→swap flow: refresh the cache once, pop `FORCE_REFRESH` to avoid a double-scrape, read the baseline, measure drift, and skip if confidently no-drift. Extend `log_run` with optional Brier context (appended only when provided, so existing lines are byte-for-byte unchanged).

**Files:**
- Modify: `backend_ml/scheduled_retrain.py` (extend `log_run` ~line 68-78; rewrite the body of `main` ~line 81-118)
- Test: `backend_ml/test_scheduled_retrain.py` (update the `fake_trainer` fixture; add gate tests)

**Interfaces:**
- Consumes: `read_baseline_brier` (Task 2), `should_retrain` (Task 3), `measure_drift` (Task 4), `data_engine.load_or_build_training_dataset` (existing).
- Produces: the gated nightly run; new log outcome `skipped: no drift`.

- [ ] **Step 1: Write/adjust the failing tests**

First, update the existing `fake_trainer` fixture in `backend_ml/test_scheduled_retrain.py` so `main()` can call the cache loader without scraping nba_api. Replace the fixture body's opening (the `monkeypatch.chdir(tmp_path)` line stays) by adding a stub right after it:

```python
@pytest.fixture
def fake_trainer(tmp_path, monkeypatch):
    """Run main() in an isolated cwd with a fake trainer that writes 4 artifacts."""
    monkeypatch.chdir(tmp_path)

    # main() now refreshes the cache up front; stub it so tests never scrape.
    import data_engine
    monkeypatch.setattr(data_engine, "load_or_build_training_dataset",
                        lambda *a, **k: pd.DataFrame())

    state = {"called": 0, "new_accuracy": 0.70, "succeed": True}
    # ... rest of the existing fixture unchanged ...
```

(The rest of the fixture — the `_train` closure, the `train_model` monkeypatch, `return state` — is unchanged.)

Then add the gate tests. These stub `sr.measure_drift` directly so they exercise the decision wiring without artifacts:

```python
# --- main: drift gate ---------------------------------------------------------

def _write_live_weights(tmp_path, test_brier):
    (tmp_path / "ensemble_weights.json").write_text(
        json.dumps({"test_accuracy": 0.66, "test_brier": test_brier}))


def test_main_skips_when_no_drift(fake_trainer, tmp_path, monkeypatch):
    _write_live_weights(tmp_path, test_brier=0.20)
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (0.20, 200))
    assert sr.main(today=IN_SEASON) == 0
    assert fake_trainer["called"] == 0                       # trainer NOT run
    log = (tmp_path / sr.LOG_PATH).read_text()
    assert "skipped: no drift" in log
    assert "new_brier=0.2000 baseline_brier=0.2000" in log


def test_main_retrains_when_drifted(fake_trainer, tmp_path, monkeypatch):
    _write_live_weights(tmp_path, test_brier=0.20)
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (0.30, 200))
    assert sr.main(today=IN_SEASON) == 0
    assert fake_trainer["called"] == 1                       # trainer ran


def test_main_retrains_on_low_recent_data(fake_trainer, tmp_path, monkeypatch):
    _write_live_weights(tmp_path, test_brier=0.20)
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (0.10, 50))  # n < MIN_RECENT
    assert sr.main(today=IN_SEASON) == 0
    assert fake_trainer["called"] == 1


def test_main_retrains_on_measure_error(fake_trainer, tmp_path, monkeypatch):
    _write_live_weights(tmp_path, test_brier=0.20)
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (None, 0))
    assert sr.main(today=IN_SEASON) == 0
    assert fake_trainer["called"] == 1                       # fallback -> trainer ran


def test_main_pops_force_refresh_before_training(fake_trainer, tmp_path, monkeypatch):
    monkeypatch.setenv("FORCE_REFRESH", "1")
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (0.30, 200))  # force drift
    sr.main(today=IN_SEASON)
    assert "FORCE_REFRESH" not in os.environ                 # popped by main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_scheduled_retrain.py -q`
Expected: FAIL — the new gate tests fail (no `skipped: no drift` path yet; `main()` does not yet pop `FORCE_REFRESH` or call `measure_drift`). Some previously-passing `main` tests may also now fail because `main()` does not yet load the cache — that is expected and fixed in Step 3.

- [ ] **Step 3: Implement**

In `backend_ml/scheduled_retrain.py`, first extend `log_run` (replace the whole function, lines 68-78) so Brier context is appended only when provided:

```python
def log_run(outcome, new_acc=None, current_acc=None,
            new_brier=None, baseline_brier=None, log_path=LOG_PATH):
    """Append one run-summary line; fall back to stderr if the log is unwritable.

    The new_brier/baseline_brier suffix is appended only when at least one is
    provided, so existing 'deployed'/'rejected'/'skipped: offseason' lines keep
    their original format.
    """
    def fmt(v):
        return "none" if v is None else f"{v:.4f}"
    line = (f"{datetime.now().isoformat(timespec='seconds')} {outcome} "
            f"new_acc={fmt(new_acc)} current_acc={fmt(current_acc)}")
    if new_brier is not None or baseline_brier is not None:
        line += f" new_brier={fmt(new_brier)} baseline_brier={fmt(baseline_brier)}"
    line += "\n"
    try:
        with open(log_path, "a") as f:
            f.write(line)
    except OSError:
        print(line, end="", file=sys.stderr)
```

Then replace the body of `main` (lines 81-118) so the gate runs inside the existing `try/except/finally`:

```python
def main(today=None):
    """Run one scheduled-retrain cycle. Returns a process exit code."""
    from datetime import date
    today = today or date.today()

    if not in_season(today):
        log_run("skipped: offseason")
        return 0

    temp_dir = None
    try:
        # Refresh the cache once (honors FORCE_REFRESH), then pop FORCE_REFRESH
        # so the later train step reuses this fresh cache instead of re-scraping.
        import data_engine
        df = data_engine.load_or_build_training_dataset()
        os.environ.pop("FORCE_REFRESH", None)

        baseline_brier = read_baseline_brier("ensemble_weights.json")
        recent_brier, n_recent = measure_drift(df)
        if not should_retrain(recent_brier, baseline_brier, n_recent):
            log_run("skipped: no drift",
                    new_brier=recent_brier, baseline_brier=baseline_brier)
            return 0

        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="nba_retrain_")

        # Lazy import: keeps module import light and lets tests patch it.
        import train_model
        if not train_model.train_and_optimize_model(output_dir=temp_dir):
            log_run("failed: train_and_optimize_model returned False")
            return 1

        new_acc = read_test_accuracy(os.path.join(temp_dir, "ensemble_weights.json"))
        if new_acc is None:
            log_run("failed: no test_accuracy in new ensemble_weights.json")
            return 1
        current_acc = read_test_accuracy("ensemble_weights.json")

        if should_deploy(new_acc, current_acc):
            deploy_artifacts(temp_dir)
            log_run("deployed", new_acc, current_acc,
                    new_brier=recent_brier, baseline_brier=baseline_brier)
        else:
            log_run("rejected", new_acc, current_acc,
                    new_brier=recent_brier, baseline_brier=baseline_brier)
        return 0
    except Exception as e:
        log_run(f"failed: {e!r}")
        return 1
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
```

- [ ] **Step 4: Run the full suite to verify everything passes**

Run: `cd backend_ml && ../.venv/bin/python -m pytest test_scheduled_retrain.py test_train_model_save.py -q`
Expected: PASS — all tests (the original 25 in `test_scheduled_retrain.py`, the new gate/measure/decision tests, and both `test_train_model_save.py` groups). Note the pre-existing `main` tests (`test_main_deploys_when_no_prior_model`, `test_main_rejects_...`, the failure tests) still pass: their live dir has no `.pkl`s, so `measure_drift` returns `(None, 0)` → `should_retrain` is `True` → the existing train/deploy/reject path runs exactly as before, and the `deployed`/`rejected` substrings still match because the appended Brier context (`none`/`none`) comes after them.

- [ ] **Step 5: Run the broader backend_ml suite for regressions**

Run: `cd backend_ml && ../.venv/bin/python -m pytest -q`
Expected: PASS (or the same pre-existing pass/skip state as before this branch — no *new* failures). If any test unrelated to this change was already failing on `main`, note it but do not attempt to fix it here.

- [ ] **Step 6: Commit**

```bash
cd backend_ml
git add scheduled_retrain.py test_scheduled_retrain.py
git commit -m "feat: gate nightly retrain on measured drift"
```

---

## Self-Review Notes (for the executor)

- **Spec coverage:** Task 1 → baseline recording (`test_brier`); Task 2 → `read_baseline_brier`; Task 3 → `should_retrain` + constants; Task 4 → `measure_drift`; Task 5 → `main()` orchestration, `FORCE_REFRESH` pop, `skipped: no drift` log, drift context on deploy/reject. The spec's "in-sample night" and "rejected model persists baseline" behaviors are emergent from this wiring, not separate code.
- **Free-tier / non-goals:** unchanged plist and `FORCE_REFRESH=1` are explicitly preserved (Global Constraints + Task 5). No new external API calls are introduced — `measure_drift` is pure local scoring over the already-refreshed cache.
- **No `git add -A`:** every Commit step stages only exact paths, none of which is `ensemble_weights.json`, `*.pkl`, or `*.csv`.
