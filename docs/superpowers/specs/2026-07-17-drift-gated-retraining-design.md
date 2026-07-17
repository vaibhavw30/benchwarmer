# Drift-Gated Retraining — Design (#4)

**Date:** 2026-07-17
**Status:** Approved, ready for implementation planning
**Builds on:** [2026-07-16-scheduled-retrain-job-design.md](2026-07-16-scheduled-retrain-job-design.md) (the nightly train→validate→swap job, merged at `50e6be4`)

## Problem

The nightly retrain job (`scheduled_retrain.py`) currently runs a full
GridSearchCV train + validate-before-swap **every in-season night**, whether
or not the deployed model has actually degraded. Most nights the model is
fine and the retrain is wasted compute. We want to spend a retrain only when
the model has measurably drifted.

## Non-goals / scope boundaries

- **Not** reducing `nba_api` pulls. `nba_api` (`stats.nba.com`) has no key,
  no billing, and no hard quota — it is only politeness-rate-limited (the
  `time.sleep(1)` between season fetches). The cache is refreshed nightly
  (`FORCE_REFRESH=1`) regardless of the gate, because measuring drift needs
  fresh data. Drift-gating saves **training compute**, not data pulls. The
  Odds API free-tier quota is consumed only by the serving/publish path and
  is untouched by retraining.
- **Not** rolling-window training — that is #5, its own spec.
- **Not** changing the validate-before-swap logic — the gate sits *in front
  of* the existing flow; the flow itself is unchanged.

## Design decisions (all confirmed in brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Drift data source | Recompute on recent cache | Self-contained; no Supabase coupling; as-of features → no leakage |
| Drift trigger | Recent Brier vs stored baseline + margin | Self-calibrating per model; single proper scoring rule; no magic absolute number |
| Baseline storage | `test_brier` field in `ensemble_weights.json` | Auto-updates on every deploy; no separate state file |
| Low-data fallback | Retrain anyway | Gate can only ever *reduce* retrains, never regress below current behavior |
| Refresh cadence | Keep `FORCE_REFRESH=1` | Drift measured on truly-fresh data (includes last night's games) |
| Error handling | Any measurement error → retrain-anyway | An unmeasurable drift signal must never cause a silent skip |

### Tuning constants (named constants in `scheduled_retrain.py`)

```python
DRIFT_WINDOW = 150     # most-recent games scored to measure current drift (~2–3 weeks)
MIN_RECENT   = 100     # below this many recent games → retrain-anyway fallback
DRIFT_MARGIN = 0.02    # recent Brier worse than baseline by more than this → drift
```

These are starting points, tunable once real `retrain_log.txt` output exists.

## Architecture

The gate lives inside `scheduled_retrain.main()`, in front of the existing
train→validate→swap flow. The existing flow is untouched; drift-gating only
decides whether to enter it.

### Critical data-flow ordering (avoid double-scrape)

`FORCE_REFRESH=1` (from the plist) causes
`load_or_build_training_dataset` to rebuild the cache. Both the drift check
**and** the training step need that fresh cache. To avoid scraping all 11
seasons twice in one night, `main()`:

1. Refreshes the cache **once** up front: `df = load_or_build_training_dataset()`
   (honors `FORCE_REFRESH`, pulls last night's games).
2. `os.environ.pop("FORCE_REFRESH", None)` so the later `train_model` call
   **reuses** that just-built cache (age 0–1d ≤ 3d threshold) instead of
   re-scraping.

## Components

### `train_model.py` — record the baseline

Add `test_brier` to `weights_config`, computed on the same held-out ensemble
probabilities already scored for `ensemble_logloss`:

```python
from sklearn.metrics import brier_score_loss
# ... beside the existing ensemble evaluation ...
ensemble_brier = brier_score_loss(y_test, ensemble_probs_best)
weights_config = {
    "xgb_weight": xgb_w,
    "ridge_weight": ridge_w,
    "test_accuracy": float(best_ensemble_acc),
    "test_brier": float(ensemble_brier),   # NEW: drift baseline
    "train_date": str(pd.Timestamp.now()),
}
```

Backward-compat: a model trained before this change has no `test_brier`;
`read_baseline_brier` returns `None` for it, and a `None` baseline forces a
retrain (which then writes the field). Self-healing on first run.

### `scheduled_retrain.py` — new pieces

**`read_baseline_brier(weights_path) -> float | None`**
Mirror of the existing `read_test_accuracy`: returns `test_brier` from an
`ensemble_weights.json`, or `None` if absent/unreadable.

**`measure_drift(df, live_dir=".", window=DRIFT_WINDOW) -> tuple[float | None, int]`**
- Load the deployed `xgboost_nba_model.pkl`, `ridge_nba_model.pkl`,
  `feature_scaler.pkl`, and `ensemble_weights.json` from `live_dir`.
- Take the tail `window` rows of `df` sorted by `GAME_DATE_H`.
- Score them via `dataset.build_recompute_dataset(...)`.
- Compute Brier via `calibration.brier_score(p_model, outcome)`.
- Return `(recent_brier, n_recent)`. On any load/score error, return
  `(None, 0)` so the caller hits the retrain-anyway fallback.

**`should_retrain(recent_brier, baseline_brier, n_recent, min_recent=MIN_RECENT, margin=DRIFT_MARGIN) -> bool`**
Pure decision. Returns `True` (retrain) if **any** of:
- `n_recent < min_recent` (low-data fallback), or
- `recent_brier is None` (measurement error fallback), or
- `baseline_brier is None` (no baseline yet), or
- `recent_brier > baseline_brier + margin` (genuine drift).

Returns `False` (skip) **only** when we confidently see no drift.

**`main()`** — new orchestration in front of the existing flow:

```
if not in_season(today): log "skipped: offseason"; return 0
df = load_or_build_training_dataset()          # fresh cache
os.environ.pop("FORCE_REFRESH", None)          # avoid double-scrape
baseline = read_baseline_brier("ensemble_weights.json")
recent_brier, n = measure_drift(df)
if not should_retrain(recent_brier, baseline, n):
    log "skipped: no drift" (recent_brier, baseline); return 0
# --- existing train→temp, validate, deploy-or-reject, log path, unchanged ---
```

## Data flow

```
launchd 04:00 (FORCE_REFRESH=1)
  └─ main()
       ├─ in_season? no → log "skipped: offseason"; exit 0
       ├─ df = load_or_build_training_dataset()   # fresh cache, last night's games
       ├─ os.environ.pop("FORCE_REFRESH")          # avoid double-scrape
       ├─ baseline = read_baseline_brier("ensemble_weights.json")
       ├─ recent_brier, n = measure_drift(df)
       ├─ should_retrain(recent_brier, baseline, n)?
       │    ├─ no  → log "skipped: no drift"; exit 0
       │    └─ yes → train→temp → validate → deploy-or-reject → log; exit 0   [existing]
       └─ finally: rmtree(temp_dir)                [existing]
```

The newly-deployed model's `ensemble_weights.json` carries its own fresh
`test_brier`, so the baseline auto-updates on every deploy — no separate
state file.

### Note on the in-sample night

The night immediately after a deploy, the recent window is in the new
model's training set → Brier looks excellent → no drift → skip. This is
**correct**: we just retrained, no need to again. As nights pass without a
retrain, recent games accumulate out-of-sample; if drift is real, Brier
degrades and eventually trips the gate. If a retrained model is *rejected*
by validate-before-swap, the old baseline persists and the gate keeps
firing nightly — degrading gracefully to today's always-retrain behavior,
with `retrain_log.txt` showing repeated `drift → rejected` as a signal that
something structural is wrong.

## Error handling

- `measure_drift` wraps all I/O and scoring; any failure → `(None, 0)` →
  `should_retrain` returns `True`. A broken/missing artifact can never turn
  into a silent skip that strands a bad model.
- All new logic runs inside `main()`'s existing `try/except`; an unexpected
  exception still logs `failed: <repr>` and returns 1, temp dir cleaned in
  `finally`.
- New/extended log outcomes:
  - `skipped: no drift new_brier=… baseline_brier=…`
  - existing `deployed` / `rejected` lines gain the drift context so
    `retrain_log.txt` explains *why* each night acted.

## Testing

Extends the existing `test_scheduled_retrain.py` / `test_train_model_save.py`
suites. No `.pkl` files required — models are faked with tiny objects
exposing `predict_proba` / `decision_function`, matching the existing
`dataset.py` test fixtures.

**Pure functions:**
- `should_retrain`: drift, no-drift, low-data (`n_recent < MIN_RECENT`),
  measurement-error (`recent_brier is None`), no-baseline (`baseline is None`),
  exactly-at-margin boundary.
- `read_baseline_brier`: present, missing key, unreadable file, non-numeric.

**`measure_drift`:** in-memory `df` + fake models → returns finite Brier and
correct `n_recent`; window larger than available rows → uses all rows;
artifact-load failure → `(None, 0)`.

**`main()` (monkeypatched):**
- no-drift → asserts `train_model.train_and_optimize_model` is **not** called
  and log says `skipped: no drift`.
- drift → asserts training runs (existing deploy/reject paths still pass).
- low-data → training runs (fallback).
- measure-error → training runs (fallback).
- asserts `FORCE_REFRESH` is popped before the training step.

**`train_model`:** `test_brier` is present and numeric in the written
`ensemble_weights.json`; existing `save_artifacts` / default-cwd regression
tests still pass.

## Rollout

1. Implement on `feature/drift-gated-retrain` via subagent-driven-development.
2. Merge to main; the existing plist and launchd job need **no change**
   (same entry point, same `FORCE_REFRESH=1`).
3. First nightly run with no `test_brier` in the live weights → `None`
   baseline → retrains once → writes `test_brier` → gate active from then on.
