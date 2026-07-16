# Scheduled Retrain Job — Design

**Date:** 2026-07-16
**Status:** Approved (brainstormed + user-approved in session)
**Depends on:** cache-age check + `FORCE_REFRESH` flag in `data_engine.load_or_build_training_dataset()` (built 2026-07-16, uncommitted at time of writing)

## Purpose

The NBA model's training path goes stale silently: `train_model.py` is only ever
run by hand, and once `nba_training_cache.csv` exists it is reused indefinitely.
The serving path (`predict.py`) always fetches fresh features but scores them
with whatever `.pkl` sits on disk — in-season, that can be a model that has
learned nothing from recent weeks of games.

This feature adds an unattended nightly retrain, run locally via launchd, with a
validation guardrail so a bad automated run can never silently degrade the
live-serving model.

## Scope decisions (user-approved)

| Decision | Choice |
|---|---|
| Scheduler host | Local (launchd on the user's Mac) — no cloud infra, matches repo's local-first posture |
| Cadence | Daily, 4:00 AM (after games finish) |
| Guardrail | Validate before swapping: retrain into temp dir, deploy only if new held-out accuracy ≥ current − TOLERANCE |
| Season awareness | Skip outside a configured season window (approx. Oct 1 – Jun 30) |
| Failure visibility | Log file only (`retrain_log.txt`, one line per run); no notification infra |
| Trading posture | Paper-only unchanged; this touches only Layer 1 training artifacts |

Items #4 (drift-gated retraining) and #5 (rolling-window training) are
**separate follow-up designs**, deliberately not folded in here.

## Architecture

New pieces:

1. **`backend_ml/scheduled_retrain.py`** — orchestrator, invoked by launchd.
   Single process, whole-run try/except; a crash never leaves half-written
   state.
2. **`train_model.py` refactor** — `train_and_optimize_model()` gains an
   `output_dir` parameter (default `"."`). A bare manual
   `python train_model.py` run stays byte-identical to today.
3. **`backend_ml/scripts/com.benchwarmer.nba-retrain.plist`** — launchd job:
   `StartCalendarInterval` at 04:00 daily, runs
   `.venv/bin/python backend_ml/scheduled_retrain.py` with `backend_ml/` as
   working directory, `StandardErrorPath` →
   `backend_ml/scheduled_retrain.stderr.log`.
4. **`backend_ml/retrain_log.txt`** — gitignored (alongside `.pkl`/`.csv`
   rules in `backend_ml/.gitignore`), one appended line per run:
   `<timestamp> <outcome> new_acc=<x> current_acc=<y>` where outcome ∈
   {`deployed`, `rejected`, `skipped: offseason`, `failed: <exc>`}.

Constants in `scheduled_retrain.py`:

- `TOLERANCE = 0.01` — allow up to a 1-point held-out-accuracy dip before
  rejecting (accuracy on the ~15% chronological test split is noisy; observed
  3-way ties during the 2026-07-16 ensemble re-tune).
- `SEASON_START = (10, 1)`, `SEASON_END = (6, 30)` — month/day tuples; the
  window wraps the year boundary.

## Flow

```
launchd (04:00 daily)
   │
   ▼
scheduled_retrain.py
   1. in_season(today)? ── No ─▶ log "skipped: offseason", exit 0
   2. temp_dir = tempfile.mkdtemp()
   3. train_and_optimize_model(output_dir=temp_dir)
        └─ internally: load_or_build_training_dataset()
           (cache ≤3d old → read CSV; else nba_api refetch ~2 min)
   4. new_acc = temp_dir/ensemble_weights.json["test_accuracy"]
      cur_acc = live ensemble_weights.json["test_accuracy"] (None if absent)
   5. should_deploy(new_acc, cur_acc, TOLERANCE)?
        Yes → copy-then-rename temp files over the 4 live paths, log "deployed"
        No  → discard temp_dir, log "rejected"
   6. cleanup temp_dir
```

Downstream (`predict.py`, `publish_fair_values.py`) is untouched — those
already read whatever sits at the live paths on every call. The scheduler's
only job is deciding what sits there.

## Error handling

- **Whole-script try/except:** any exception (nba_api failure, training crash,
  disk error) → log `failed: <exception>`, exit non-zero. The live `.pkl`/
  `.json` files are never touched on a failure path — the deploy step only
  runs after training succeeds and all 4 files exist in the temp dir.
- **Copy-then-rename:** new files land at `<live_path>.new`, then each is
  `os.rename`'d over the live path in a tight loop. No live file is ever
  half-written. The 4-file swap is not atomic as a unit — accepted narrow
  edge case (the rename loop takes microseconds; `predict.py` reads all 4 in
  one call).
- **launchd stderr capture:** the plist's `StandardErrorPath` catches failures
  that occur before the script's own logging initializes (bad venv path,
  import error).
- **Logging failure fallback:** the log write is wrapped in its own
  try/except, falling back to stderr (which launchd captures).
- **Manual-run invariant:** the `output_dir` refactor must leave
  `python train_model.py` behavior identical to today (writes to cwd) —
  covered by a regression test.

## Testing

All pure logic is unit-tested; nothing in tests touches network, real training,
or real repo paths (injection style matches `market_capture.py`/`report.py`
tests):

- `in_season(date)` — parametrized: mid-season, mid-offseason, Dec→Jan wrap
  boundary, exact start/end days.
- `should_deploy(new_acc, current_acc, tolerance)` — better / worse /
  within-tolerance / no-prior-model.
- Deploy step — against `tmp_path`; verifies live files untouched when a
  fault is injected mid-copy.
- `main()` orchestration — monkeypatched `train_and_optimize_model` +
  `load_or_build_training_dataset`; asserts log lines and exit codes per
  outcome.
- `train_model.py` `output_dir` default — regression test: no-arg call writes
  to cwd exactly as before.

## Deliverables

- `backend_ml/scheduled_retrain.py` (orchestrator + `in_season` +
  `should_deploy` + deploy step)
- `train_model.py` — `output_dir` parameter, default-unchanged
- `backend_ml/scripts/com.benchwarmer.nba-retrain.plist` + a short
  install note (one `launchctl load` command) in the plist header comment
- `backend_ml/.gitignore` — add `retrain_log.txt`,
  `scheduled_retrain.stderr.log`
- Tests as above

## Security / safety posture

- No new secrets: the retrain path uses `nba_api` (no key) and optionally
  Supabase env keys already handled by `data_engine.py`'s dotenv loading.
- Paper-only posture unchanged: nothing here touches the C++ engine,
  `fair_values.json` publishing, or any Kalshi credential.
- launchd plist contains no secrets — only paths.
