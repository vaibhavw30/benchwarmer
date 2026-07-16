# Scheduled Retrain Job Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nightly launchd-driven retrain of the NBA model with a validate-before-swap guardrail, so a bad unattended run can never silently degrade the live-serving `.pkl`/`ensemble_weights.json`.

**Architecture:** A new `backend_ml/scheduled_retrain.py` orchestrator trains into a temp dir via `train_and_optimize_model(output_dir=...)` (new parameter), compares the new held-out `test_accuracy` against the currently-deployed one, and copy-then-renames the 4 artifacts over the live paths only if the new model is within tolerance. A launchd plist runs it at 04:00 daily; a season-window check no-ops it in the offseason.

**Tech Stack:** Python 3 (repo `.venv`), pytest, launchd (macOS). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-16-scheduled-retrain-job-design.md`

## Global Constraints

- Work on a feature branch (e.g. `feature/scheduled-retrain`) off `main`, not on `main` directly.
- Manual `python train_model.py` behavior must remain byte-identical: default `output_dir="."`, same 4 filenames written to cwd.
- The 4 artifacts, exactly: `xgboost_nba_model.pkl`, `ridge_nba_model.pkl`, `feature_scaler.pkl`, `ensemble_weights.json`.
- `TOLERANCE = 0.01`; `SEASON_START = (10, 1)`; `SEASON_END = (6, 30)` (window wraps the year boundary).
- Live artifacts are NEVER touched on any failure path; deploy stages all copies to `<name>.new` before any rename.
- Log line format: `<ISO timestamp> <outcome> new_acc=<x|none> current_acc=<y|none>`, outcome ∈ {`deployed`, `rejected`, `skipped: offseason`, `failed: <exc>`}.
- NEVER commit: `backend_ml/ensemble_weights.json` (tracked file, has uncommitted local re-tune edits that are out of scope), any `.pkl`, any `.csv`. Use exact-path `git add` only — no `git add -A` / `git add .`.
- Tests never touch the network, real training, or real repo paths — monkeypatch + `tmp_path`, same style as `backend_ml/signal_research/tests/`.
- All Python runs use the repo venv: `.venv/bin/python`, `.venv/bin/pytest` from the repo root (`cd backend_ml` first where noted — `conftest.py` there puts `backend_ml/` on `sys.path`).

---

### Task 1: Lock in the cache-freshness foundation (tests + commit of pending #1/#2 work)

The working tree already contains uncommitted, smoke-tested changes: `data_engine.load_or_build_training_dataset()` (cache-age check + `FORCE_REFRESH=1`) and its adoption in `train_model.py`/`backtest.py`. This task adds real unit tests for that function and commits the whole foundation.

**Files:**
- Test (create): `backend_ml/test_data_cache.py`
- Commit (already modified, do not edit further): `backend_ml/data_engine.py`, `backend_ml/train_model.py`, `backend_ml/backtest.py`

**Interfaces:**
- Produces: `data_engine.load_or_build_training_dataset(cache_path="nba_training_cache.csv", max_age_days=3) -> pd.DataFrame` — committed and tested; Task 2's `train_and_optimize_model` already calls it (no-arg).

- [ ] **Step 1: Write the failing tests**

```python
# backend_ml/test_data_cache.py
"""Tests for data_engine.load_or_build_training_dataset cache-freshness logic."""
import pandas as pd
import pytest

import data_engine


@pytest.fixture
def fake_build(monkeypatch):
    calls = {"n": 0}

    def _build():
        calls["n"] += 1
        return pd.DataFrame({"GAME_DATE_H": ["2026-07-16"], "HOME_WIN": [1]})

    monkeypatch.setattr(data_engine, "build_training_dataset", _build)
    return calls


def _write_cache(path, newest_game_date):
    pd.DataFrame({"GAME_DATE_H": [newest_game_date]}).to_csv(path, index=False)


def test_fresh_cache_is_read_without_rebuild(tmp_path, fake_build):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, pd.Timestamp.now().strftime("%Y-%m-%d"))
    df = data_engine.load_or_build_training_dataset(cache_path=str(cache), max_age_days=3)
    assert fake_build["n"] == 0
    assert not df.empty


def test_stale_cache_triggers_rebuild(tmp_path, fake_build):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, "2026-04-12")  # months older than any sane max_age
    data_engine.load_or_build_training_dataset(cache_path=str(cache), max_age_days=3)
    assert fake_build["n"] == 1


def test_missing_cache_triggers_rebuild(tmp_path, fake_build):
    data_engine.load_or_build_training_dataset(
        cache_path=str(tmp_path / "absent.csv"), max_age_days=3)
    assert fake_build["n"] == 1


def test_force_refresh_rebuilds_even_when_fresh(tmp_path, fake_build, monkeypatch):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, pd.Timestamp.now().strftime("%Y-%m-%d"))
    monkeypatch.setenv("FORCE_REFRESH", "1")
    data_engine.load_or_build_training_dataset(cache_path=str(cache), max_age_days=3)
    assert fake_build["n"] == 1


def test_cache_exactly_at_threshold_is_still_fresh(tmp_path, fake_build):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, (pd.Timestamp.now() - pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
    data_engine.load_or_build_training_dataset(cache_path=str(cache), max_age_days=3)
    assert fake_build["n"] == 0
```

- [ ] **Step 2: Run tests — they should PASS immediately** (the implementation already exists in the working tree; these are lock-in tests, not TDD-red tests)

Run: `cd backend_ml && ../.venv/bin/pytest test_data_cache.py -v`
Expected: 5 passed. If any fail, the implementation in `data_engine.py` has a bug — fix `data_engine.py`, not the test expectations.

- [ ] **Step 3: Commit the foundation (exact paths only)**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git add backend_ml/data_engine.py backend_ml/train_model.py backend_ml/backtest.py backend_ml/test_data_cache.py
git status --short   # MUST NOT show ensemble_weights.json as staged
git commit -m "Add cache-age check + FORCE_REFRESH to training data load

load_or_build_training_dataset() rebuilds when the newest cached
GAME_DATE_H exceeds max_age_days (default 3) or FORCE_REFRESH=1;
train_model.py and backtest.py now share it instead of duplicating
the exists()-check."
```

---

### Task 2: `output_dir` parameter for `train_model.py`

**Files:**
- Modify: `backend_ml/train_model.py` (imports at top; save section currently at lines 234–247; signature at line 17)
- Test (create): `backend_ml/test_train_model_save.py`

**Interfaces:**
- Consumes: nothing new (Task 1's function is already wired in).
- Produces: `train_model.save_artifacts(xgb_model, ridge_model, scaler, weights_config, output_dir=".") -> None` and `train_and_optimize_model(output_dir=".") -> bool`. Task 3/4's orchestrator calls `train_and_optimize_model(output_dir=temp_dir)`.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/test_train_model_save.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend_ml && ../.venv/bin/pytest test_train_model_save.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_artifacts'`

- [ ] **Step 3: Implement**

In `backend_ml/train_model.py`:

(a) Restore `os` to the imports (it was removed when the old cache check left this file):

```python
import pandas as pd
import joblib
import os
import sys
```

(b) Add `save_artifacts` just above `train_and_optimize_model` (after the `ENSEMBLE_WEIGHTS_PATH` constant):

```python
def save_artifacts(xgb_model, ridge_model, scaler, weights_config, output_dir="."):
    """Write the 4 model artifacts to output_dir (default: cwd, as always)."""
    joblib.dump(xgb_model, os.path.join(output_dir, MODEL_PATH))
    joblib.dump(ridge_model, os.path.join(output_dir, RIDGE_MODEL_PATH))
    joblib.dump(scaler, os.path.join(output_dir, SCALER_PATH))
    with open(os.path.join(output_dir, ENSEMBLE_WEIGHTS_PATH), 'w') as f:
        json.dump(weights_config, f, indent=2)
```

(c) Change the signature `def train_and_optimize_model():` → `def train_and_optimize_model(output_dir="."):`

(d) Replace the save section (currently the block below the `# 6. SAVE ALL MODELS` comment — the three `joblib.dump(...)` lines and the `with open(ENSEMBLE_WEIGHTS_PATH, ...)` block) with:

```python
    # 6. SAVE ALL MODELS
    weights_config = {
        "xgb_weight": xgb_w,
        "ridge_weight": ridge_w,
        "test_accuracy": float(best_ensemble_acc),
        "train_date": str(pd.Timestamp.now())
    }
    save_artifacts(best_model, ridge, scaler, weights_config, output_dir)
```

Keep the `print(f"\n💾 Models Saved:")` block after it unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend_ml && ../.venv/bin/pytest test_train_model_save.py test_data_cache.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git add backend_ml/train_model.py backend_ml/test_train_model_save.py
git status --short   # MUST NOT show ensemble_weights.json as staged
git commit -m "Add output_dir parameter to model artifact saving

save_artifacts() writes the 4 artifacts to a target dir; default '.'
keeps manual train_model.py runs byte-identical. Lets the scheduled
retrain job train into a temp dir before deciding to deploy."
```

---

### Task 3: `scheduled_retrain.py` pure logic (season window, deploy decision, atomic-ish swap, run log)

**Files:**
- Create: `backend_ml/scheduled_retrain.py` (logic only — `main()` comes in Task 4)
- Test (create): `backend_ml/test_scheduled_retrain.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure functions).
- Produces (Task 4's `main()` uses all of these): `in_season(today, start=SEASON_START, end=SEASON_END) -> bool`; `should_deploy(new_acc, current_acc, tolerance=TOLERANCE) -> bool`; `read_test_accuracy(weights_path) -> float | None`; `deploy_artifacts(temp_dir, live_dir=".") -> None`; `log_run(outcome, new_acc=None, current_acc=None, log_path=LOG_PATH) -> None`; constants `TOLERANCE`, `SEASON_START`, `SEASON_END`, `ARTIFACTS`, `LOG_PATH`.

- [ ] **Step 1: Write the failing tests**

```python
# backend_ml/test_scheduled_retrain.py
"""Tests for the nightly retrain orchestrator's pure logic."""
from datetime import date

import pytest

import scheduled_retrain as sr


# --- in_season -------------------------------------------------------------

@pytest.mark.parametrize("d,expected", [
    (date(2026, 1, 15), True),    # mid-season, after the year wrap
    (date(2026, 12, 25), True),   # mid-season, before the year wrap
    (date(2026, 8, 1), False),    # deep offseason
    (date(2026, 10, 1), True),    # exact season start
    (date(2026, 6, 30), True),    # exact season end
    (date(2026, 7, 1), False),    # day after season end
    (date(2026, 9, 30), False),   # day before season start
])
def test_in_season(d, expected):
    assert sr.in_season(d) is expected


# --- should_deploy -----------------------------------------------------------

@pytest.mark.parametrize("new,cur,expected", [
    (0.67, 0.66, True),    # better
    (0.66, 0.66, True),    # equal
    (0.655, 0.66, True),   # worse but within TOLERANCE (0.01)
    (0.64, 0.66, False),   # worse beyond TOLERANCE
    (0.10, None, True),    # no prior model: always deploy
])
def test_should_deploy(new, cur, expected):
    assert sr.should_deploy(new, cur) is expected


# --- read_test_accuracy -------------------------------------------------------

def test_read_test_accuracy_happy_path(tmp_path):
    p = tmp_path / "w.json"
    p.write_text('{"test_accuracy": 0.66}')
    assert sr.read_test_accuracy(str(p)) == 0.66


@pytest.mark.parametrize("content", [
    None,                        # file absent
    "not json",                  # unparseable
    '{"xgb_weight": 0.7}',       # key missing (e.g. the re-tuned local file)
])
def test_read_test_accuracy_returns_none_on_problems(tmp_path, content):
    p = tmp_path / "w.json"
    if content is not None:
        p.write_text(content)
    assert sr.read_test_accuracy(str(p)) is None


# --- deploy_artifacts ----------------------------------------------------------

def _fill(dirpath, names, tag):
    for n in names:
        (dirpath / n).write_text(f"{tag}:{n}")


def test_deploy_overwrites_all_live_artifacts(tmp_path):
    temp, live = tmp_path / "temp", tmp_path / "live"
    temp.mkdir(); live.mkdir()
    _fill(temp, sr.ARTIFACTS, "new")
    _fill(live, sr.ARTIFACTS, "old")
    sr.deploy_artifacts(str(temp), str(live))
    for n in sr.ARTIFACTS:
        assert (live / n).read_text() == f"new:{n}"
        assert not (live / (n + ".new")).exists()   # staging files cleaned up


def test_deploy_leaves_live_untouched_when_a_source_is_missing(tmp_path):
    temp, live = tmp_path / "temp", tmp_path / "live"
    temp.mkdir(); live.mkdir()
    _fill(temp, sr.ARTIFACTS[:-1], "new")           # one artifact missing
    _fill(live, sr.ARTIFACTS, "old")
    with pytest.raises(FileNotFoundError):
        sr.deploy_artifacts(str(temp), str(live))
    for n in sr.ARTIFACTS:
        assert (live / n).read_text() == f"old:{n}"  # nothing renamed


# --- log_run --------------------------------------------------------------------

def test_log_run_appends_formatted_line(tmp_path):
    log = tmp_path / "log.txt"
    sr.log_run("deployed", 0.67, 0.66, log_path=str(log))
    sr.log_run("skipped: offseason", log_path=str(log))
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("deployed new_acc=0.6700 current_acc=0.6600")
    assert lines[1].endswith("skipped: offseason new_acc=none current_acc=none")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend_ml && ../.venv/bin/pytest test_scheduled_retrain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scheduled_retrain'`

- [ ] **Step 3: Implement**

```python
# backend_ml/scheduled_retrain.py
"""Nightly retrain orchestrator, invoked by launchd (see
scripts/com.benchwarmer.nba-retrain.plist).

Trains into a temp dir and deploys over the live model artifacts only if
the new held-out accuracy is within TOLERANCE of (or better than) the
currently-deployed model's. Live files are never touched on a failure
path. One line is appended to retrain_log.txt per run.

Spec: docs/superpowers/specs/2026-07-16-scheduled-retrain-job-design.md
"""
import json
import os
import shutil
import sys
from datetime import datetime

TOLERANCE = 0.01              # allow up to a 1-point held-out-accuracy dip
SEASON_START = (10, 1)        # Oct 1
SEASON_END = (6, 30)          # Jun 30 (window wraps the year boundary)

ARTIFACTS = [
    "xgboost_nba_model.pkl",
    "ridge_nba_model.pkl",
    "feature_scaler.pkl",
    "ensemble_weights.json",
]
LOG_PATH = "retrain_log.txt"


def in_season(today, start=SEASON_START, end=SEASON_END):
    """True if today's (month, day) falls in the season window (wraps Dec->Jan)."""
    md = (today.month, today.day)
    if start <= end:
        return start <= md <= end
    return md >= start or md <= end


def should_deploy(new_acc, current_acc, tolerance=TOLERANCE):
    """Deploy when there's no prior model, or new is within tolerance of current."""
    if current_acc is None:
        return True
    return new_acc >= current_acc - tolerance


def read_test_accuracy(weights_path):
    """test_accuracy from an ensemble_weights.json; None if absent/unreadable."""
    try:
        with open(weights_path) as f:
            return float(json.load(f)["test_accuracy"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def deploy_artifacts(temp_dir, live_dir="."):
    """Copy-then-rename each artifact so no live file is ever half-written.

    All copies are staged to <name>.new first: if any source is missing or
    any copy fails, no rename has happened and the live set is untouched.
    """
    for name in ARTIFACTS:
        shutil.copy2(os.path.join(temp_dir, name),
                     os.path.join(live_dir, name + ".new"))
    for name in ARTIFACTS:
        os.rename(os.path.join(live_dir, name + ".new"),
                  os.path.join(live_dir, name))


def log_run(outcome, new_acc=None, current_acc=None, log_path=LOG_PATH):
    """Append one run-summary line; fall back to stderr if the log is unwritable."""
    def fmt(v):
        return "none" if v is None else f"{v:.4f}"
    line = (f"{datetime.now().isoformat(timespec='seconds')} {outcome} "
            f"new_acc={fmt(new_acc)} current_acc={fmt(current_acc)}\n")
    try:
        with open(log_path, "a") as f:
            f.write(line)
    except OSError:
        print(line, end="", file=sys.stderr)
```

Note: `json.JSONDecodeError` subclasses `ValueError`, so the `except` tuple in `read_test_accuracy` covers unparseable JSON.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend_ml && ../.venv/bin/pytest test_scheduled_retrain.py -v`
Expected: 19 passed (7 in_season + 5 should_deploy + 4 read_test_accuracy + 2 deploy + 1 log_run).

- [ ] **Step 5: Commit**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git add backend_ml/scheduled_retrain.py backend_ml/test_scheduled_retrain.py
git commit -m "Add scheduled-retrain pure logic

in_season (Oct 1-Jun 30, wraps year), should_deploy (tolerance 0.01,
unconditional on no prior model), read_test_accuracy, staged
copy-then-rename deploy, and one-line run logging."
```

---

### Task 4: `main()` orchestration + launchd plist + gitignore

**Files:**
- Modify: `backend_ml/scheduled_retrain.py` (append `main()` + `__main__` block)
- Create: `backend_ml/scripts/com.benchwarmer.nba-retrain.plist`
- Modify: `backend_ml/.gitignore` (add `retrain_log.txt` under the `# Logs` section; `*.log` already covers the launchd stderr/stdout captures)
- Test (modify): `backend_ml/test_scheduled_retrain.py` (append `main()` tests)

**Interfaces:**
- Consumes: everything Task 3 produced, plus Task 2's `train_and_optimize_model(output_dir=...)` (imported lazily inside `main()` so importing `scheduled_retrain` in tests doesn't drag in xgboost).
- Produces: `main(today=None) -> int` (exit code); the plist (install instructions in its header comment).

- [ ] **Step 1: Write the failing tests** (append to `backend_ml/test_scheduled_retrain.py`)

```python
# --- main -----------------------------------------------------------------------

from datetime import date as _date


@pytest.fixture
def fake_trainer(tmp_path, monkeypatch):
    """Run main() in an isolated cwd with a fake trainer that writes 4 artifacts."""
    monkeypatch.chdir(tmp_path)

    state = {"called": 0, "new_accuracy": 0.70, "succeed": True}

    def _train(output_dir="."):
        state["called"] += 1
        if not state["succeed"]:
            return False
        for n in sr.ARTIFACTS[:-1]:
            with open(os.path.join(output_dir, n), "w") as f:
                f.write("model-bytes")
        with open(os.path.join(output_dir, "ensemble_weights.json"), "w") as f:
            json.dump({"test_accuracy": state["new_accuracy"]}, f)
        return True

    import train_model
    monkeypatch.setattr(train_model, "train_and_optimize_model", _train)
    return state


IN_SEASON = _date(2026, 1, 15)
OFFSEASON = _date(2026, 8, 1)


def test_main_skips_in_offseason(fake_trainer, tmp_path):
    assert sr.main(today=OFFSEASON) == 0
    assert fake_trainer["called"] == 0
    assert "skipped: offseason" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_deploys_when_no_prior_model(fake_trainer, tmp_path):
    assert sr.main(today=IN_SEASON) == 0
    for n in sr.ARTIFACTS:
        assert (tmp_path / n).exists()
    assert "deployed new_acc=0.7000 current_acc=none" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_rejects_worse_model_and_keeps_live_files(fake_trainer, tmp_path):
    (tmp_path / "ensemble_weights.json").write_text('{"test_accuracy": 0.80}')
    (tmp_path / sr.ARTIFACTS[0]).write_text("live-model")
    fake_trainer["new_accuracy"] = 0.70    # 10 points worse: beyond tolerance
    assert sr.main(today=IN_SEASON) == 0
    assert (tmp_path / sr.ARTIFACTS[0]).read_text() == "live-model"
    assert "rejected new_acc=0.7000 current_acc=0.8000" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_logs_failure_and_exits_nonzero_when_trainer_fails(fake_trainer, tmp_path):
    fake_trainer["succeed"] = False
    assert sr.main(today=IN_SEASON) == 1
    assert "failed:" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_logs_failure_on_exception(fake_trainer, tmp_path, monkeypatch):
    import train_model

    def _boom(output_dir="."):
        raise RuntimeError("nba_api down")

    monkeypatch.setattr(train_model, "train_and_optimize_model", _boom)
    assert sr.main(today=IN_SEASON) == 1
    assert "failed: RuntimeError('nba_api down')" in (tmp_path / sr.LOG_PATH).read_text()
```

The fixture uses `os` and `json` — add `import json` and `import os` to the test file's top-level imports when appending this block.

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `cd backend_ml && ../.venv/bin/pytest test_scheduled_retrain.py -v`
Expected: prior 19 pass; the 5 new `test_main_*` FAIL with `AttributeError: module 'scheduled_retrain' has no attribute 'main'`.

- [ ] **Step 3: Implement `main()`** (append to `backend_ml/scheduled_retrain.py`)

```python
def main(today=None):
    """Run one scheduled-retrain cycle. Returns a process exit code."""
    from datetime import date
    today = today or date.today()

    if not in_season(today):
        log_run("skipped: offseason")
        return 0

    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="nba_retrain_")
    try:
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
            log_run("deployed", new_acc, current_acc)
        else:
            log_run("rejected", new_acc, current_acc)
        return 0
    except Exception as e:
        log_run(f"failed: {e!r}")
        return 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create the plist**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!--
  Nightly NBA model retrain (04:00 daily; no-ops outside the season window).
  Spec: docs/superpowers/specs/2026-07-16-scheduled-retrain-job-design.md

  Install:
    cp backend_ml/scripts/com.benchwarmer.nba-retrain.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.benchwarmer.nba-retrain.plist
  Uninstall:
    launchctl unload ~/Library/LaunchAgents/com.benchwarmer.nba-retrain.plist
    rm ~/Library/LaunchAgents/com.benchwarmer.nba-retrain.plist
  Check outcomes:
    cat backend_ml/retrain_log.txt
-->
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.benchwarmer.nba-retrain</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/vaibhav.wudaru/benchwarmer-nba/.venv/bin/python</string>
        <string>scheduled_retrain.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/vaibhav.wudaru/benchwarmer-nba/backend_ml</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>4</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/vaibhav.wudaru/benchwarmer-nba/backend_ml/scheduled_retrain.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/vaibhav.wudaru/benchwarmer-nba/backend_ml/scheduled_retrain.stderr.log</string>
</dict>
</plist>
```

Validate it: `plutil -lint backend_ml/scripts/com.benchwarmer.nba-retrain.plist` → expected `OK`.

- [ ] **Step 5: Add the gitignore entry**

In `backend_ml/.gitignore`, under the existing `# Logs` section, add one line:

```
retrain_log.txt
```

- [ ] **Step 6: Run the full backend test set**

Run: `cd backend_ml && ../.venv/bin/pytest test_scheduled_retrain.py test_train_model_save.py test_data_cache.py -v && ../.venv/bin/pytest signal_research/ -q`
Expected: 31 passed in the first run (24 scheduled_retrain + 2 save + 5 cache); 75 passed in signal_research.

- [ ] **Step 7: Commit**

```bash
cd /Users/vaibhav.wudaru/benchwarmer-nba
git add backend_ml/scheduled_retrain.py backend_ml/test_scheduled_retrain.py backend_ml/scripts/com.benchwarmer.nba-retrain.plist backend_ml/.gitignore
git status --short   # MUST NOT show ensemble_weights.json as staged
git commit -m "Add nightly retrain orchestration + launchd job

main() trains into a temp dir, deploys only if new held-out accuracy
is within tolerance of the current model's, logs one line per run,
and never touches live artifacts on failure. Plist runs it 04:00
daily from backend_ml/ via the repo venv."
```

---

## Self-Review Notes

- **Spec coverage:** season-window skip (T3/T4), temp-dir training (T2/T4), validate-before-swap with TOLERANCE (T3/T4), copy-then-rename with fault-injection test (T3), log format incl. all 4 outcomes (T3/T4), launchd plist with stderr capture + install note (T4), gitignore (T4), manual-run regression guard (T2), dependency on #1/#2 committed with tests (T1). No gaps found.
- **Known interaction (intentional):** the currently-deployed local `ensemble_weights.json` (the 2026-07-16 re-tune) has no `test_accuracy` key → `read_test_accuracy` returns `None` → the first scheduled run deploys a fresh full retrain unconditionally. That is the desired outcome per that file's own caveat note.
- **Type consistency check:** `train_and_optimize_model(output_dir=".") -> bool` used identically in T2 signature and T4 `main()`; `ARTIFACTS` order (weights JSON last) relied on by T4's fixture (`ARTIFACTS[:-1]` = the 3 pkl names) — consistent with T3's definition.
