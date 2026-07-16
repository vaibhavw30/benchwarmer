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
