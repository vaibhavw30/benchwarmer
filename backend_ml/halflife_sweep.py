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
