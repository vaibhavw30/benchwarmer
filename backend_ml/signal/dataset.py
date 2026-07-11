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
