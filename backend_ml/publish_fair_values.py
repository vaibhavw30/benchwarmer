"""Publish per-Kalshi-ticker fair values from the existing ensemble model.

Reuses backend_ml/predict.py:predict_games() output. The only new Python file
in this project; the model math is untouched.
"""
import json
import datetime
import os
from pathlib import Path


def build_fair_values(predictions, watchlist):
    """Pure mapping: (model predictions, ticker watchlist) -> fair-value rows.

    Joins on (home_team_id, away_team_id, game_date). Unmapped games are
    skipped (fail-closed): a wrong ticker means trading the wrong game.
    """
    index = {(w["home_team_id"], w["away_team_id"], w["game_date"]): w["ticker"]
             for w in watchlist}
    rows = []
    for p in predictions:
        key = (p["home_team_id"], p["away_team_id"], p["date"])
        ticker = index.get(key)
        if ticker is None:
            continue
        rows.append({
            "ticker": ticker,
            # Clamp to [0,1] at the boundary: a bad/out-of-range model value
            # here must not propagate into edge_threshold_cents (which can go
            # <= 0 for confidence outside [0,1]) or into a nonsensical price.
            "p_yes": max(0.0, min(1.0, float(p["home_win_probability"]))),   # YES = home wins
            "confidence": max(0.0, min(1.0, float(p["confidence_score"]))),
            "asof": datetime.datetime.utcnow().isoformat() + "Z",
            "game_id": p["game_id"],
        })
    return rows


def main():
    # Import predict lazily (heavy deps). Put backend_ml/ on sys.path so
    # predict.py's script-style imports (e.g. `from data_engine import ...`)
    # resolve.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # backend_ml/ dir
    from predict import predict_games

    wl_path = os.getenv("WATCHLIST_PATH", "trading_engine/config/watchlist.json")
    out_path = os.getenv("FAIR_VALUES_PATH", "trading_engine/fair_values.json")
    watchlist = json.loads(Path(wl_path).read_text())
    predictions = predict_games(day_offset=0)   # existing model entrypoint
    if not isinstance(predictions, list):
        raise SystemExit("predict_games did not return a prediction list")
    rows = build_fair_values(predictions, watchlist)
    Path(out_path).write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} fair values -> {out_path}")


if __name__ == "__main__":
    main()
