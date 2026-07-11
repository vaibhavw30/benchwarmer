"""Publish per-Kalshi-ticker fair values from the existing ensemble model.

Reuses backend_ml/predict.py:predict_games() output. The only new Python file
in this project; the model math is untouched.
"""
import json
import datetime
import os
from pathlib import Path


def build_fair_values(predictions, watchlist, recalibrator=None):
    """Pure mapping: (model predictions, ticker watchlist) -> fair-value rows.

    Joins on (home_team_id, away_team_id, game_date). Unmapped games are
    skipped (fail-closed): a wrong ticker means trading the wrong game.

    When `recalibrator` is provided, p_yes is passed through the recalibration
    map and confidence is recomputed as max(p, 1-p). When None, behavior is
    byte-identical to the pre-recalibration publisher.
    """
    index = {(w["home_team_id"], w["away_team_id"], w["game_date"]): w["ticker"]
             for w in watchlist}
    rows = []
    for p in predictions:
        key = (p["home_team_id"], p["away_team_id"], p["date"])
        ticker = index.get(key)
        if ticker is None:
            continue
        # Clamp to [0,1] at the boundary: a bad/out-of-range model value
        # here must not propagate into edge_threshold_cents (which can go
        # <= 0 for confidence outside [0,1]) or into a nonsensical price.
        p_yes = max(0.0, min(1.0, float(p["home_win_probability"])))
        confidence = max(0.0, min(1.0, float(p["confidence_score"])))
        if recalibrator is not None:
            p_yes = float(recalibrator.transform([p_yes])[0])
            confidence = max(p_yes, 1.0 - p_yes)
        rows.append({
            "ticker": ticker,
            "p_yes": p_yes,               # YES = home wins
            "confidence": confidence,
            "asof": datetime.datetime.utcnow().isoformat() + "Z",
            "game_id": p["game_id"],
        })
    return rows


def main():
    # Import predict lazily (heavy deps). Append (not insert-0) the backend_ml/
    # dir so predict.py's script-style imports (e.g. `from data_engine import
    # ...`) resolve WITHOUT letting backend_ml/signal/ shadow the stdlib
    # `signal` module for a downstream bare `import signal`. See
    # before-live-checklist F.
    import sys
    _bml = str(Path(__file__).resolve().parent)  # backend_ml/ dir
    if _bml not in sys.path:
        sys.path.append(_bml)
    from predict import predict_games

    wl_path = os.getenv("WATCHLIST_PATH", "trading_engine/config/watchlist.json")
    out_path = os.getenv("FAIR_VALUES_PATH", "trading_engine/fair_values.json")
    watchlist = json.loads(Path(wl_path).read_text())
    predictions = predict_games(day_offset=0)   # existing model entrypoint
    if not isinstance(predictions, list):
        raise SystemExit("predict_games did not return a prediction list")
    recalibrator = None
    if os.getenv("RECALIBRATE") == "1":
        from backend_ml.signal.recalibration import Recalibrator
        recalibrator = Recalibrator.load("backend_ml/signal/artifacts/recalibrator.json")
    rows = build_fair_values(predictions, watchlist, recalibrator=recalibrator)
    Path(out_path).write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} fair values -> {out_path}")


if __name__ == "__main__":
    main()
