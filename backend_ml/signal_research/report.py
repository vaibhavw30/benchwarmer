"""CLI + orchestration for the signal harness.

Testable functions (run_evaluate, load_snapshots) take plain data. Only main()
touches real .pkl models, data_engine, Supabase, and the live market fetchers.
"""
import argparse
import json
import os
from pathlib import Path

from backend_ml.signal_research import calibration as cal
from backend_ml.signal_research import recalibration as rc
from backend_ml.signal_research import clv as clv_mod
from backend_ml.signal_research import config

ARTIFACT_PATH = "backend_ml/signal_research/artifacts/recalibrator.json"
SNAPSHOTS_PATH = os.getenv("MARKET_SNAPSHOTS_PATH",
                           "backend_ml/signal_research/artifacts/market_snapshots.jsonl")


def run_evaluate(dataset_df, method="isotonic", artifact_path=ARTIFACT_PATH, min_n=200) -> dict:
    p = dataset_df["p_model"].to_numpy(dtype=float)
    y = dataset_df["outcome"].to_numpy(dtype=float)
    if len(p) < min_n:
        return {"insufficient": True, "n": int(len(p)),
                "message": f"need >= {min_n} labeled games, have {len(p)}"}
    calibration = cal.calibration_report(p, y)
    recal_eval = rc.evaluate_recalibration(p, y, method=method)
    rc.fit_recalibrator(p, y, method=method).save(artifact_path)
    return {"insufficient": False, "calibration": calibration,
            "recalibration": recal_eval, "artifact": str(artifact_path)}


def load_snapshots(path=SNAPSHOTS_PATH) -> dict:
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.setdefault(row["game_id"], {})[row["moment"]] = row
    return out


def _cmd_evaluate(args):
    # Heavy, real-artifact path lives here only.
    import joblib
    import sys
    # Append (not insert-0) the absolute backend_ml/ dir so data_engine's
    # script-style imports resolve WITHOUT letting backend_ml/signal_research/ shadow
    # the stdlib `signal` module for a downstream bare `import signal`
    # (joblib/sklearn/numpy). See before-live-checklist F.
    _bml = str(Path(__file__).resolve().parents[1])   # backend_ml/ dir
    if _bml not in sys.path:
        sys.path.append(_bml)
    from data_engine import build_training_dataset
    from backend_ml.signal_research import dataset as ds

    games = build_training_dataset()
    xgb = joblib.load("backend_ml/xgboost_nba_model.pkl")
    ridge = joblib.load("backend_ml/ridge_nba_model.pkl")
    scaler = joblib.load("backend_ml/feature_scaler.pkl")
    weights = json.loads(Path("backend_ml/ensemble_weights.json").read_text())
    df = ds.build_recompute_dataset(games, xgb, ridge, scaler, weights)
    out = run_evaluate(df, method=args.method)
    print(json.dumps(out, indent=2, default=str))


def _cmd_clv_report(args):
    fee = config.load_fee_cents()
    snaps = load_snapshots()
    out = clv_mod.clv_report(snaps, fee_cents=fee, min_samples=args.min_samples)
    print(json.dumps(out, indent=2))


def _cmd_capture(args):
    # Live fetchers wired here; deferred to user's live verification.
    raise SystemExit("capture requires live Kalshi/odds credentials; run manually "
                     "with fetchers wired per before-live-checklist.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="signal")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("evaluate")
    pe.add_argument("--method", default="isotonic", choices=["isotonic", "platt"])
    pe.set_defaults(func=_cmd_evaluate)

    pc = sub.add_parser("clv-report")
    pc.add_argument("--min-samples", type=int, default=30)
    pc.set_defaults(func=_cmd_clv_report)

    pcap = sub.add_parser("capture")
    pcap.set_defaults(func=_cmd_capture)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
