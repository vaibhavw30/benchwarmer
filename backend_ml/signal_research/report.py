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
DEFAULT_PLOT_PATH = "backend_ml/signal_research/artifacts/reliability.png"
SNAPSHOTS_PATH = os.getenv("MARKET_SNAPSHOTS_PATH",
                           "backend_ml/signal_research/artifacts/market_snapshots.jsonl")
SETTLEMENTS_PATH = os.getenv("SETTLEMENTS_PATH",
                             "backend_ml/signal_research/artifacts/settlements.json")


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


def load_settlements(path=SETTLEMENTS_PATH) -> dict:
    """Map ticker -> outcome (1 = YES/home won, 0). {} if the file is absent.

    Populated live by the deferred Kalshi settlement fetch (see
    before-live-checklist). The file format is a JSON object {ticker: 0|1}.
    """
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for k, v in json.loads(p.read_text()).items():
        outcome = int(v)
        if outcome not in (0, 1):
            raise ValueError(f"settlement outcome must be 0 or 1: {k!r}={v!r}")
        out[k] = outcome
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
    import os
    import pandas as pd

    # Prefer the cached history (build_training_dataset re-scrapes nba_api on
    # every call). Paths are anchored to the backend_ml/ dir so evaluate works
    # regardless of the caller's cwd.
    cache = os.path.join(_bml, "nba_training_cache.csv")
    if os.path.exists(cache):
        games = pd.read_csv(cache)
    else:
        games = build_training_dataset()
    xgb = joblib.load(os.path.join(_bml, "xgboost_nba_model.pkl"))
    ridge = joblib.load(os.path.join(_bml, "ridge_nba_model.pkl"))
    scaler = joblib.load(os.path.join(_bml, "feature_scaler.pkl"))
    weights = json.loads(Path(_bml, "ensemble_weights.json").read_text())
    df = ds.build_recompute_dataset(games, xgb, ridge, scaler, weights)
    out = run_evaluate(df, method=args.method)
    print(json.dumps(out, indent=2, default=str))
    if getattr(args, "plot", None) and not out.get("insufficient"):
        from backend_ml.signal_research import plots
        from backend_ml.signal_research.recalibration import Recalibrator
        recal = Recalibrator.load(out["artifact"])
        c = out["calibration"]
        title = (f"Reliability — Brier {c['brier']:.4f}  ECE {c['ece']:.3f}  "
                 f"n={c['n']}")
        path = plots.plot_reliability(c, args.plot, recalibrator=recal, title=title)
        print(f"wrote plot -> {path}")


def _cmd_clv_report(args):
    fee = config.load_fee_cents()
    snaps = load_snapshots()
    out = clv_mod.clv_report(snaps, fee_cents=fee, min_samples=args.min_samples)
    print(json.dumps(out, indent=2))


def _cmd_model_clv_report(args):
    from backend_ml.signal_research import model_clv
    params = config.load_edge_params()
    snaps = load_snapshots()
    settlements = load_settlements()
    out = model_clv.model_clv_report(
        snaps, settlements, min_samples=args.min_samples,
        signal_version=args.signal_version, **params)
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
    pe.add_argument("--plot", metavar="PATH", nargs="?", const=DEFAULT_PLOT_PATH,
                    help="write a reliability-diagram PNG (default: "
                         "artifacts/reliability.png)")
    pe.set_defaults(func=_cmd_evaluate)

    pc = sub.add_parser("clv-report")
    pc.add_argument("--min-samples", type=int, default=30)
    pc.set_defaults(func=_cmd_clv_report)

    pmc = sub.add_parser("model-clv-report")
    pmc.add_argument("--min-samples", type=int, default=30)
    pmc.add_argument("--signal-version", default="unknown",
                     choices=["raw", "recalibrated", "unknown"])
    pmc.set_defaults(func=_cmd_model_clv_report)

    pcap = sub.add_parser("capture")
    pcap.set_defaults(func=_cmd_capture)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
