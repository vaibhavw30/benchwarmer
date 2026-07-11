"""Pure calibration metrics on (predicted probability, binary outcome).

No I/O, no model loading. Every function takes array-likes and returns a
number or a tidy DataFrame/dict. This is the harness's TDD core.
"""
import numpy as np
import pandas as pd

_EPS = 1e-15


def _arrays(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"p and y shape mismatch: {p.shape} vs {y.shape}")
    return p, y


def brier_score(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p, y = _arrays(p, y)
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    p, y = _arrays(p, y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False; clip so p==1.0 lands in the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        rows.append({
            "bin_lo": round(float(edges[b]), 4),
            "bin_hi": round(float(edges[b + 1]), 4),
            "count": count,
            "mean_pred": float(p[mask].mean()) if count else float("nan"),
            "mean_obs": float(y[mask].mean()) if count else float("nan"),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    p, y = _arrays(p, y)
    tbl = reliability_table(p, y, n_bins)
    tbl = tbl[tbl["count"] > 0]
    weights = tbl["count"] / len(p)
    gaps = (tbl["mean_pred"] - tbl["mean_obs"]).abs()
    return float((weights * gaps).sum())


def calibration_report(p, y, n_bins: int = 10) -> dict:
    p, y = _arrays(p, y)
    tbl = reliability_table(p, y, n_bins)
    return {
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "ece": expected_calibration_error(p, y, n_bins),
        "n": int(p.size),
        "reliability": tbl.to_dict(orient="records"),
    }
