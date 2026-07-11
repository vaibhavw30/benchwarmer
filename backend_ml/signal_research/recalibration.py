"""Fit a monotone recalibration map and evaluate it out-of-sample.

Isotonic (default) is stored JSON-native as knot points and applied with
np.interp, so loading and transforming need no sklearn. Platt stores two
scalars. Improvement is always measured on a held-out test split.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from backend_ml.signal_research.calibration import brier_score

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Recalibrator:
    def __init__(self, method, params):
        self.method = method
        self._params = params

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            x = np.asarray(self._params["x"], dtype=float)
            y = np.asarray(self._params["y"], dtype=float)
            return np.clip(np.interp(p, x, y), 0.0, 1.0)
        if self.method == "platt":
            a, b = self._params["a"], self._params["b"]
            return _sigmoid(a * _logit(p) + b)
        raise ValueError(f"unknown method {self.method!r}")

    def to_dict(self):
        return {"method": self.method, **self._params}

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        method = d.pop("method")
        return cls(method, d)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_recalibrator(p, y, method: str = "isotonic") -> Recalibrator:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        # knot points sufficient to reproduce the step function via interp
        xs = np.unique(np.clip(p, 0, 1))
        ys = iso.predict(xs)
        return Recalibrator("isotonic", {"x": xs.tolist(), "y": ys.tolist()})
    if method == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(_logit(p).reshape(-1, 1), y)
        return Recalibrator("platt", {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])})
    raise ValueError(f"unknown method {method!r}")


def evaluate_recalibration(p, y, method="isotonic", test_size=0.3, seed=0) -> dict:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p_tr, p_te, y_tr, y_te = train_test_split(p, y, test_size=test_size, random_state=seed)
    r = fit_recalibrator(p_tr, y_tr, method=method)
    brier_raw = brier_score(p_te, y_te)
    brier_recal = brier_score(r.transform(p_te), y_te)
    return {
        "method": method,
        "n_train": int(p_tr.size),
        "n_test": int(p_te.size),
        "brier_raw": brier_raw,
        "brier_recal": brier_recal,
        "brier_delta": brier_recal - brier_raw,
    }
