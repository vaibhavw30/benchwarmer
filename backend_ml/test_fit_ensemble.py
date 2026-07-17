"""Layer 2 (fit_ensemble behavior) + Layer 4 (regression equivalence) tests.

No .pkl loading: tiny real models on a tiny synthetic frame carrying the 18
feature columns + HOME_WIN.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from signal_research.dataset import FEATURES
from train_model import FitResult, fit_ensemble

TARGET = "HOME_WIN"


def _synth_frame(n=160, seed=0):
    """Separable-ish synthetic frame: label driven by ELO_H sign + noise."""
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(size=n) for f in FEATURES}
    df = pd.DataFrame(data)
    signal = df["ELO_H"].values + 0.3 * rng.normal(size=n)
    df[TARGET] = (signal > 0).astype(int)
    return df


def _Xy(df):
    return df[FEATURES], df[TARGET]


def test_returns_fitted_xgb_ridge_scaler():
    X, y = _Xy(_synth_frame())
    res = fit_ensemble(X, y, params={"n_estimators": 40, "max_depth": 3,
                                     "learning_rate": 0.1})
    assert isinstance(res, FitResult)
    # Fitted: these calls must not raise.
    assert res.xgb_model.predict_proba(X).shape == (len(X), 2)
    assert res.ridge_model.decision_function(res.scaler.transform(X)).shape == (len(X),)


def test_fixed_params_path_skips_grid_search():
    X, y = _Xy(_synth_frame())
    res = fit_ensemble(X, y, params={"n_estimators": 30, "max_depth": 2,
                                     "learning_rate": 0.1})
    # Fixed-param path returns a bare XGBClassifier, never a fitted GridSearchCV.
    assert isinstance(res.xgb_model, XGBClassifier)
    assert not isinstance(res.xgb_model, GridSearchCV)


def test_none_sample_weight_matches_unweighted():
    X, y = _Xy(_synth_frame())
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1}
    a = fit_ensemble(X, y, sample_weight=None, params=params)
    b = fit_ensemble(X, y, sample_weight=np.ones(len(X)), params=params)
    pa = a.xgb_model.predict_proba(X)[:, 1]
    pb = b.xgb_model.predict_proba(X)[:, 1]
    assert np.allclose(pa, pb, atol=1e-6)


def test_sample_weight_changes_the_fit():
    df = _synth_frame(n=200, seed=1)
    X, y = _Xy(df)
    params = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1}
    uniform = fit_ensemble(X, y, sample_weight=None, params=params)
    # Lopsided weights: heavily up-weight the most recent quarter of rows.
    w = np.ones(len(X))
    w[-len(X) // 4:] = 50.0
    weighted = fit_ensemble(X, y, sample_weight=w, params=params)
    p_uniform = uniform.xgb_model.predict_proba(X)[:, 1]
    p_weighted = weighted.xgb_model.predict_proba(X)[:, 1]
    # Threading proof: the two fits must differ materially somewhere.
    assert np.max(np.abs(p_uniform - p_weighted)) > 1e-3


def test_scaler_fit_on_train_only():
    df = _synth_frame()
    X, y = _Xy(df)
    res = fit_ensemble(X, y, params={"n_estimators": 20, "max_depth": 2,
                                     "learning_rate": 0.1})
    manual = StandardScaler().fit(X)
    assert np.allclose(res.scaler.mean_, manual.mean_)
    assert np.allclose(res.scaler.var_, manual.var_)


def _inline_reference_fit(X_train, y_train):
    """Byte-for-byte replica of train_and_optimize_model's pre-refactor fitting
    sequence (train_model.py lines ~99-138 before this task). This IS the
    pre-refactor behavior; fit_ensemble(params=None) must reproduce it."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = {
        "n_estimators": [150, 250],
        "learning_rate": [0.03, 0.05],
        "max_depth": [3, 4],
        "scale_pos_weight": [0.85, 0.9],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(xgb, param_grid, cv=tscv, scoring="accuracy",
                        verbose=0, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge = RidgeClassifierCV(alphas=alphas, cv=tscv, scoring="accuracy")
    ridge.fit(X_train_scaled, y_train)
    return best, ridge, scaler


def test_unweighted_equivalence_smoke():
    """Layer 4 regression guard: the extracted production path (params=None,
    sample_weight=None) reproduces the inline pre-refactor fit exactly."""
    df = _synth_frame(n=180, seed=7)
    X, y = _Xy(df)
    ref_xgb, ref_ridge, ref_scaler = _inline_reference_fit(X, y)
    res = fit_ensemble(X, y, sample_weight=None, params=None)
    # XGBoost probabilities match.
    assert np.allclose(ref_xgb.predict_proba(X)[:, 1],
                       res.xgb_model.predict_proba(X)[:, 1], atol=1e-6)
    # Ridge decisions match (scaler is equivalent -> transform equivalent).
    assert np.allclose(ref_ridge.decision_function(ref_scaler.transform(X)),
                       res.ridge_model.decision_function(res.scaler.transform(X)),
                       atol=1e-6)
