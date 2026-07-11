import numpy as np
import pandas as pd
from backend_ml.signal import calibration as cal


def test_brier_half_p_half_win():
    p = np.full(100, 0.5)
    y = np.array([1] * 50 + [0] * 50)
    assert cal.brier_score(p, y) == 0.25


def test_brier_perfect():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1, 0, 1, 0])
    assert cal.brier_score(p, y) == 0.0


def test_log_loss_is_clipped_and_finite():
    p = np.array([0.0, 1.0])          # extreme, would be inf unclipped
    y = np.array([1, 0])              # both "wrong"
    assert np.isfinite(cal.log_loss(p, y))


def test_reliability_table_bins_and_counts():
    p = np.array([0.05, 0.15, 0.95, 0.95])
    y = np.array([0, 0, 1, 1])
    tbl = cal.reliability_table(p, y, n_bins=10)
    assert list(tbl.columns) == ["bin_lo", "bin_hi", "count", "mean_pred", "mean_obs"]
    # two points land in the top bin, both win -> mean_obs == 1.0
    top = tbl[tbl["bin_lo"] == 0.9].iloc[0]
    assert top["count"] == 2
    assert top["mean_obs"] == 1.0


def test_ece_zero_for_perfectly_calibrated():
    # 1000 points where observed rate matches predicted per bin
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert cal.expected_calibration_error(p, y, n_bins=10) < 0.03


def test_ece_large_for_overconfident():
    p = np.full(1000, 0.9)
    y = np.array([1] * 500 + [0] * 500)   # only 50% actually win
    assert cal.expected_calibration_error(p, y, n_bins=10) > 0.35


def test_calibration_report_keys():
    p = np.array([0.5, 0.5])
    y = np.array([1, 0])
    rep = cal.calibration_report(p, y)
    assert set(rep) == {"brier", "log_loss", "ece", "n", "reliability"}
    assert rep["n"] == 2
