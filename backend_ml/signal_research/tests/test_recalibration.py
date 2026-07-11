import numpy as np
import pytest
from backend_ml.signal_research import recalibration as rc
from backend_ml.signal_research import calibration as cal


def _miscalibrated(n=4000, seed=1):
    # true outcome prob is p_raw**2 -> model is systematically overconfident
    rng = np.random.default_rng(seed)
    p_raw = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p_raw ** 2).astype(int)
    return p_raw, y


def test_isotonic_reduces_out_of_sample_brier():
    p, y = _miscalibrated()
    res = rc.evaluate_recalibration(p, y, method="isotonic", test_size=0.3, seed=0)
    assert res["brier_recal"] < res["brier_raw"]
    assert res["brier_delta"] < 0            # delta = recal - raw, improvement is negative
    assert res["n_test"] > 0


def test_transform_is_monotonic_and_bounded():
    p, y = _miscalibrated()
    r = rc.fit_recalibrator(p, y, method="isotonic")
    grid = np.linspace(0, 1, 50)
    out = r.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)     # non-decreasing
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_save_load_roundtrip(tmp_path):
    p, y = _miscalibrated()
    r = rc.fit_recalibrator(p, y, method="isotonic")
    path = tmp_path / "recal.json"
    r.save(path)
    r2 = rc.Recalibrator.load(path)
    grid = np.linspace(0, 1, 20)
    assert np.allclose(r.transform(grid), r2.transform(grid))


def test_platt_also_reduces_brier():
    p, y = _miscalibrated()
    res = rc.evaluate_recalibration(p, y, method="platt", test_size=0.3, seed=0)
    assert res["brier_recal"] <= res["brier_raw"] + 1e-6


def test_unknown_method_raises():
    p, y = _miscalibrated()
    with pytest.raises(ValueError):
        rc.fit_recalibrator(p, y, method="banana")
