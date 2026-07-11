"""Smoke tests for the reliability-diagram plot.

matplotlib is a project dependency but may be absent in a minimal test env, so
skip cleanly if it isn't importable.
"""
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from backend_ml.signal_research import calibration as cal
from backend_ml.signal_research import plots
from backend_ml.signal_research.recalibration import fit_recalibrator


def test_plot_reliability_writes_nonempty_png(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = (rng.uniform(0, 1, 1000) < p).astype(int)
    rep = cal.calibration_report(p, y)
    out = tmp_path / "rel.png"
    ret = plots.plot_reliability(rep, out)
    assert ret == out
    assert out.exists() and out.stat().st_size > 0


def test_plot_reliability_with_recalibrator_overlay(tmp_path):
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 1000)
    y = (rng.uniform(0, 1, 1000) < p ** 2).astype(int)   # miscalibrated
    rep = cal.calibration_report(p, y)
    recal = fit_recalibrator(p, y, method="isotonic")
    out = tmp_path / "rel_recal.png"
    plots.plot_reliability(rep, out, recalibrator=recal, title="test")
    assert out.exists() and out.stat().st_size > 0
