"""Layer 1 (pure unit) tests for recency_weights."""
import numpy as np
import pytest

from recency import recency_weights


def test_none_halflife_returns_ones():
    w = recency_weights(5, None)
    assert np.array_equal(w, np.ones(5))


def test_inf_halflife_returns_ones():
    w = recency_weights(5, np.inf)
    assert np.array_equal(w, np.ones(5))


def test_newest_row_weight_is_one():
    w = recency_weights(10, 3.0)
    assert w[-1] == 1.0


def test_weights_strictly_increasing_with_index():
    w = recency_weights(20, 5.0)
    assert np.all(np.diff(w) > 0)


def test_one_halflife_older_row_is_half():
    # Row exactly H games older than the newest has weight 0.5.
    n, H = 11, 4.0
    w = recency_weights(n, H)
    # newest index = n-1 = 10; H games older => index 10 - 4 = 6
    assert w[6] == pytest.approx(0.5, abs=1e-12)


def test_two_halflives_older_row_is_quarter():
    n, H = 11, 4.0
    w = recency_weights(n, H)
    # 2H games older => index 10 - 8 = 2
    assert w[2] == pytest.approx(0.25, abs=1e-12)


def test_n_zero_returns_empty():
    w = recency_weights(0, 3.0)
    assert isinstance(w, np.ndarray) and w.shape == (0,)


def test_n_one_returns_single_one():
    w = recency_weights(1, 3.0)
    assert np.array_equal(w, np.array([1.0]))


@pytest.mark.parametrize("bad_H", [0, -1, -0.5, 0.0])
def test_nonpositive_halflife_raises(bad_H):
    with pytest.raises(ValueError):
        recency_weights(5, bad_H)


def test_output_dtype_length_and_range():
    w = recency_weights(50, 7.0)
    assert w.dtype == np.float64
    assert len(w) == 50
    assert np.all(np.isfinite(w))
    assert np.all(w > 0) and np.all(w <= 1.0)


@pytest.mark.parametrize("n", [2, 3, 8, 50, 500])
@pytest.mark.parametrize("H", [1.0, 10.0, 137.0, 2500.0])
def test_property_monotone_max_one_and_bounded(n, H):
    # Parametrized stand-in for property-based testing (hypothesis not installed).
    w = recency_weights(n, H)
    assert np.all(np.diff(w) >= 0)          # monotone non-decreasing
    assert w.max() == pytest.approx(1.0)    # newest is the max, == 1.0
    assert np.all(w > 0) and np.all(w <= 1.0)
