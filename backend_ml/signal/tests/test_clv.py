import pytest

from backend_ml.signal import clv


def test_clv_yes_positive_when_price_rises():
    assert clv.clv_cents(40, 46, "YES") == 6
    assert clv.clv_cents(40, 46, "NO") == -6


def test_clv_cents_invalid_side_raises():
    with pytest.raises(ValueError):
        clv.clv_cents(40, 46, "MAYBE")


def test_edge_after_fee_can_flip_negative():
    # raw edge 3c, fee 1c -> +2c
    assert abs(clv.edge_vs_close_cents(0.55, 0.52, fee_cents=1) - 2.0) < 1e-9
    # raw edge 0.5c, fee 1c -> negative (not real edge)
    assert clv.edge_vs_close_cents(0.505, 0.50, fee_cents=1) < 0


def test_clv_report_insufficient_below_min():
    snaps = {"g1": {"t-60": {"kalshi_p": 0.40}, "tipoff": {"kalshi_p": 0.46}}}
    rep = clv.clv_report(snaps, fee_cents=1, min_samples=5)
    assert rep["insufficient"] is True
    assert rep["mean_clv_cents"] is None
    assert rep["n"] == 1


def test_clv_report_computes_means_when_enough():
    snaps = {
        f"g{i}": {"t-60": {"kalshi_p": 0.40, "book_p": 0.50},
                  "tipoff": {"kalshi_p": 0.46, "book_p": 0.50}}
        for i in range(5)
    }
    rep = clv.clv_report(snaps, fee_cents=1, min_samples=5)
    assert rep["insufficient"] is False
    assert rep["n"] == 5
    assert abs(rep["mean_clv_cents"] - 6.0) < 1e-9   # YES side, 40 -> 46
    assert abs(rep["mean_edge_cents"] - 9.0) < 1e-9  # abs(0.50-0.40)*100 - 1 fee


def test_clv_report_skips_games_missing_a_leg():
    snaps = {
        "g1": {"t-60": {"kalshi_p": 0.40}, "tipoff": {"kalshi_p": 0.46}},
        "g2": {"t-60": {"kalshi_p": 0.40}},           # no closing leg -> skipped
    }
    rep = clv.clv_report(snaps, fee_cents=1, min_samples=1)
    assert rep["n"] == 1


def test_clv_report_empty_is_insufficient_not_crash():
    rep = clv.clv_report({}, fee_cents=1, min_samples=0)
    assert rep["n"] == 0
    assert rep["insufficient"] is True
    assert rep["mean_clv_cents"] is None


def test_moment_constants_match_stored_keys():
    from backend_ml.signal.market_capture import ENTRY_MOMENT, CLOSING_MOMENT
    assert (ENTRY_MOMENT, CLOSING_MOMENT) == ("t-60", "tipoff")
