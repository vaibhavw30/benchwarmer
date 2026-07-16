from backend_ml.signal_research import model_clv as mc
from backend_ml.signal_research.market_capture import ENTRY_MOMENT, CLOSING_MOMENT

EDGE = {"base_edge_cents": 2, "fee_cents": 1, "confidence_k": 8.0}


def test_edge_threshold_parity_with_cpp_engine():
    # Locks the Python formula to the C++ engine's known values.
    assert mc.edge_threshold_cents(0.95, 2, 1, 8.0) == 3
    assert mc.edge_threshold_cents(0.55, 2, 1, 8.0) == 6


def test_model_side_sign():
    assert mc.model_side(0.62, 0.48) == "YES"
    assert mc.model_side(0.30, 0.48) == "NO"


def test_entry_edge_after_fee_can_go_negative():
    assert abs(mc.entry_edge_cents(0.62, 0.48, 1) - 13.0) < 1e-9   # 14c - 1 fee
    assert mc.entry_edge_cents(0.505, 0.50, 1) < 0                 # 0.5c - 1 fee


def test_would_trade_threshold():
    # p=0.62 vs 0.48: conf 0.62 -> thresh floor(2+1+8*0.38)=6; divergence 14 >= 6
    assert mc.would_trade(0.62, 0.48, 2, 1, 8.0) is True
    # p=0.52 vs 0.50: conf 0.52 -> thresh floor(2+1+8*0.48)=6; divergence 2 < 6
    assert mc.would_trade(0.52, 0.50, 2, 1, 8.0) is False


def _game(ticker, p_model, k_entry, k_close):
    return {
        ENTRY_MOMENT: {"ticker": ticker, "kalshi_p": k_entry, "p_model": p_model},
        CLOSING_MOMENT: {"ticker": ticker, "kalshi_p": k_close, "p_model": p_model},
    }


def test_report_insufficient_below_min():
    snaps = {"g1": _game("T1", 0.62, 0.48, 0.55)}
    out = mc.model_clv_report(snaps, {}, min_samples=5, **EDGE)
    assert out["insufficient"] is True
    assert out["raw_sign"]["mean_clv_cents"] is None
    assert out["beats_close"] is None


def test_report_clv_and_buckets():
    # Two would-trade games (big divergence) + one marginal (no trade).
    snaps = {
        "g1": _game("T1", 0.62, 0.48, 0.55),   # YES, close 48->55 => +7 clv, trades
        "g2": _game("T2", 0.70, 0.50, 0.60),   # YES, close 50->60 => +10 clv, trades
        "g3": _game("T3", 0.52, 0.50, 0.49),   # YES, close 50->49 => -1 clv, no-trade
    }
    out = mc.model_clv_report(snaps, {}, min_samples=1, signal_version="recalibrated", **EDGE)
    assert out["insufficient"] is False
    assert out["signal_version"] == "recalibrated"
    assert out["raw_sign"]["n"] == 3
    assert abs(out["raw_sign"]["mean_clv_cents"] - (7 + 10 - 1) / 3) < 1e-9
    assert out["would_trade"]["n"] == 2                      # g1, g2 only
    assert abs(out["would_trade"]["mean_clv_cents"] - (7 + 10) / 2) < 1e-9
    assert out["would_trade"]["pct_positive_clv"] == 1.0
    assert out["beats_close"] is None                        # no settlements


def test_report_beats_close_with_settlement():
    snaps = {"g1": _game("T1", 0.70, 0.50, 0.60)}
    # home won -> outcome 1. brier_model=(0.7-1)^2=0.09 < brier_close=(0.6-1)^2=0.16
    out = mc.model_clv_report(snaps, {"T1": 1}, min_samples=1, **EDGE)
    bc = out["beats_close"]
    assert bc["n_settled"] == 1
    assert abs(bc["brier_model"] - 0.09) < 1e-9
    assert abs(bc["brier_close"] - 0.16) < 1e-9
    assert bc["brier_delta"] < 0
    assert bc["pct_model_beats_close"] == 1.0


def test_report_skips_games_without_p_model_or_a_leg():
    snaps = {
        "g1": {ENTRY_MOMENT: {"ticker": "T1", "kalshi_p": 0.48, "p_model": None},
               CLOSING_MOMENT: {"ticker": "T1", "kalshi_p": 0.55, "p_model": None}},
        "g2": {ENTRY_MOMENT: {"ticker": "T2", "kalshi_p": 0.48, "p_model": 0.62}},  # no close
    }
    out = mc.model_clv_report(snaps, {}, min_samples=1, **EDGE)
    assert out["insufficient"] is True    # zero usable games
    assert out["raw_sign"]["n"] == 0
