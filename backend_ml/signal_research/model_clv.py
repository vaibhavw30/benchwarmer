"""Model-vs-market CLV metrics (pure).

Brings the model's own probability (p_model, captured from the published
fair_values.json) into the CLV analysis:
  - model CLV: when the model takes a side at entry, does the close move toward us
  - entry edge: |p_model - kalshi_entry| after fee
  - model beats close: Brier(p_model) vs Brier(closing kalshi price) at settlement

Reuses clv.clv_cents for side-signed price drift. The edge threshold mirrors the
C++ engine (trading_engine/src/strategy/pricing.cpp); test_edge_threshold_parity
locks the formula to the engine's known values.
"""
import math

from backend_ml.signal_research.clv import clv_cents
from backend_ml.signal_research.market_capture import ENTRY_MOMENT, CLOSING_MOMENT


def edge_threshold_cents(confidence, base_edge_cents, fee_cents, confidence_k) -> int:
    return int(math.floor(base_edge_cents + fee_cents + confidence_k * (1.0 - confidence)))


def model_side(p_model, kalshi_entry_p) -> str:
    return "YES" if p_model > kalshi_entry_p else "NO"


def entry_edge_cents(p_model, kalshi_entry_p, fee_cents) -> float:
    return abs(p_model - kalshi_entry_p) * 100.0 - fee_cents


def would_trade(p_model, kalshi_entry_p, base_edge_cents, fee_cents, confidence_k) -> bool:
    """Mid-price, continuous-fair approximation of the engine's trade decision.

    The live engine's detect_take (trading_engine/src/strategy/edge_taker.cpp)
    compares an integer fair = clamp(lround(100*p_yes), 1, 99) against the
    executable top-of-book (buy iff best_yes_ask <= fair - threshold), whereas
    this compares one continuous kalshi_entry_p to fair. As a result the
    would_trade bucket is an upper bound: it over-includes marginal games
    relative to what the engine would actually trade.
    """
    confidence = max(p_model, 1.0 - p_model)
    threshold = edge_threshold_cents(confidence, base_edge_cents, fee_cents, confidence_k)
    divergence_cents = abs(p_model - kalshi_entry_p) * 100.0
    return divergence_cents >= threshold


def _bucket(clvs, edges) -> dict:
    n = len(clvs)
    if n == 0:
        return {"n": 0, "mean_clv_cents": None, "pct_positive_clv": None,
                "mean_entry_edge_cents": None}
    return {
        "n": n,
        "mean_clv_cents": sum(clvs) / n,
        "pct_positive_clv": sum(1 for c in clvs if c > 0) / n,
        "mean_entry_edge_cents": sum(edges) / n,
    }


def model_clv_report(snapshots_by_game, settlements, *, base_edge_cents, fee_cents,
                     confidence_k, min_samples, signal_version="unknown") -> dict:
    raw_clv, raw_edge = [], []
    wt_clv, wt_edge = [], []
    brier_model, brier_close, model_wins = [], [], 0

    for _game, legs in snapshots_by_game.items():
        entry = legs.get(ENTRY_MOMENT)
        closing = legs.get(CLOSING_MOMENT)
        if entry is None or closing is None:
            continue
        pm = entry.get("p_model")
        if pm is None:
            continue
        ke, kc = entry["kalshi_p"], closing["kalshi_p"]
        side = model_side(pm, ke)
        clv = clv_cents(ke * 100.0, kc * 100.0, side)
        edge = entry_edge_cents(pm, ke, fee_cents)
        raw_clv.append(clv)
        raw_edge.append(edge)
        if would_trade(pm, ke, base_edge_cents, fee_cents, confidence_k):
            wt_clv.append(clv)
            wt_edge.append(edge)
        outcome = settlements.get(entry["ticker"])
        if outcome is not None:
            bm = (pm - outcome) ** 2
            bc = (kc - outcome) ** 2
            brier_model.append(bm)
            brier_close.append(bc)
            if bm < bc:
                model_wins += 1

    n = len(raw_clv)
    if n < min_samples or n == 0:
        return {
            "signal_version": signal_version,
            "insufficient": True,
            "raw_sign": {"n": n, "mean_clv_cents": None, "pct_positive_clv": None,
                         "mean_entry_edge_cents": None},
            "would_trade": {"n": len(wt_clv), "mean_clv_cents": None,
                            "pct_positive_clv": None, "mean_entry_edge_cents": None},
            "beats_close": None,
        }

    beats = None
    ns = len(brier_model)
    if ns > 0:
        bm_mean = sum(brier_model) / ns
        bc_mean = sum(brier_close) / ns
        beats = {
            "n_settled": ns,
            "brier_model": bm_mean,
            "brier_close": bc_mean,
            "brier_delta": bm_mean - bc_mean,
            "pct_model_beats_close": model_wins / ns,
        }

    return {
        "signal_version": signal_version,
        "insufficient": False,
        "raw_sign": _bucket(raw_clv, raw_edge),
        "would_trade": _bucket(wt_clv, wt_edge),
        "beats_close": beats,
    }
