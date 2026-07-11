"""Closing Line Value and fee-aware edge-vs-close.

Pure. Operates on captured snapshots. CLV pairs the t-60 (entry) and tipoff
(closing) Kalshi prices per game. Forward-accruing: below min_samples the
report refuses to summarize, so an empty result reads as 'not enough captured
slates yet', never 'broken'.

The position side is inferred once per game from the entry snapshot: if the
market implies YES is underpriced vs the book consensus we'd be long YES,
else NO. Absent a book leg we default to YES (the report still measures raw
price drift; side only sets the sign convention).
"""


def clv_cents(entry_price_cents, closing_price_cents, side: str) -> float:
    delta = closing_price_cents - entry_price_cents
    if side == "YES":
        return float(delta)
    if side == "NO":
        return float(-delta)
    raise ValueError(f"side must be YES or NO, got {side!r}")


def edge_vs_close_cents(p_model, market_p, fee_cents) -> float:
    return abs(p_model - market_p) * 100.0 - fee_cents


def _side_for_game(entry) -> str:
    book_p = entry.get("book_p")
    if book_p is None:
        return "YES"
    # long YES when Kalshi's implied YES prob is below the book consensus
    return "YES" if entry["kalshi_p"] < book_p else "NO"


def clv_report(snapshots_by_game, fee_cents, min_samples: int) -> dict:
    clvs, edges = [], []
    for _game, legs in snapshots_by_game.items():
        entry = legs.get("t-60")
        closing = legs.get("tipoff")
        if entry is None or closing is None:
            continue
        side = _side_for_game(entry)
        clvs.append(clv_cents(entry["kalshi_p"] * 100.0,
                              closing["kalshi_p"] * 100.0, side))
        if entry.get("book_p") is not None:
            edges.append(edge_vs_close_cents(entry["book_p"], entry["kalshi_p"], fee_cents))
    n = len(clvs)
    if n < min_samples:
        return {"n": n, "insufficient": True,
                "mean_clv_cents": None, "mean_edge_cents": None}
    return {
        "n": n,
        "insufficient": False,
        "mean_clv_cents": sum(clvs) / n,
        "mean_edge_cents": (sum(edges) / len(edges)) if edges else None,
    }
