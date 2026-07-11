"""Forward market-price capture for CLV.

Pure pieces (american_to_prob, devig, build_snapshot_rows) are fully tested.
The live fetchers (Kalshi REST GET, sportsbook odds) are injected so tests
never touch the network; the real fetchers are wired in the CLI (Task 6) and
live-verified by the user.

Snapshots are appended as JSONL. Path is gitignored.
"""
import json
from pathlib import Path

# Canonical capture moments. The live capture caller MUST pass these exact
# values as `moment`; clv.clv_report pairs legs by them. Do not inline the
# string literals elsewhere — a mismatch silently pairs zero legs.
ENTRY_MOMENT = "t-60"
CLOSING_MOMENT = "tipoff"


def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def devig(home_american: float, away_american: float):
    ph = american_to_prob(home_american)
    pa = american_to_prob(away_american)
    total = ph + pa
    return ph / total, pa / total


def build_snapshot_rows(watchlist, moment, kalshi_prices, book_odds, asof):
    rows = []
    for w in watchlist:
        ticker = w["ticker"]
        cents = kalshi_prices.get(ticker)
        odds = book_odds.get(ticker)
        if cents is None or odds is None:
            continue                      # fail-closed: incomplete market data
        book_p, _ = devig(odds[0], odds[1])
        rows.append({
            "ticker": ticker,
            "game_id": w["game_id"],
            "moment": moment,
            "kalshi_p": float(cents) / 100.0,
            "book_p": book_p,
            "asof": asof,
        })
    return rows


def append_snapshots(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def capture(watchlist, moment, asof, fetch_kalshi_price, fetch_two_way_odds, path):
    kalshi_prices, book_odds = {}, {}
    for w in watchlist:
        ticker = w["ticker"]
        try:
            kalshi_prices[ticker] = fetch_kalshi_price(ticker)
            book_odds[ticker] = fetch_two_way_odds(w)
        except Exception:
            continue                      # skip this ticker, keep the rest
    rows = build_snapshot_rows(watchlist, moment, kalshi_prices, book_odds, asof)
    append_snapshots(rows, path)
    return len(rows)
