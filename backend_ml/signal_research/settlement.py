"""Map captured Kalshi tickers to settled outcomes {ticker: 0|1}.

build_settlements / merge_settlements are pure (fetcher injected). The live
fetch_kalshi_result issues one signed REST GET and is deferred-verified by the
user (needs creds + in-season markets); tests exercise it with a fake session.

Convention: YES = home-win, so result "yes" -> 1, "no" -> 0.
"""
import time

DEFAULT_BASE_URL = "https://api.elections.kalshi.com"


def build_settlements(tickers, fetch_result) -> dict:
    out = {}
    for ticker in tickers:
        try:
            result = fetch_result(ticker)
        except Exception:
            continue                      # skip this ticker, keep the rest
        if result == "yes":
            out[ticker] = 1
        elif result == "no":
            out[ticker] = 0
        # else: unsettled/unknown -> skip (retried on a later run)
    return out


def merge_settlements(existing: dict, new: dict) -> dict:
    merged = dict(new)
    merged.update(existing)               # existing wins: settlements immutable
    return merged


def fetch_kalshi_result(ticker, signer, session, base_url=DEFAULT_BASE_URL):
    """Signed GET /trade-api/v2/markets/{ticker}; return market.result or None.

    Live seam (deferred-verify). Raises on network/HTTP error so the caller's
    per-ticker try/except in build_settlements can skip just this ticker.
    """
    path = f"/trade-api/v2/markets/{ticker}"
    ts_ms = int(time.time() * 1000)
    headers = signer.headers("GET", path, ts_ms)
    resp = session.get(base_url + path, headers=headers, timeout=10)
    resp.raise_for_status()
    market = resp.json().get("market", {})
    result = market.get("result")
    if result in ("yes", "no"):
        return result
    return None
