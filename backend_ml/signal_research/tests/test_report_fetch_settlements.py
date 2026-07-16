import json
import pytest
from backend_ml.signal_research import report
from backend_ml.signal_research.market_capture import ENTRY_MOMENT, CLOSING_MOMENT


def test_unique_tickers_dedups_across_games_and_moments():
    snaps = {
        "g1": {ENTRY_MOMENT: {"ticker": "T1"}, CLOSING_MOMENT: {"ticker": "T1"}},
        "g2": {ENTRY_MOMENT: {"ticker": "T2"}},
        "g3": {ENTRY_MOMENT: {}},          # no ticker -> ignored
    }
    assert report._unique_tickers(snaps) == ["T1", "T2"]


def test_cmd_fetch_settlements_merges_and_writes(tmp_path, monkeypatch):
    # Snapshot store with two captured tickers; T1 already settled.
    snaps = {
        "g1": {ENTRY_MOMENT: {"ticker": "T1"}},
        "g2": {ENTRY_MOMENT: {"ticker": "T2"}},
    }
    out_path = tmp_path / "settlements.json"
    out_path.write_text('{"T1": 1}')      # T1 already settled -> skipped

    monkeypatch.setattr(report, "load_snapshots", lambda: snaps)
    monkeypatch.setattr(report, "SETTLEMENTS_PATH", str(out_path))
    # load_settlements's default path arg is bound at def-time, so point the
    # whole function at our pre-seeded file (re-reads on call).
    monkeypatch.setattr(
        report, "load_settlements",
        lambda: {k: int(v) for k, v in json.loads(out_path.read_text()).items()})

    # Bypass real creds: from_env returns a dummy signer (fetch is stubbed).
    monkeypatch.setattr(
        "backend_ml.signal_research.kalshi_auth.KalshiSigner.from_env",
        classmethod(lambda cls: object()))
    # Fetcher: only T2 should be queried (T1 already settled); it settles "no".
    queried = []
    def fake_fetch(ticker, signer, session, base_url=None):
        queried.append(ticker)
        return "no"
    monkeypatch.setattr(
        "backend_ml.signal_research.settlement.fetch_kalshi_result", fake_fetch)

    report._cmd_fetch_settlements(None)

    assert queried == ["T2"]              # already-settled T1 not re-queried
    written = json.loads(out_path.read_text())
    assert written == {"T1": 1, "T2": 0}  # existing T1 preserved, T2 added
