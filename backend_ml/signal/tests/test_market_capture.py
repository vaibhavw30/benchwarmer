import json
import numpy as np
from backend_ml.signal import market_capture as mc


def test_american_to_prob_even_and_favorite():
    assert abs(mc.american_to_prob(100) - 0.5) < 1e-9
    assert abs(mc.american_to_prob(-110) - (110 / 210)) < 1e-9
    assert abs(mc.american_to_prob(150) - (100 / 250)) < 1e-9


def test_devig_even_market_is_half():
    ph, pa = mc.devig(-110, -110)
    assert abs(ph - 0.5) < 1e-9 and abs(pa - 0.5) < 1e-9


def test_devig_sums_to_one_and_orders_favorite():
    ph, pa = mc.devig(-200, +170)     # home favorite
    assert abs((ph + pa) - 1.0) < 1e-9
    assert ph > pa


def test_build_snapshot_rows_joins_and_shapes():
    watchlist = [{"ticker": "KXNBA-LAL-BOS", "game_id": "g1",
                  "home_team_id": 1, "away_team_id": 2, "game_date": "2026-01-01"}]
    rows = mc.build_snapshot_rows(
        watchlist, moment="tipoff",
        kalshi_prices={"KXNBA-LAL-BOS": 55},          # cents
        book_odds={"KXNBA-LAL-BOS": (-110, -110)},
        asof="2026-01-01T00:00:00Z")
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "KXNBA-LAL-BOS"
    assert r["game_id"] == "g1"
    assert r["moment"] == "tipoff"
    assert abs(r["kalshi_p"] - 0.55) < 1e-9
    assert abs(r["book_p"] - 0.5) < 1e-9
    assert r["asof"] == "2026-01-01T00:00:00Z"


def test_build_snapshot_rows_skips_missing_market_data():
    watchlist = [{"ticker": "T1", "game_id": "g1", "home_team_id": 1,
                  "away_team_id": 2, "game_date": "2026-01-01"}]
    # no kalshi price for T1 -> row skipped (fail-closed)
    rows = mc.build_snapshot_rows(watchlist, "tipoff", {}, {}, "2026-01-01T00:00:00Z")
    assert rows == []


def test_append_snapshots_writes_jsonl(tmp_path):
    path = tmp_path / "snaps.jsonl"
    mc.append_snapshots([{"ticker": "T1", "kalshi_p": 0.5}], path)
    mc.append_snapshots([{"ticker": "T2", "kalshi_p": 0.6}], path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["ticker"] == "T2"


def test_capture_orchestrates_fetchers(tmp_path):
    watchlist = [{"ticker": "T1", "game_id": "g1", "home_team_id": 1,
                  "away_team_id": 2, "game_date": "2026-01-01"}]
    path = tmp_path / "snaps.jsonl"
    n = mc.capture(
        watchlist, moment="t-60", asof="2026-01-01T00:00:00Z",
        fetch_kalshi_price=lambda ticker: 60,
        fetch_two_way_odds=lambda w: (-150, +130),
        path=path)
    assert n == 1
    row = json.loads(path.read_text().strip())
    assert abs(row["kalshi_p"] - 0.60) < 1e-9
    assert 0 < row["book_p"] < 1
