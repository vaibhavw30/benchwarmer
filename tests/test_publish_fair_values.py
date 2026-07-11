from backend_ml.publish_fair_values import build_fair_values


def test_maps_prediction_to_ticker_by_teams_and_date():
    preds = [{"game_id": "0022500123", "date": "2026-07-10",
              "home_team_id": 1610612744, "away_team_id": 1610612747,
              "home_win_probability": 0.62, "confidence_score": 0.62}]
    watchlist = [{"ticker": "KXNBA-26JUL10-GSWLAL",
                  "home_team_id": 1610612744, "away_team_id": 1610612747,
                  "game_date": "2026-07-10"}]
    rows = build_fair_values(preds, watchlist)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "KXNBA-26JUL10-GSWLAL"
    assert abs(rows[0]["p_yes"] - 0.62) < 1e-9
    assert rows[0]["confidence"] == 0.62


def test_skips_unmapped_game():
    preds = [{"game_id": "x", "date": "2026-07-10", "home_team_id": 1, "away_team_id": 2,
              "home_win_probability": 0.5, "confidence_score": 0.5}]
    assert build_fair_values(preds, watchlist=[]) == []


def test_clamps_out_of_range_confidence_and_p_yes():
    preds = [{"game_id": "0022500124", "date": "2026-07-10",
              "home_team_id": 1610612744, "away_team_id": 1610612747,
              "home_win_probability": 1.5, "confidence_score": 1.5}]
    watchlist = [{"ticker": "KXNBA-26JUL10-GSWLAL",
                  "home_team_id": 1610612744, "away_team_id": 1610612747,
                  "game_date": "2026-07-10"}]
    rows = build_fair_values(preds, watchlist)
    assert len(rows) == 1
    assert rows[0]["p_yes"] == 1.0
    assert rows[0]["confidence"] == 1.0
