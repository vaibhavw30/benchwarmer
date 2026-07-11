from backend_ml.publish_fair_values import build_fair_values
from backend_ml.signal.recalibration import Recalibrator


PRED = [{"home_team_id": 1, "away_team_id": 2, "date": "2026-01-01",
         "home_win_probability": 0.80, "confidence_score": 0.80, "game_id": "g1"}]
WL = [{"home_team_id": 1, "away_team_id": 2, "game_date": "2026-01-01",
       "ticker": "T1"}]


def test_flag_off_is_unchanged():
    rows = build_fair_values(PRED, WL)                 # no recalibrator
    assert rows[0]["p_yes"] == 0.80
    assert rows[0]["confidence"] == 0.80


def test_recalibrator_transforms_and_recomputes_confidence():
    # identity-ish isotonic that maps 0.80 -> 0.60
    r = Recalibrator("isotonic", {"x": [0.0, 0.8, 1.0], "y": [0.0, 0.60, 1.0]})
    rows = build_fair_values(PRED, WL, recalibrator=r)
    assert abs(rows[0]["p_yes"] - 0.60) < 1e-9
    assert abs(rows[0]["confidence"] - 0.60) < 1e-9    # max(0.6, 0.4)


def test_recalibrated_confidence_uses_max_of_p_and_complement():
    r = Recalibrator("isotonic", {"x": [0.0, 0.8, 1.0], "y": [0.0, 0.30, 1.0]})
    rows = build_fair_values(PRED, WL, recalibrator=r)
    assert abs(rows[0]["p_yes"] - 0.30) < 1e-9
    assert abs(rows[0]["confidence"] - 0.70) < 1e-9    # max(0.3, 0.7)
