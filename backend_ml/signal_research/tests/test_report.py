import json
import numpy as np
import pandas as pd
from backend_ml.signal_research import report, config


def test_load_fee_cents_reads_engine_json(tmp_path):
    ej = tmp_path / "engine.json"
    ej.write_text(json.dumps({"fee_cents_per_contract": 3}))
    assert config.load_fee_cents(ej) == 3


def test_load_fee_cents_fallback(tmp_path):
    assert config.load_fee_cents(tmp_path / "missing.json") == 1


def test_run_evaluate_reports_and_saves_artifact(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p ** 2).astype(int)   # miscalibrated
    df = pd.DataFrame({"p_model": p, "outcome": y})
    art = tmp_path / "recal.json"
    out = report.run_evaluate(df, method="isotonic", artifact_path=art, min_n=200)
    assert out["calibration"]["n"] == 2000
    assert out["recalibration"]["brier_recal"] < out["recalibration"]["brier_raw"]
    assert art.exists()


def test_run_evaluate_refuses_small_sample(tmp_path):
    df = pd.DataFrame({"p_model": [0.5, 0.6], "outcome": [1, 0]})
    out = report.run_evaluate(df, artifact_path=tmp_path / "r.json", min_n=200)
    assert out["insufficient"] is True
    assert not (tmp_path / "r.json").exists()


def test_load_snapshots_groups_by_game(tmp_path):
    path = tmp_path / "snaps.jsonl"
    path.write_text(
        json.dumps({"game_id": "g1", "moment": "t-60", "kalshi_p": 0.4}) + "\n" +
        json.dumps({"game_id": "g1", "moment": "tipoff", "kalshi_p": 0.46}) + "\n")
    snaps = report.load_snapshots(path)
    assert set(snaps["g1"]) == {"t-60", "tipoff"}
    assert snaps["g1"]["tipoff"]["kalshi_p"] == 0.46
