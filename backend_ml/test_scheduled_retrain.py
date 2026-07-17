"""Tests for the nightly retrain orchestrator's pure logic."""
from datetime import date
import json
import os

import pytest

import scheduled_retrain as sr

import joblib
import numpy as np
import pandas as pd
from signal_research import dataset as _ds


class _FakeXGB:
    """predict_proba -> P(home win) = 0.7 for every row."""
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


class _FakeRidge:
    def decision_function(self, X):
        return np.zeros(len(X))            # sigmoid(0) = 0.5


class _FakeScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def _drift_games(n, all_home_wins=True):
    """A minimal recompute-ready frame: FEATURES + id/date/outcome columns."""
    row = {f: 1.0 for f in _ds.FEATURES}
    df = pd.DataFrame([row] * n)
    df["GAME_ID"] = [f"g{i}" for i in range(n)]
    df["GAME_DATE_H"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(range(n), unit="D")
    df["TEAM_ID_H"] = 100
    df["TEAM_ID_A"] = 200
    df["HOME_WIN"] = 1 if all_home_wins else 0
    return df


def _write_live_artifacts(live_dir, xgb_w=0.5, ridge_w=0.5):
    joblib.dump(_FakeXGB(), str(live_dir / "xgboost_nba_model.pkl"))
    joblib.dump(_FakeRidge(), str(live_dir / "ridge_nba_model.pkl"))
    joblib.dump(_FakeScaler(), str(live_dir / "feature_scaler.pkl"))
    (live_dir / "ensemble_weights.json").write_text(
        json.dumps({"xgb_weight": xgb_w, "ridge_weight": ridge_w,
                    "test_accuracy": 0.66, "test_brier": 0.2}))


# --- in_season -------------------------------------------------------------

@pytest.mark.parametrize("d,expected", [
    (date(2026, 1, 15), True),    # mid-season, after the year wrap
    (date(2026, 12, 25), True),   # mid-season, before the year wrap
    (date(2026, 8, 1), False),    # deep offseason
    (date(2026, 10, 1), True),    # exact season start
    (date(2026, 6, 30), True),    # exact season end
    (date(2026, 7, 1), False),    # day after season end
    (date(2026, 9, 30), False),   # day before season start
])
def test_in_season(d, expected):
    assert sr.in_season(d) is expected


# --- should_deploy -----------------------------------------------------------

@pytest.mark.parametrize("new,cur,expected", [
    (0.67, 0.66, True),    # better
    (0.66, 0.66, True),    # equal
    (0.655, 0.66, True),   # worse but within TOLERANCE (0.01)
    (0.64, 0.66, False),   # worse beyond TOLERANCE
    (0.10, None, True),    # no prior model: always deploy
])
def test_should_deploy(new, cur, expected):
    assert sr.should_deploy(new, cur) is expected


# --- read_test_accuracy -------------------------------------------------------

def test_read_test_accuracy_happy_path(tmp_path):
    p = tmp_path / "w.json"
    p.write_text('{"test_accuracy": 0.66}')
    assert sr.read_test_accuracy(str(p)) == 0.66


@pytest.mark.parametrize("content", [
    None,                        # file absent
    "not json",                  # unparseable
    '{"xgb_weight": 0.7}',       # key missing (e.g. the re-tuned local file)
])
def test_read_test_accuracy_returns_none_on_problems(tmp_path, content):
    p = tmp_path / "w.json"
    if content is not None:
        p.write_text(content)
    assert sr.read_test_accuracy(str(p)) is None


# --- read_baseline_brier ------------------------------------------------------

def test_read_baseline_brier_happy_path(tmp_path):
    p = tmp_path / "w.json"
    p.write_text('{"test_accuracy": 0.66, "test_brier": 0.21}')
    assert sr.read_baseline_brier(str(p)) == 0.21


@pytest.mark.parametrize("content", [
    None,                                          # file absent
    "not json",                                    # unparseable
    '{"test_accuracy": 0.66}',                     # key missing (legacy model)
    '{"test_accuracy": 0.66, "test_brier": null}', # non-numeric
])
def test_read_baseline_brier_returns_none_on_problems(tmp_path, content):
    p = tmp_path / "w.json"
    if content is not None:
        p.write_text(content)
    assert sr.read_baseline_brier(str(p)) is None


# --- should_retrain -----------------------------------------------------------

@pytest.mark.parametrize("recent,baseline,n,expected", [
    (0.30, 0.20, 200, True),    # drift: recent worse than baseline+margin(0.02)
    (0.20, 0.20, 200, False),   # no drift: equal, within margin
    (0.215, 0.20, 200, False),  # no drift: worse but within margin (0.215 <= 0.22)
    (0.221, 0.20, 200, True),   # drift: just beyond margin (0.221 > 0.22)
    (0.10, 0.20, 50, True),     # low data: n_recent < MIN_RECENT overrides no-drift
    (None, 0.20, 200, True),    # measurement error: recent is None -> retrain
    (0.20, None, 200, True),    # no baseline yet -> retrain
])
def test_should_retrain(recent, baseline, n, expected):
    assert sr.should_retrain(recent, baseline, n) is expected


def test_should_retrain_boundary_exactly_at_margin_is_no_drift():
    # recent == baseline + margin exactly -> not strictly greater -> skip
    assert sr.should_retrain(0.22, 0.20, 200) is False


def test_should_retrain_non_finite_recent_brier_retrains():
    import math
    assert sr.should_retrain(math.nan, 0.20, 200) is True
    assert sr.should_retrain(math.inf, 0.20, 200) is True


def test_drift_constants_have_expected_values():
    assert sr.DRIFT_WINDOW == 150
    assert sr.MIN_RECENT == 100
    assert sr.DRIFT_MARGIN == 0.02


# --- measure_drift --------------------------------------------------------------

def test_measure_drift_returns_brier_and_count(tmp_path):
    _write_live_artifacts(tmp_path)
    df = _drift_games(200, all_home_wins=True)
    brier, n = sr.measure_drift(df, live_dir=str(tmp_path), window=150)
    # p = 0.5*0.7 + 0.5*0.5 = 0.6; outcome = 1 -> brier = (0.6-1)^2 = 0.16
    assert n == 150
    assert brier == pytest.approx(0.16, abs=1e-9)


def test_measure_drift_window_larger_than_rows_uses_all(tmp_path):
    _write_live_artifacts(tmp_path)
    df = _drift_games(5)
    brier, n = sr.measure_drift(df, live_dir=str(tmp_path), window=150)
    assert n == 5
    assert brier == pytest.approx(0.16, abs=1e-9)


def test_measure_drift_missing_artifacts_returns_none(tmp_path):
    # empty live_dir -> joblib.load raises -> caught -> (None, 0)
    df = _drift_games(10)
    assert sr.measure_drift(df, live_dir=str(tmp_path), window=150) == (None, 0)


# --- deploy_artifacts ----------------------------------------------------------

def _fill(dirpath, names, tag):
    for n in names:
        (dirpath / n).write_text(f"{tag}:{n}")


def test_deploy_overwrites_all_live_artifacts(tmp_path):
    temp, live = tmp_path / "temp", tmp_path / "live"
    temp.mkdir(); live.mkdir()
    _fill(temp, sr.ARTIFACTS, "new")
    _fill(live, sr.ARTIFACTS, "old")
    sr.deploy_artifacts(str(temp), str(live))
    for n in sr.ARTIFACTS:
        assert (live / n).read_text() == f"new:{n}"
        assert not (live / (n + ".new")).exists()   # staging files cleaned up


def test_deploy_leaves_live_untouched_when_a_source_is_missing(tmp_path):
    temp, live = tmp_path / "temp", tmp_path / "live"
    temp.mkdir(); live.mkdir()
    _fill(temp, sr.ARTIFACTS[:-1], "new")           # one artifact missing
    _fill(live, sr.ARTIFACTS, "old")
    with pytest.raises(FileNotFoundError):
        sr.deploy_artifacts(str(temp), str(live))
    for n in sr.ARTIFACTS:
        assert (live / n).read_text() == f"old:{n}"  # nothing renamed


# --- log_run --------------------------------------------------------------------

def test_log_run_appends_formatted_line(tmp_path):
    log = tmp_path / "log.txt"
    sr.log_run("deployed", 0.67, 0.66, log_path=str(log))
    sr.log_run("skipped: offseason", log_path=str(log))
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("deployed new_acc=0.6700 current_acc=0.6600")
    assert lines[1].endswith("skipped: offseason new_acc=none current_acc=none")


# --- main -----------------------------------------------------------------------

from datetime import date as _date


@pytest.fixture
def fake_trainer(tmp_path, monkeypatch):
    """Run main() in an isolated cwd with a fake trainer that writes 4 artifacts."""
    monkeypatch.chdir(tmp_path)

    # main() now refreshes the cache up front; stub it so tests never scrape.
    import data_engine
    monkeypatch.setattr(data_engine, "load_or_build_training_dataset",
                        lambda *a, **k: pd.DataFrame())

    state = {"called": 0, "new_accuracy": 0.70, "succeed": True}

    def _train(output_dir="."):
        state["called"] += 1
        if not state["succeed"]:
            return False
        for n in sr.ARTIFACTS[:-1]:
            with open(os.path.join(output_dir, n), "w") as f:
                f.write("model-bytes")
        with open(os.path.join(output_dir, "ensemble_weights.json"), "w") as f:
            json.dump({"test_accuracy": state["new_accuracy"]}, f)
        return True

    import train_model
    monkeypatch.setattr(train_model, "train_and_optimize_model", _train)
    return state


IN_SEASON = _date(2026, 1, 15)
OFFSEASON = _date(2026, 8, 1)


def test_main_skips_in_offseason(fake_trainer, tmp_path):
    assert sr.main(today=OFFSEASON) == 0
    assert fake_trainer["called"] == 0
    assert "skipped: offseason" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_deploys_when_no_prior_model(fake_trainer, tmp_path):
    assert sr.main(today=IN_SEASON) == 0
    for n in sr.ARTIFACTS:
        assert (tmp_path / n).exists()
    assert "deployed new_acc=0.7000 current_acc=none" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_rejects_worse_model_and_keeps_live_files(fake_trainer, tmp_path):
    (tmp_path / "ensemble_weights.json").write_text('{"test_accuracy": 0.80}')
    (tmp_path / sr.ARTIFACTS[0]).write_text("live-model")
    fake_trainer["new_accuracy"] = 0.70    # 10 points worse: beyond tolerance
    assert sr.main(today=IN_SEASON) == 0
    assert (tmp_path / sr.ARTIFACTS[0]).read_text() == "live-model"
    assert "rejected new_acc=0.7000 current_acc=0.8000" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_logs_failure_and_exits_nonzero_when_trainer_fails(fake_trainer, tmp_path):
    fake_trainer["succeed"] = False
    assert sr.main(today=IN_SEASON) == 1
    assert "failed:" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_logs_failure_on_exception(fake_trainer, tmp_path, monkeypatch):
    import train_model

    def _boom(output_dir="."):
        raise RuntimeError("nba_api down")

    monkeypatch.setattr(train_model, "train_and_optimize_model", _boom)
    assert sr.main(today=IN_SEASON) == 1
    assert "failed: RuntimeError('nba_api down')" in (tmp_path / sr.LOG_PATH).read_text()


def test_main_logs_failure_when_mkdtemp_raises(fake_trainer, tmp_path, monkeypatch):
    import tempfile

    def _boom(prefix=None):
        raise OSError("disk full")

    monkeypatch.setattr(tempfile, "mkdtemp", _boom)
    assert sr.main(today=IN_SEASON) == 1
    assert "failed:" in (tmp_path / sr.LOG_PATH).read_text()


# --- main: drift gate ---------------------------------------------------------

def _write_live_weights(tmp_path, test_brier):
    (tmp_path / "ensemble_weights.json").write_text(
        json.dumps({"test_accuracy": 0.66, "test_brier": test_brier}))


def test_main_skips_when_no_drift(fake_trainer, tmp_path, monkeypatch):
    _write_live_weights(tmp_path, test_brier=0.20)
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (0.20, 200))
    assert sr.main(today=IN_SEASON) == 0
    assert fake_trainer["called"] == 0                       # trainer NOT run
    log = (tmp_path / sr.LOG_PATH).read_text()
    assert "skipped: no drift" in log
    assert "new_brier=0.2000 baseline_brier=0.2000" in log


def test_main_retrains_when_drifted(fake_trainer, tmp_path, monkeypatch):
    _write_live_weights(tmp_path, test_brier=0.20)
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (0.30, 200))
    assert sr.main(today=IN_SEASON) == 0
    assert fake_trainer["called"] == 1                       # trainer ran


def test_main_retrains_on_low_recent_data(fake_trainer, tmp_path, monkeypatch):
    _write_live_weights(tmp_path, test_brier=0.20)
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (0.10, 50))  # n < MIN_RECENT
    assert sr.main(today=IN_SEASON) == 0
    assert fake_trainer["called"] == 1


def test_main_retrains_on_measure_error(fake_trainer, tmp_path, monkeypatch):
    _write_live_weights(tmp_path, test_brier=0.20)
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (None, 0))
    assert sr.main(today=IN_SEASON) == 0
    assert fake_trainer["called"] == 1                       # fallback -> trainer ran


def test_main_pops_force_refresh_before_training(fake_trainer, tmp_path, monkeypatch):
    monkeypatch.setenv("FORCE_REFRESH", "1")
    monkeypatch.setattr(sr, "measure_drift", lambda *a, **k: (0.30, 200))  # force drift
    sr.main(today=IN_SEASON)
    assert "FORCE_REFRESH" not in os.environ                 # popped by main()
