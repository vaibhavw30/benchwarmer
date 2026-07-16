"""Tests for the nightly retrain orchestrator's pure logic."""
from datetime import date
import json
import os

import pytest

import scheduled_retrain as sr


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
