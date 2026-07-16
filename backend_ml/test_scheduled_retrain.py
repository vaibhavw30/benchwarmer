"""Tests for the nightly retrain orchestrator's pure logic."""
from datetime import date

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
