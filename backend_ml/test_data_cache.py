"""Tests for data_engine.load_or_build_training_dataset cache-freshness logic."""
import pandas as pd
import pytest

import data_engine


@pytest.fixture
def fake_build(monkeypatch):
    calls = {"n": 0}

    def _build():
        calls["n"] += 1
        return pd.DataFrame({"GAME_DATE_H": ["2026-07-16"], "HOME_WIN": [1]})

    monkeypatch.setattr(data_engine, "build_training_dataset", _build)
    return calls


def _write_cache(path, newest_game_date):
    pd.DataFrame({"GAME_DATE_H": [newest_game_date]}).to_csv(path, index=False)


def test_fresh_cache_is_read_without_rebuild(tmp_path, fake_build):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, pd.Timestamp.now().strftime("%Y-%m-%d"))
    df = data_engine.load_or_build_training_dataset(cache_path=str(cache), max_age_days=3)
    assert fake_build["n"] == 0
    assert not df.empty


def test_stale_cache_triggers_rebuild(tmp_path, fake_build):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, "2026-04-12")  # months older than any sane max_age
    data_engine.load_or_build_training_dataset(cache_path=str(cache), max_age_days=3)
    assert fake_build["n"] == 1


def test_missing_cache_triggers_rebuild(tmp_path, fake_build):
    data_engine.load_or_build_training_dataset(
        cache_path=str(tmp_path / "absent.csv"), max_age_days=3)
    assert fake_build["n"] == 1


def test_force_refresh_rebuilds_even_when_fresh(tmp_path, fake_build, monkeypatch):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, pd.Timestamp.now().strftime("%Y-%m-%d"))
    monkeypatch.setenv("FORCE_REFRESH", "1")
    data_engine.load_or_build_training_dataset(cache_path=str(cache), max_age_days=3)
    assert fake_build["n"] == 1


def test_cache_exactly_at_threshold_is_still_fresh(tmp_path, fake_build):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, (pd.Timestamp.now() - pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
    data_engine.load_or_build_training_dataset(cache_path=str(cache), max_age_days=3)
    assert fake_build["n"] == 0
