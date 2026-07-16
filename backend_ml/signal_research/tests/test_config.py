import json
from backend_ml.signal_research import config


def test_load_edge_params_reads_engine_json(tmp_path):
    ej = tmp_path / "engine.json"
    ej.write_text(json.dumps({"base_edge_cents": 2, "confidence_k": 8.0,
                              "fee_cents_per_contract": 1}))
    assert config.load_edge_params(ej) == {
        "base_edge_cents": 2, "fee_cents": 1, "confidence_k": 8.0}


def test_load_edge_params_defaults_when_missing_file(tmp_path):
    assert config.load_edge_params(tmp_path / "nope.json") == {
        "base_edge_cents": 2, "fee_cents": 1, "confidence_k": 8.0}


def test_load_edge_params_partial_falls_back_per_key(tmp_path):
    ej = tmp_path / "engine.json"
    ej.write_text(json.dumps({"base_edge_cents": 5}))   # others missing
    p = config.load_edge_params(ej)
    assert p == {"base_edge_cents": 5, "fee_cents": 1, "confidence_k": 8.0}
