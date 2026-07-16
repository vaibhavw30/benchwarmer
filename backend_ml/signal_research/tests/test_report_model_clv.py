import json
from backend_ml.signal_research import report, config
from backend_ml.signal_research import model_clv
from backend_ml.signal_research.market_capture import ENTRY_MOMENT, CLOSING_MOMENT


def test_load_settlements_reads_and_defaults(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"T1": 1, "T2": 0}')
    assert report.load_settlements(p) == {"T1": 1, "T2": 0}
    assert report.load_settlements(tmp_path / "nope.json") == {}


def test_model_clv_report_wiring_end_to_end():
    # The CLI helpers (load_edge_params + model_clv_report) compose correctly.
    snaps = {"g1": {
        ENTRY_MOMENT: {"ticker": "T1", "kalshi_p": 0.50, "p_model": 0.70},
        CLOSING_MOMENT: {"ticker": "T1", "kalshi_p": 0.60, "p_model": 0.70}}}
    params = config.load_edge_params()
    out = model_clv.model_clv_report(snaps, {"T1": 1}, min_samples=1,
                                     signal_version="recalibrated", **params)
    assert out["insufficient"] is False
    assert out["raw_sign"]["n"] == 1
    assert out["beats_close"]["n_settled"] == 1
