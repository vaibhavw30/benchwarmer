"""Single-source config reads for the signal harness."""
import json
from pathlib import Path

DEFAULT_ENGINE_JSON = "trading_engine/config/engine.json"


def load_fee_cents(engine_json_path=DEFAULT_ENGINE_JSON) -> int:
    try:
        data = json.loads(Path(engine_json_path).read_text())
        return int(data["fee_cents_per_contract"])
    except Exception:
        return 1


def load_edge_params(engine_json_path=DEFAULT_ENGINE_JSON) -> dict:
    """Edge-threshold constants, single-sourced from engine.json.

    Returns {base_edge_cents, fee_cents, confidence_k}; each key falls back to
    the v1 engine default if the file or key is missing.
    """
    defaults = {"base_edge_cents": 2, "fee_cents": 1, "confidence_k": 8.0}
    try:
        data = json.loads(Path(engine_json_path).read_text())
    except Exception:
        return dict(defaults)
    return {
        "base_edge_cents": int(data.get("base_edge_cents", defaults["base_edge_cents"])),
        "fee_cents": int(data.get("fee_cents_per_contract", defaults["fee_cents"])),
        "confidence_k": float(data.get("confidence_k", defaults["confidence_k"])),
    }
