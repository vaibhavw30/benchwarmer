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
