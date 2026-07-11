#!/usr/bin/env bash
#
# Offseason-safe offline paper-session demo.
#
# Publishes RAW vs RECALIBRATED fair values for a sample favorite through the
# real publisher (publish_fair_values.build_fair_values), then replays the
# recorded orderbook fixture against the PaperVenue trader (paper_session) with
# each — showing how RECALIBRATE=1 shifts the paper trader's decisions. No
# network, no Kalshi credentials, no live NBA slate required.
#
# In season, the live equivalent is:
#     RECALIBRATE=1 python -m backend_ml.publish_fair_values   # -> fair_values.json
#     ./trading_engine/build/te_engine                         # live WS + PaperVenue
#
#   usage: ./run_paper.sh [raw_p_yes]      (default 0.62; try 0.75 for a strong favorite)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
DRIVER="$ROOT/trading_engine/build/paper_session"
FIXTURE="$ROOT/trading_engine/tests/fixtures/replay_sample.jsonl"
RECAL="$ROOT/backend_ml/signal_research/artifacts/recalibrator.json"
TICKER="T"
P_RAW="${1:-0.62}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- guards: fail with the exact command that fixes each prerequisite ---
[ -x "$PY" ] || { echo "ERROR: venv missing at $PY"; echo "  fix: python3 -m venv .venv && .venv/bin/pip install -r backend_ml/requirements.txt"; exit 1; }
[ -x "$DRIVER" ] || { echo "ERROR: paper_session not built at $DRIVER"; echo "  fix: cmake -S trading_engine -B trading_engine/build && cmake --build trading_engine/build --target paper_session"; exit 1; }
[ -f "$RECAL" ] || { echo "ERROR: no recalibrator at $RECAL"; echo "  fix: $PY -m backend_ml.signal_research.report evaluate   (needs trained models)"; exit 1; }

# --- publish RAW and RECALIBRATED fair values for ticker $TICKER ---
# Exercises the real publisher. asof is stamped to the fixture's fixed replay
# clock (2026-07-10T00:00:00Z) so the value is fresh under the driver's
# deterministic now_ms (that + 60s), instead of wall-clock "now".
echo "=== fair value (raw model p_yes=$P_RAW -> recalibrated) ==="
"$PY" - "$P_RAW" "$TICKER" "$TMP/fv_raw.json" "$TMP/fv_recal.json" "$RECAL" <<'PYEOF'
import json, sys
from backend_ml.publish_fair_values import build_fair_values
from backend_ml.signal_research.recalibration import Recalibrator

p_raw, ticker, out_raw, out_recal, recal_path = sys.argv[1:6]
p_raw = float(p_raw)
pred = [{"home_team_id": 1, "away_team_id": 2, "date": "2026-07-10",
         "home_win_probability": p_raw, "confidence_score": max(p_raw, 1 - p_raw),
         "game_id": "DEMO"}]
wl = [{"home_team_id": 1, "away_team_id": 2, "game_date": "2026-07-10", "ticker": ticker}]

def stamp(rows):
    for r in rows:
        r["asof"] = "2026-07-10T00:00:00Z"   # match the fixture's fixed replay clock
    return rows

raw = stamp(build_fair_values(pred, wl))                                        # RECALIBRATE off
recal = stamp(build_fair_values(pred, wl, recalibrator=Recalibrator.load(recal_path)))  # RECALIBRATE=1
json.dump(raw, open(out_raw, "w"))
json.dump(recal, open(out_recal, "w"))
print(f"  RAW        p_yes={raw[0]['p_yes']:.4f}  confidence={raw[0]['confidence']:.4f}")
print(f"  RECALIBRATED p_yes={recal[0]['p_yes']:.4f}  confidence={recal[0]['confidence']:.4f}")
PYEOF

echo
echo "=== PAPER SESSION: RAW fair value (RECALIBRATE off) ==="
"$DRIVER" "$TMP/fv_raw.json" "$FIXTURE" "$TICKER"
echo
echo "=== PAPER SESSION: RECALIBRATED fair value (RECALIBRATE=1) ==="
"$DRIVER" "$TMP/fv_recal.json" "$FIXTURE" "$TICKER"
echo
echo "Both run on the same recorded book against PaperVenue — zero real orders."
echo "Compare the 'quote' bid/ask: the recalibrated (more bullish) fair value"
echo "leans the maker quotes up on this favorite."
