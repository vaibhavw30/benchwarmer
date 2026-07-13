# Model-vs-Close CLV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `signal_research` forward-capture pipeline to carry the model's own probability (`p_yes` from the published `fair_values.json`) into snapshots, and add the model-vs-market metric suite: model CLV, model-beats-close (Brier vs closing line), and fee-aware entry edge — in both raw-sign and would-trade buckets.

**Architecture:** Four focused touch points in the existing `backend_ml/signal_research/` package: extend `config.py` (edge-threshold constants), extend `market_capture.py` (`p_model` in snapshot rows), add pure `model_clv.py` (the metric suite, reusing `clv.clv_cents`), and extend the `report.py` CLI (`model-clv-report`). Pure metric code is fully unit-tested on synthetic data; the live Kalshi settlement call is injected and deferred.

**Tech Stack:** Python 3, pytest. No new deps. Reuses `backend_ml/signal_research/clv.py`, `market_capture.py`, and `trading_engine/config/engine.json`.

## Global Constraints

- **Package location:** all new/changed code under `backend_ml/signal_research/`; tests under `backend_ml/signal_research/tests/`. Run tests from repo root with `python -m pytest`.
- **Edge threshold mirrors the C++ engine verbatim:** `edge_threshold_cents(conf) = floor(base_edge_cents + fee_cents_per_contract + confidence_k * (1 - conf))`. Known values (base_edge 2, fee 1, k 8.0): `conf 0.95 → 3`, `conf 0.55 → 6`. A parity test locks these.
- **Constants single-sourced from `engine.json`:** `base_edge_cents` (2), `confidence_k` (8.0), `fee_cents_per_contract` (1). Never hard-code a second copy; read via `config.py`.
- **Confidence is derived, not stored:** `confidence = max(p_model, 1 - p_model)` — exactly how the publisher sets it for both raw and recalibrated values. The snapshot schema gets no separate `confidence` field.
- **Forward-accruing:** `model_clv_report` refuses to summarize below `min_samples` (both buckets `insufficient=True`, `None` means); the beats-close block is `None` until games settle.
- **`p_model` provenance:** whatever `fair_values.json` held at capture (raw or recalibrated per `RECALIBRATE`). The report records `signal_version`.
- **No real keys/models in tests.** The live Kalshi settlement fetch is injected/deferred, like the existing price/odds fetchers.
- **Reuse, don't reinvent:** `clv.clv_cents` (side-signed drift) and the `ENTRY_MOMENT`/`CLOSING_MOMENT` constants from `market_capture.py`.

---

### Task 1: Edge-threshold constants in config

**Files:**
- Modify: `backend_ml/signal_research/config.py`
- Test: `backend_ml/signal_research/tests/test_config.py`

**Interfaces:**
- Consumes: `trading_engine/config/engine.json`.
- Produces: `load_edge_params(engine_json_path=DEFAULT_ENGINE_JSON) -> dict` with keys `base_edge_cents: int`, `fee_cents: int`, `confidence_k: float`. Missing file or keys fall back to `{2, 1, 8.0}`. Existing `load_fee_cents` is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_config.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'load_edge_params'`

- [ ] **Step 3: Write minimal implementation**

Append to `backend_ml/signal_research/config.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/config.py backend_ml/signal_research/tests/test_config.py
git commit -m "feat(signal): load_edge_params single-sources edge-threshold constants"
```

---

### Task 2: Model-vs-market metrics (pure TDD core)

**Files:**
- Create: `backend_ml/signal_research/model_clv.py`
- Test: `backend_ml/signal_research/tests/test_model_clv.py`

**Interfaces:**
- Consumes: `clv.clv_cents`, `market_capture.ENTRY_MOMENT`/`CLOSING_MOMENT`.
- Produces:
  - `edge_threshold_cents(confidence, base_edge_cents, fee_cents, confidence_k) -> int`
  - `model_side(p_model, kalshi_entry_p) -> "YES"|"NO"`
  - `entry_edge_cents(p_model, kalshi_entry_p, fee_cents) -> float`
  - `would_trade(p_model, kalshi_entry_p, base_edge_cents, fee_cents, confidence_k) -> bool`
  - `model_clv_report(snapshots_by_game, settlements, *, base_edge_cents, fee_cents, confidence_k, min_samples, signal_version="unknown") -> dict` with keys `signal_version, insufficient, raw_sign, would_trade, beats_close`. `raw_sign`/`would_trade` each = `{n, mean_clv_cents, pct_positive_clv, mean_entry_edge_cents}`. `beats_close` = `{n_settled, brier_model, brier_close, brier_delta, pct_model_beats_close}` or `None`. `settlements` is `{ticker: outcome∈{0,1}}`.

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_model_clv.py
from backend_ml.signal_research import model_clv as mc
from backend_ml.signal_research.market_capture import ENTRY_MOMENT, CLOSING_MOMENT

EDGE = {"base_edge_cents": 2, "fee_cents": 1, "confidence_k": 8.0}


def test_edge_threshold_parity_with_cpp_engine():
    # Locks the Python formula to the C++ engine's known values.
    assert mc.edge_threshold_cents(0.95, 2, 1, 8.0) == 3
    assert mc.edge_threshold_cents(0.55, 2, 1, 8.0) == 6


def test_model_side_sign():
    assert mc.model_side(0.62, 0.48) == "YES"
    assert mc.model_side(0.30, 0.48) == "NO"


def test_entry_edge_after_fee_can_go_negative():
    assert abs(mc.entry_edge_cents(0.62, 0.48, 1) - 13.0) < 1e-9   # 14c - 1 fee
    assert mc.entry_edge_cents(0.505, 0.50, 1) < 0                 # 0.5c - 1 fee


def test_would_trade_threshold():
    # p=0.62 vs 0.48: conf 0.62 -> thresh floor(2+1+8*0.38)=6; divergence 14 >= 6
    assert mc.would_trade(0.62, 0.48, 2, 1, 8.0) is True
    # p=0.52 vs 0.50: conf 0.52 -> thresh floor(2+1+8*0.48)=6; divergence 2 < 6
    assert mc.would_trade(0.52, 0.50, 2, 1, 8.0) is False


def _game(ticker, p_model, k_entry, k_close):
    return {
        ENTRY_MOMENT: {"ticker": ticker, "kalshi_p": k_entry, "p_model": p_model},
        CLOSING_MOMENT: {"ticker": ticker, "kalshi_p": k_close, "p_model": p_model},
    }


def test_report_insufficient_below_min():
    snaps = {"g1": _game("T1", 0.62, 0.48, 0.55)}
    out = mc.model_clv_report(snaps, {}, min_samples=5, **EDGE)
    assert out["insufficient"] is True
    assert out["raw_sign"]["mean_clv_cents"] is None
    assert out["beats_close"] is None


def test_report_clv_and_buckets():
    # Two would-trade games (big divergence) + one marginal (no trade).
    snaps = {
        "g1": _game("T1", 0.62, 0.48, 0.55),   # YES, close 48->55 => +7 clv, trades
        "g2": _game("T2", 0.70, 0.50, 0.60),   # YES, close 50->60 => +10 clv, trades
        "g3": _game("T3", 0.52, 0.50, 0.49),   # YES, close 50->49 => -1 clv, no-trade
    }
    out = mc.model_clv_report(snaps, {}, min_samples=1, signal_version="recalibrated", **EDGE)
    assert out["insufficient"] is False
    assert out["signal_version"] == "recalibrated"
    assert out["raw_sign"]["n"] == 3
    assert abs(out["raw_sign"]["mean_clv_cents"] - (7 + 10 - 1) / 3) < 1e-9
    assert out["would_trade"]["n"] == 2                      # g1, g2 only
    assert abs(out["would_trade"]["mean_clv_cents"] - (7 + 10) / 2) < 1e-9
    assert out["would_trade"]["pct_positive_clv"] == 1.0
    assert out["beats_close"] is None                        # no settlements


def test_report_beats_close_with_settlement():
    snaps = {"g1": _game("T1", 0.70, 0.50, 0.60)}
    # home won -> outcome 1. brier_model=(0.7-1)^2=0.09 < brier_close=(0.6-1)^2=0.16
    out = mc.model_clv_report(snaps, {"T1": 1}, min_samples=1, **EDGE)
    bc = out["beats_close"]
    assert bc["n_settled"] == 1
    assert abs(bc["brier_model"] - 0.09) < 1e-9
    assert abs(bc["brier_close"] - 0.16) < 1e-9
    assert bc["brier_delta"] < 0
    assert bc["pct_model_beats_close"] == 1.0


def test_report_skips_games_without_p_model_or_a_leg():
    snaps = {
        "g1": {ENTRY_MOMENT: {"ticker": "T1", "kalshi_p": 0.48, "p_model": None},
               CLOSING_MOMENT: {"ticker": "T1", "kalshi_p": 0.55, "p_model": None}},
        "g2": {ENTRY_MOMENT: {"ticker": "T2", "kalshi_p": 0.48, "p_model": 0.62}},  # no close
    }
    out = mc.model_clv_report(snaps, {}, min_samples=1, **EDGE)
    assert out["insufficient"] is True    # zero usable games
    assert out["raw_sign"]["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_model_clv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research.model_clv'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend_ml/signal_research/model_clv.py
"""Model-vs-market CLV metrics (pure).

Brings the model's own probability (p_model, captured from the published
fair_values.json) into the CLV analysis:
  - model CLV: when the model takes a side at entry, does the close move toward us
  - entry edge: |p_model - kalshi_entry| after fee
  - model beats close: Brier(p_model) vs Brier(closing kalshi price) at settlement

Reuses clv.clv_cents for side-signed price drift. The edge threshold mirrors the
C++ engine (trading_engine/src/strategy/pricing.cpp); test_edge_threshold_parity
locks the formula to the engine's known values.
"""
import math

from backend_ml.signal_research.clv import clv_cents
from backend_ml.signal_research.market_capture import ENTRY_MOMENT, CLOSING_MOMENT


def edge_threshold_cents(confidence, base_edge_cents, fee_cents, confidence_k) -> int:
    return int(math.floor(base_edge_cents + fee_cents + confidence_k * (1.0 - confidence)))


def model_side(p_model, kalshi_entry_p) -> str:
    return "YES" if p_model > kalshi_entry_p else "NO"


def entry_edge_cents(p_model, kalshi_entry_p, fee_cents) -> float:
    return abs(p_model - kalshi_entry_p) * 100.0 - fee_cents


def would_trade(p_model, kalshi_entry_p, base_edge_cents, fee_cents, confidence_k) -> bool:
    confidence = max(p_model, 1.0 - p_model)
    threshold = edge_threshold_cents(confidence, base_edge_cents, fee_cents, confidence_k)
    divergence_cents = abs(p_model - kalshi_entry_p) * 100.0
    return divergence_cents >= threshold


def _bucket(clvs, edges) -> dict:
    n = len(clvs)
    if n == 0:
        return {"n": 0, "mean_clv_cents": None, "pct_positive_clv": None,
                "mean_entry_edge_cents": None}
    return {
        "n": n,
        "mean_clv_cents": sum(clvs) / n,
        "pct_positive_clv": sum(1 for c in clvs if c > 0) / n,
        "mean_entry_edge_cents": sum(edges) / n,
    }


def model_clv_report(snapshots_by_game, settlements, *, base_edge_cents, fee_cents,
                     confidence_k, min_samples, signal_version="unknown") -> dict:
    raw_clv, raw_edge = [], []
    wt_clv, wt_edge = [], []
    brier_model, brier_close, model_wins = [], [], 0

    for _game, legs in snapshots_by_game.items():
        entry = legs.get(ENTRY_MOMENT)
        closing = legs.get(CLOSING_MOMENT)
        if entry is None or closing is None:
            continue
        pm = entry.get("p_model")
        if pm is None:
            continue
        ke, kc = entry["kalshi_p"], closing["kalshi_p"]
        side = model_side(pm, ke)
        clv = clv_cents(ke * 100.0, kc * 100.0, side)
        edge = entry_edge_cents(pm, ke, fee_cents)
        raw_clv.append(clv)
        raw_edge.append(edge)
        if would_trade(pm, ke, base_edge_cents, fee_cents, confidence_k):
            wt_clv.append(clv)
            wt_edge.append(edge)
        outcome = settlements.get(entry["ticker"])
        if outcome is not None:
            bm = (pm - outcome) ** 2
            bc = (kc - outcome) ** 2
            brier_model.append(bm)
            brier_close.append(bc)
            if bm < bc:
                model_wins += 1

    n = len(raw_clv)
    if n < min_samples or n == 0:
        return {
            "signal_version": signal_version,
            "insufficient": True,
            "raw_sign": {"n": n, "mean_clv_cents": None, "pct_positive_clv": None,
                         "mean_entry_edge_cents": None},
            "would_trade": {"n": len(wt_clv), "mean_clv_cents": None,
                            "pct_positive_clv": None, "mean_entry_edge_cents": None},
            "beats_close": None,
        }

    beats = None
    ns = len(brier_model)
    if ns > 0:
        bm_mean = sum(brier_model) / ns
        bc_mean = sum(brier_close) / ns
        beats = {
            "n_settled": ns,
            "brier_model": bm_mean,
            "brier_close": bc_mean,
            "brier_delta": bm_mean - bc_mean,
            "pct_model_beats_close": model_wins / ns,
        }

    return {
        "signal_version": signal_version,
        "insufficient": False,
        "raw_sign": _bucket(raw_clv, raw_edge),
        "would_trade": _bucket(wt_clv, wt_edge),
        "beats_close": beats,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_model_clv.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/model_clv.py backend_ml/signal_research/tests/test_model_clv.py
git commit -m "feat(signal): model-vs-market CLV metrics (CLV, beats-close, entry edge)"
```

---

### Task 3: Carry p_model into market snapshots

**Files:**
- Modify: `backend_ml/signal_research/market_capture.py`
- Test: `backend_ml/signal_research/tests/test_market_capture.py` (add cases; do not remove existing)

**Interfaces:**
- Consumes: `fair_values.json` (list of `{ticker, p_yes, ...}`).
- Produces:
  - `load_fair_values(path) -> {ticker: p_yes}` (`{}` if file absent).
  - `build_snapshot_rows(watchlist, moment, kalshi_prices, book_odds, asof, fair_values=None)` — new trailing `fair_values` param; each row gains `"p_model": fair_values.get(ticker)` (`None` if absent). Existing 5-positional-arg callers keep working (`p_model` becomes `None`).
  - `capture(..., path, fair_values=None)` — new trailing `fair_values` param, threaded into `build_snapshot_rows`.

- [ ] **Step 1: Write the failing test**

Add to `backend_ml/signal_research/tests/test_market_capture.py`:

```python
def test_load_fair_values_maps_ticker_to_p_yes(tmp_path):
    from backend_ml.signal_research import market_capture as mc
    fv = tmp_path / "fv.json"
    fv.write_text('[{"ticker":"T1","p_yes":0.62,"confidence":0.62},'
                  ' {"ticker":"T2","p_yes":0.4,"confidence":0.6}]')
    assert mc.load_fair_values(fv) == {"T1": 0.62, "T2": 0.4}


def test_load_fair_values_missing_file_is_empty(tmp_path):
    from backend_ml.signal_research import market_capture as mc
    assert mc.load_fair_values(tmp_path / "nope.json") == {}


def test_build_snapshot_rows_stamps_p_model():
    from backend_ml.signal_research import market_capture as mc
    wl = [{"ticker": "T1", "game_id": "g1", "home_team_id": 1, "away_team_id": 2,
           "game_date": "2026-01-01"}]
    rows = mc.build_snapshot_rows(wl, "t-60", {"T1": 55}, {"T1": (-110, -110)},
                                  "2026-01-01T00:00:00Z", fair_values={"T1": 0.62})
    assert rows[0]["p_model"] == 0.62


def test_build_snapshot_rows_p_model_none_when_ticker_absent():
    from backend_ml.signal_research import market_capture as mc
    wl = [{"ticker": "T1", "game_id": "g1", "home_team_id": 1, "away_team_id": 2,
           "game_date": "2026-01-01"}]
    # no fair_values passed -> p_model None, existing behavior otherwise intact
    rows = mc.build_snapshot_rows(wl, "t-60", {"T1": 55}, {"T1": (-110, -110)},
                                  "2026-01-01T00:00:00Z")
    assert rows[0]["p_model"] is None
    assert rows[0]["kalshi_p"] == 0.55        # unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_market_capture.py -v`
Expected: FAIL — `AttributeError: ... 'load_fair_values'` and `KeyError: 'p_model'`

- [ ] **Step 3: Write minimal implementation**

In `backend_ml/signal_research/market_capture.py`, add `load_fair_values` (after `devig`) and thread `fair_values` through `build_snapshot_rows` and `capture`:

```python
def load_fair_values(path):
    """Map ticker -> published p_yes from a fair_values.json list. {} if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    rows = json.loads(p.read_text())
    return {r["ticker"]: float(r["p_yes"]) for r in rows}
```

Change `build_snapshot_rows` to:

```python
def build_snapshot_rows(watchlist, moment, kalshi_prices, book_odds, asof,
                        fair_values=None):
    fair_values = fair_values or {}
    rows = []
    for w in watchlist:
        ticker = w["ticker"]
        cents = kalshi_prices.get(ticker)
        odds = book_odds.get(ticker)
        if cents is None or odds is None:
            continue                      # fail-closed: incomplete market data
        book_p, _ = devig(odds[0], odds[1])
        rows.append({
            "ticker": ticker,
            "game_id": w["game_id"],
            "moment": moment,
            "kalshi_p": float(cents) / 100.0,
            "book_p": book_p,
            "p_model": fair_values.get(ticker),   # None if ticker absent
            "asof": asof,
        })
    return rows
```

Change `capture` to thread `fair_values`:

```python
def capture(watchlist, moment, asof, fetch_kalshi_price, fetch_two_way_odds, path,
            fair_values=None):
    kalshi_prices, book_odds = {}, {}
    for w in watchlist:
        ticker = w["ticker"]
        try:
            kalshi_prices[ticker] = fetch_kalshi_price(ticker)
            book_odds[ticker] = fetch_two_way_odds(w)
        except Exception:
            continue                      # skip this ticker, keep the rest
    rows = build_snapshot_rows(watchlist, moment, kalshi_prices, book_odds, asof,
                               fair_values)
    append_snapshots(rows, path)
    return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_market_capture.py -v`
Expected: PASS (all prior tests + 4 new; the prior tests are unaffected because `fair_values` defaults to `{}` and they never assert the row's full key set)

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/market_capture.py backend_ml/signal_research/tests/test_market_capture.py
git commit -m "feat(signal): capture p_model (published fair value) into market snapshots"
```

---

### Task 4: model-clv-report CLI + settlement loader + checklist

**Files:**
- Modify: `backend_ml/signal_research/report.py`
- Modify: `docs/superpowers/before-live-checklist.md`
- Test: `backend_ml/signal_research/tests/test_report_model_clv.py`

**Interfaces:**
- Consumes: `config.load_edge_params`, `report.load_snapshots`, `model_clv.model_clv_report`.
- Produces:
  - `load_settlements(path=SETTLEMENTS_PATH) -> {ticker: int}` (`{}` if absent).
  - `_cmd_model_clv_report(args)` and a `model-clv-report` subcommand with `--min-samples` (default 30) and `--signal-version` (`raw|recalibrated|unknown`, default `unknown`).

- [ ] **Step 1: Write the failing test**

```python
# backend_ml/signal_research/tests/test_report_model_clv.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend_ml/signal_research/tests/test_report_model_clv.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'load_settlements'`

- [ ] **Step 3: Write minimal implementation**

In `backend_ml/signal_research/report.py`, add near the other path constants:

```python
SETTLEMENTS_PATH = os.getenv("SETTLEMENTS_PATH",
                             "backend_ml/signal_research/artifacts/settlements.json")


def load_settlements(path=SETTLEMENTS_PATH) -> dict:
    """Map ticker -> outcome (1 = YES/home won, 0). {} if the file is absent.

    Populated live by the deferred Kalshi settlement fetch (see
    before-live-checklist). The file format is a JSON object {ticker: 0|1}.
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {}
    return {k: int(v) for k, v in json.loads(p.read_text()).items()}
```

Add the command handler (near `_cmd_clv_report`):

```python
def _cmd_model_clv_report(args):
    from backend_ml.signal_research import model_clv
    params = config.load_edge_params()
    snaps = load_snapshots()
    settlements = load_settlements()
    out = model_clv.model_clv_report(
        snaps, settlements, min_samples=args.min_samples,
        signal_version=args.signal_version, **params)
    print(json.dumps(out, indent=2))
```

Register the subcommand in `main()` (beside the `clv-report` parser):

```python
    pmc = sub.add_parser("model-clv-report")
    pmc.add_argument("--min-samples", type=int, default=30)
    pmc.add_argument("--signal-version", default="unknown",
                     choices=["raw", "recalibrated", "unknown"])
    pmc.set_defaults(func=_cmd_model_clv_report)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend_ml/signal_research/tests/test_report_model_clv.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full signal suite (no regressions)**

Run: `python -m pytest backend_ml/signal_research/ -q`
Expected: PASS (all tasks' tests green)

- [ ] **Step 6: Add the deferred live-settlement item to the checklist**

Append to `docs/superpowers/before-live-checklist.md`, section F:

```markdown
- [ ] **Live Kalshi settlement for model-CLV.** `signal_research/report.py:load_settlements`
  reads a `{ticker: 0|1}` file that a deferred live step must populate: after a
  slate settles, query each ticker's Kalshi resolution (YES=home-win -> 1, else 0)
  and write `signal_research/artifacts/settlements.json`. Until then,
  `model-clv-report` runs with an empty settlement map: model CLV and entry edge
  accrue, but the `beats_close` block stays `null`. Pass `--signal-version
  raw|recalibrated` matching how `fair_values.json` was published at capture time.
```

- [ ] **Step 7: Commit**

```bash
git add backend_ml/signal_research/report.py backend_ml/signal_research/tests/test_report_model_clv.py docs/superpowers/before-live-checklist.md
git commit -m "feat(signal): model-clv-report CLI + settlement loader; checklist for live settlement"
```

---

## Definition of Done

- All 4 tasks' tests pass: `python -m pytest backend_ml/signal_research/ -v` (green, including the existing suite).
- Edge-threshold parity test locks the Python formula to the C++ engine (`conf 0.95 → 3`, `conf 0.55 → 6`).
- Snapshot rows carry `p_model` (published fair value); `None` when the ticker is absent — existing rows/tests unaffected.
- `model-clv-report` produces raw-sign + would-trade CLV buckets and a beats-close block (`null` until settlements exist), gated by `min_samples`, tagged with `signal_version`.
- Constants single-sourced from `engine.json`; no second copy.
- Live Kalshi settlement fetch recorded as a deferred before-live item.
