# Model-vs-Close CLV — Design Spec

**Date:** 2026-07-12
**Status:** Approved design, ready for implementation plan
**Repo:** `/Users/vaibhav.wudaru/benchwarmer-nba`
**Predecessors:** [`2026-07-11-signal-research-harness-design.md`](2026-07-11-signal-research-harness-design.md), [`../plans/2026-07-11-signal-research-harness.md`](../plans/2026-07-11-signal-research-harness.md)

---

## 1. Purpose

The signal-research harness measures whether the NBA model is *calibrated* and
captures market prices forward. Its CLV metric today (`clv.py`) measures Kalshi
price drift and a **book-consensus-vs-Kalshi** edge at entry — it never sees the
model's own probability. So it cannot answer the question that actually matters
for the trader: **is our model a better predictor than the market, and does the
market confirm our positions?**

This spec brings the model's own probability — the `p_yes` from the published
`fair_values.json`, i.e. exactly what `te_engine` trades on (raw or recalibrated
per `RECALIBRATE`) — into the snapshot store, and adds the model-vs-market
metric suite:

1. **Model CLV** — when the model takes a side at entry (t-60), does the closing
   line move *toward* us? Needs entry + close prices + model side; **no
   settlement required** (earliest usable signal).
2. **Model beats close** — is `p_model` a *sharper* predictor of the actual
   result than the closing line? Brier of `p_model` vs. Brier of the closing
   Kalshi price. Needs settlement. The gold-standard validation.
3. **Entry-edge existence** — `|p_model − kalshi_entry| − fee`: is there
   tradeable divergence at all?

**Non-goals:** an in-play model; retroactive CLV from paid historical APIs
(still forward-capture only); any change to the C++ engine; real-money routing.

---

## 2. Scope decisions (locked during brainstorming)

| Decision | Choice | Consequence |
|---|---|---|
| Metric scope | **Full suite** — Model CLV + beats-close + entry-edge | One report answers: is there edge, does the market confirm it, is the model sharper than the close |
| Side attribution | **Both buckets** — raw-sign AND would-trade | Raw signal (every game) vs. deployed-strategy CLV (only games clearing the engine edge threshold) |
| Settlement source | **Kalshi settlement** | The YES=home-win contract resolves to 100/0 — the exact outcome, no game-id mapping; injected fetcher, live call deferred |
| `p_model` source | **Published `fair_values.json`** | Captures exactly the signal `te_engine` trades on; raw or recalibrated per `RECALIBRATE`, and the report records which |

---

## 3. Architecture

An **extension of the existing `backend_ml/signal_research/` forward-capture
pipeline** — not a new subsystem. Four focused touch points.

### 3.1 Module map

| Module | Change | Responsibility |
|---|---|---|
| `market_capture.py` (extend) | add `p_model` to snapshot rows | New pure `load_fair_values(path) -> {ticker: p_yes}`; `build_snapshot_rows` gains a `fair_values` arg and stamps `p_model` per ticker (`None` if the ticker is absent) |
| `config.py` (extend) | read `base_edge_cents`, `confidence_k` | Single-source the edge-threshold constants from `engine.json` (today it reads only `fee_cents_per_contract`) |
| `model_clv.py` (new, pure) | the metric suite | `model_side`, `edge_threshold_cents`, `would_trade`, `entry_edge_cents`, `beats_close`, and `model_clv_report` producing **two buckets** (raw-sign + would-trade) plus the beats-close block |
| `report.py` (extend CLI) | `model-clv-report` subcommand | Load snapshots, fetch Kalshi settlement (injected fetcher, live-deferred), print the report; reuses the min-sample forward-accruing guard |

**Reused, not touched:** `clv.py` (its book-vs-Kalshi drift metric stays;
`model_clv.py` is the *model-attributed* sibling, kept separate because they
answer different questions); `clv.clv_cents` (side-signed drift) is **reused**
by `model_clv`, not reimplemented; the `min_samples` gate pattern; the injected-
fetcher testing pattern.

### 3.2 Snapshot row schema change

Today: `{ticker, game_id, moment, kalshi_p, book_p, asof}`
After:  `{ticker, game_id, moment, kalshi_p, book_p, p_model, asof}`

`p_model` is the entry-time published fair value. It is captured on both legs
but the report reads it from the entry (t-60) snapshot. Existing snapshot rows
without `p_model` are treated as `p_model = None` (skipped by the model report,
still usable by the legacy `clv.py`).

---

## 4. Data flow

Three phases, forward-accruing (same shape as the existing pipeline).

### Phase 1 — Capture (per slate, at t-60 and tipoff)

```
fair_values.json → load_fair_values() → {ticker: p_yes}          # what te_engine trades on
  +  Kalshi mid   +  de-vigged book consensus
  → build_snapshot_rows(watchlist, moment, kalshi_prices, book_odds, fair_values, asof)
  → row = {ticker, game_id, moment, kalshi_p, book_p, p_model, asof}
```

### Phase 2 — Settlement (after each game)

```
per settled ticker → fetch_settlement(ticker) → outcome ∈ {1 (YES/home won), 0}
```

Kalshi resolves the YES=home-win contract to 100/0 — the exact outcome we score.
Injected fetcher; the live REST call is deferred (like the price/odds fetchers).

### Phase 3 — Report

```
snapshots_by_game (+ settlements) → model_clv_report(cfg, min_samples)
  per game (needs both legs + a non-None entry p_model, else skip):
    side_raw = "YES" if p_model > kalshi_entry else "NO"
    trade?   = |p_model − kalshi_entry|·100 ≥ edge_threshold_cents(conf, cfg)
    clv      = clv_cents(kalshi_entry·100, kalshi_close·100, side_raw)   # reuse clv.py
    edge     = |p_model − kalshi_entry|·100 − fee_cents
    if settled: brier_model=(p_model−y)², brier_close=(kalshi_close−y)²
  →
  { "signal_version": "raw" | "recalibrated" | "unknown",
    "raw_sign":    {n, mean_clv_cents, pct_positive_clv, mean_entry_edge_cents},
    "would_trade": {n, mean_clv_cents, pct_positive_clv, mean_entry_edge_cents},
    "beats_close": {n_settled, brier_model, brier_close, brier_delta, pct_model_beats_close},
    "insufficient": bool }
```

`n_clv ≥ n_settled` (CLV needs no settlement; beats-close does). Below
`min_samples`, both buckets return `insufficient=True` with `None` means; the
beats-close block is `None` when nothing has settled. The `confidence` fed to
the threshold is derived as `max(p_model, 1−p_model)` — which is **exactly** how
the publisher sets it for both raw and recalibrated values
(`publish_fair_values` uses the model's `confidence_score`, and `predict.py`
defines that as `max(p, 1−p)`; the recalibrated path recomputes the same). So
the snapshot schema does **not** need a separate `confidence` field.

---

## 5. Correctness landmines (enforced, not just noted)

### 5.1 Edge-formula duplication

The edge threshold lives in C++ (`trading_engine/src/strategy/pricing.cpp`) and
is reimplemented in Python here:

```
edge_threshold_cents(conf, cfg) = floor(base_edge_cents + fee_cents_per_contract
                                        + confidence_k · (1 − conf))
```

- Constants are single-sourced from `engine.json` (`config.py` gains
  `base_edge_cents`, `confidence_k`); the harness never hard-codes a second copy.
- A **parity test** asserts the Python formula reproduces the C++ engine's known
  values: `conf 0.95 → 3`, `conf 0.55 → 6` (base_edge 2, fee 1, k 8). If the C++
  formula ever changes, this test fails and flags the divergence.

### 5.2 Forward-accruing — zero on day one

Same guard as `clv.py`: `model_clv_report` refuses to summarize below
`min_samples` (both buckets `insufficient=True`, `None` means), and the
beats-close block stays `None` until games settle — so an empty report reads
"not enough captured slates yet," never "broken."

### 5.3 Fees are load-bearing

`entry_edge_cents` subtracts `fee_cents_per_contract` (the same `engine.json`
value). An unfee'd edge is not real.

### 5.4 `p_model` provenance

`p_model` is whatever `fair_values.json` held at capture — raw or recalibrated
per `RECALIBRATE`. The report records `signal_version` so any CLV number is
attributable to a specific signal version, never silently mixed.

---

## 6. Testing (synthetic-first, pure — no real models or keys)

- **`model_clv.py`** (TDD core):
  - `model_side` sign; `entry_edge_cents` fee subtraction (can go negative).
  - `would_trade` threshold boundary.
  - **Edge-formula parity** vs the C++ known values (`conf 0.95 → 3`,
    `conf 0.55 → 6`).
  - `beats_close` hand-worked: `p_model=0.7, close=0.6, y=1` →
    `brier_model 0.09 < brier_close 0.16` → model wins.
  - `model_clv_report`: two-bucket math on synthetic snapshots; min-sample gate
    (`None` means); beats-close `None` when unsettled; `n_clv ≥ n_settled`;
    would-trade a strict subset of raw-sign.
- **`market_capture`**: `load_fair_values` parsing; `build_snapshot_rows` stamps
  `p_model`, and sets `None` when the ticker is absent from `fair_values`.
- **`config`**: reads `base_edge_cents` / `confidence_k` with fallbacks when the
  key or file is missing.
- **`report.py`**: smoke test of `model-clv-report` on injected snapshots + a
  mocked settlement dict.
- **Deferred to live (checklist):** the real Kalshi settlement REST call, like
  the existing price/odds fetchers.

---

## 7. Security & secrets

- No new secrets. Any Kalshi read key stays in env. The snapshot store and
  `signal_research/artifacts/` remain gitignored.
- Analysis + data capture only — no orders, no live venue. Reads the already-
  published `fair_values.json`; does not re-run the model.

---

## 8. Deliverables

1. `p_model` in the snapshot schema (`market_capture.py`).
2. `config.py` reads the edge-threshold constants from `engine.json`.
3. `model_clv.py` — the pure model-vs-market metric suite, with the C++
   edge-formula parity test.
4. `model-clv-report` CLI subcommand (Kalshi settlement injected, live-deferred).
5. Before-live-checklist item for the deferred live Kalshi settlement call.

---

## 9. Open follow-on specs (not this one)

- In-play / live win-probability model.
- Retroactive CLV via paid historical market APIs.
- Position-sizing / bankroll model on top of the confirmed edge.
