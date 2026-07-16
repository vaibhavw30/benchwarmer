# Project Overview — NBA Prediction → Kalshi Trading

**What this repo is:** an NBA game-outcome prediction model, a low-latency C++
engine that trades those predictions as fair values on Kalshi's game-winner
markets, and a research harness that measures and sharpens the model's signal
before it reaches the trader. **Everything is paper-only today** — no real money
is placed until the explicit before-live gates are cleared.

This is the map. Each subsystem has its own authoritative spec/plan (linked in
§7); this doc explains how the pieces fit and where to start.

---

## 1. The three layers

```
                          ┌───────────────────────────────────────────────┐
   nba_api  ─────────────▶│  NBA Prediction Platform   (pre-existing)      │
   Supabase               │  data_engine → train_model → predict           │
                          │  ensemble = XGBoost + Ridge + Elo + injury impact
                          └───────────────┬───────────────────────────────┘
                                          │  home_win_probability  (p_model)
                                          ▼
                          ┌───────────────────────────────────────────────┐
                          │  Signal Research   (backend_ml/signal_research)│
                          │  calibration → recalibration (isotonic/Platt)  │
                          │  RECALIBRATE=1 corrects p before it's traded   │
                          │  market capture → CLV (forward-accruing)       │
                          └───────────────┬───────────────────────────────┘
                                          │  publish_fair_values.py
                                          ▼  fair_values.json  {ticker, p_yes, confidence}
                          ┌───────────────────────────────────────────────┐
   Kalshi WS ────────────▶│  C++ Trading Engine   (trading_engine/)        │
   (orderbook_delta)      │  gateway → order book → strategy → risk → venue│
                          │  arb · edge-take · market-make   ·  PaperVenue │
                          └───────────────┬───────────────────────────────┘
                                          │  telemetry (paper fills, JSONL)
                                          ▼
                              market_capture (forward snapshots)
                              → CLV / model-vs-close   (measures the edge)
```

**The core thesis:** the edge is **not** latency — Kalshi is reached over the
public internet with rate limits — it is **signal quality vs. the market's
implied price**. So the model and its calibration matter more than the engine's
speed. The research layer exists to prove and sharpen that edge.

---

## 2. Layer 1 — NBA Prediction Platform (`backend_ml/`, `frontend_web/`)

The original product: predicts `home_win_probability` per game and serves it to a
React web app via Supabase.

- **`data_engine.py`** — pulls NBA history from `nba_api`, builds as-of features
  (Elo, EWMA four-factors, rest/fatigue, injury impact) → `nba_training_cache.csv`.
- **`train_model.py`** — trains the ensemble (`p = 0.7·XGB + 0.3·Ridge`, weights
  in `ensemble_weights.json`) → `xgboost_nba_model.pkl`, `ridge_nba_model.pkl`,
  `feature_scaler.pkl` (all gitignored).
- **`predict.py`** — `predict_games()` produces today's slate predictions.
- **`backtest.py`** — accuracy-only backtest; **`model_bias_analyzer.py`** —
  per-team bias. (The research layer adds what these lack: calibration + CLV.)

Reference: root [`README.md`](../README.md), [`backend_ml/README.md`](../backend_ml/README.md).

---

## 3. Layer 2 — Signal Research (`backend_ml/signal_research/`)

Offline Python harness. Answers: *is the model calibrated, can we sharpen it
cheaply, and does it beat the market after fees?*

- **`calibration.py`** — Brier, log-loss, reliability table, ECE (pure).
- **`recalibration.py`** — fits an isotonic/Platt map, evaluated **out-of-sample**;
  saves `artifacts/recalibrator.json`.
- **`dataset.py`** — assembles `(p_model, outcome)` from as-of features
  (leakage-guarded: never calls `predict_games`).
- **`market_capture.py`** — forward snapshotter: Kalshi mid + de-vigged book
  consensus → `market_snapshots.jsonl`.
- **`clv.py`** — Closing Line Value (Kalshi price drift + book-vs-Kalshi edge).
- **`model_clv.py`** — model-vs-market suite: model CLV, model-beats-close (Brier
  vs the closing line), fee-aware entry edge, in raw-sign + would-trade buckets;
  edge threshold is bit-for-bit parity with the C++ engine.
- **`plots.py`** — reliability diagram (`evaluate --plot`).
- **`report.py`** — CLI: `evaluate`, `capture`, `clv-report`,
  `model-clv-report`, `fetch-settlements` (queries Kalshi for captured tickers'
  resolutions and populates `settlements.json`; needs creds).
- **`publish_fair_values.py`** (in `backend_ml/`) — joins predictions to the Kalshi
  watchlist and, behind **`RECALIBRATE=1`**, applies the recalibration map so the
  engine trades on corrected probabilities (byte-identical when off).

**What the real run showed** (13,182 games): Brier 0.2115, ECE 0.050 — the model
is skilled but **systematically under-confident on favorites** (says 74%, they win
84%). Recalibration improves out-of-sample Brier ~2% (isotonic and Platt agree),
correcting exactly that bias. Details: [`before-live-checklist.md`](superpowers/before-live-checklist.md) and the spec below.

References: [design spec](superpowers/specs/2026-07-11-signal-research-harness-design.md),
[plan](superpowers/plans/2026-07-11-signal-research-harness.md).

---

## 4. Layer 3 — C++ Trading Engine (`trading_engine/`)

Low-latency C++20 engine (Boost.Beast WS + TLS, simdjson, OpenSSL RSA-PSS auth).
Seven deep modules under `src/`:

| Module | Role |
|---|---|
| `market_data/` | Kalshi WS gateway, RSA auth, order book (`yes_ask = 100 − no_bid`) |
| `fair_value/` | loads `fair_values.json`, staleness-aware |
| `market_map/` | ticker ↔ game watchlist |
| `strategy/` | `arb` · `edge_taker` · `market_maker`, gated by `pricing.edge_threshold_cents` |
| `risk/` | fail-closed gates, position caps, daily-loss kill-switch, `KILL` file |
| `execution/` | `PaperVenue` (avg-cost P&L); **`LiveKalshiVenue` not built** |
| `telemetry/` | JSONL decision log |

**Offline demo (no creds, offseason-safe):** `paper_session` (built from
`tools/paper_session.cpp`) replays a recorded fixture against `PaperVenue`;
[`run_paper.sh`](../run_paper.sh) wires it to raw-vs-recalibrated fair values.

References: [design spec](superpowers/specs/2026-07-10-cpp-kalshi-arb-engine-design.md),
[plan](superpowers/plans/2026-07-10-cpp-kalshi-trading-engine.md).

---

## 5. Status

| Piece | Status |
|---|---|
| NBA prediction model + web app | ✅ Built (pre-existing) |
| C++ trading engine | ✅ Built, **paper-only**, merged · 33 C++ tests |
| Signal research harness (calibration, recalibration, forward CLV) | ✅ Built, merged · 42 Python tests |
| `RECALIBRATE=1` wiring + reliability plot | ✅ Built · verified on real models |
| Offline paper demo (`paper_session`, `run_paper.sh`) | ✅ Built |
| Model-vs-close CLV (model beats the closing line) | ✅ Built, merged · edge-formula parity with the C++ engine. Accrues signal once live snapshots carry `p_model` + `settlements.json` is populated (creds/in-season). |
| Live Kalshi settlement fetch (`fetch-settlements`) | ✅ Built, merged · 75 Python tests · RSA-PSS signer ported bit-for-bit from the C++ engine. Populates `settlements.json` (`result` yes→1/no→0) so `model-clv-report` fills `beats_close`; the real REST call is deferred-verify (creds/in-season). |
| Live market data / live paper session (M1/M3) | ⏳ Deferred — needs Kalshi creds + in-season slate |
| Live order routing (`LiveKalshiVenue`, M5) | ⛔ Not built — hard-gated before real money |

---

## 6. Key commands

```bash
# --- setup (Homebrew python is externally-managed; venv required) ---
python3 -m venv .venv && .venv/bin/pip install -r backend_ml/requirements.txt

# --- Layer 1: train the model (nba_api, no creds needed) ---
cd backend_ml && python train_model.py         # -> *.pkl + nba_training_cache.csv

# --- Layer 2: measure calibration + fit recalibration ---
python -m backend_ml.signal_research.report evaluate --plot artifacts/reliability.png

# publish fair values the engine trades on (in season)
RECALIBRATE=1 python -m backend_ml.publish_fair_values   # -> trading_engine/fair_values.json

# --- Layer 3: build + test the engine, run the offline paper demo ---
cmake -S trading_engine -B trading_engine/build && cmake --build trading_engine/build
ctest --test-dir trading_engine/build            # 33 tests
./run_paper.sh 0.75                              # offline raw-vs-recalibrated paper session

# --- all Python tests ---
python -m pytest backend_ml/signal_research/ -q  # 42 tests
```

Live trading (`trading_engine/build/te_engine`) additionally needs
`KALSHI_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH` and a populated
`trading_engine/config/watchlist.json` — see the before-live checklist.

---

## 7. Document index

| Doc | What it covers |
|---|---|
| [`README.md`](../README.md) | NBA prediction platform (Layer 1) setup & structure |
| [`backend_ml/README.md`](../backend_ml/README.md) | ML backend details |
| [specs/2026-07-10 cpp-kalshi-arb-engine](superpowers/specs/2026-07-10-cpp-kalshi-arb-engine-design.md) | Trading-engine design |
| [plans/2026-07-10 cpp-kalshi-trading-engine](superpowers/plans/2026-07-10-cpp-kalshi-trading-engine.md) | Engine 20-task build plan |
| [specs/2026-07-11 signal-research-harness](superpowers/specs/2026-07-11-signal-research-harness-design.md) | Calibration/recalibration/CLV design |
| [plans/2026-07-11 signal-research-harness](superpowers/plans/2026-07-11-signal-research-harness.md) | Harness 8-task build plan |
| [specs/2026-07-12 model-vs-close-clv](superpowers/specs/2026-07-12-model-vs-close-clv-design.md) | Model-vs-market CLV design (next) |
| [plans/2026-07-12 model-vs-close-clv](superpowers/plans/2026-07-12-model-vs-close-clv.md) | Model-vs-close 4-task plan (next) |
| [before-live-checklist.md](superpowers/before-live-checklist.md) | **Every gate between paper and real money** |

---

## 8. Safety posture

- **Paper-only.** `PaperVenue` only; `LiveKalshiVenue` is not built. No real order
  is placed until the before-live checklist (sections A–F) is cleared.
- **Secrets in env, never committed.** Kalshi RSA key + key id, `ODDS_API_KEY`,
  Supabase keys come from environment. `.pkl` models, `.csv` caches,
  `signal_research/artifacts/`, and the market-snapshots file are gitignored.
- **Fees are load-bearing.** Every "edge" is computed net of
  `fee_cents_per_contract` (single-sourced from `engine.json`) — an unfee'd edge
  is not real.
- **Model is read-only to the trader.** The research layer only *reads* the
  published `fair_values.json`; it never changes the model except via the
  explicit, flag-gated recalibration map.
