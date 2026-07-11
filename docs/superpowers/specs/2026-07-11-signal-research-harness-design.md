# Signal-Research Harness — Design Spec

**Date:** 2026-07-11
**Status:** Approved design, ready for implementation plan
**Repo:** `/Users/vaibhav.wudaru/benchwarmer-nba`
**Predecessors:** [`2026-07-10-cpp-kalshi-arb-engine-design.md`](2026-07-10-cpp-kalshi-arb-engine-design.md), [`../plans/2026-07-10-cpp-kalshi-trading-engine.md`](../plans/2026-07-10-cpp-kalshi-trading-engine.md)

---

## 1. Purpose

The C++ Kalshi trading engine (v1, paper-only, merged to `main`) trades on fair
values published by the existing `backend_ml` ensemble. Its edge is **not**
latency — Kalshi is reached over the public internet with rate limits — it is
**signal quality vs. the market's implied price.** This harness measures and
improves that signal.

It answers three questions the current tooling does not:

1. **Is the model calibrated?** `backtest.py` measures only accuracy (hit-rate).
   A trading signal needs *calibrated probabilities* — when the model says 60%,
   it must win ~60%. Accuracy and calibration are different properties.
2. **Can we make it better-calibrated cheaply?** Fit a recalibration map
   (isotonic/Platt) and feed the corrected probabilities back into the fair
   values the engine trades on.
3. **Does the signal beat the market after fees?** Track Closing Line Value
   (CLV) and edge-vs-close — the single best predictor of long-run profitability
   in prediction markets — by capturing market prices going forward.

**Non-goals (explicitly out of scope for this spec):** an in-play/live model
(the current model is pre-game only); retroactive CLV from paid historical
market APIs (we chose forward-capture only); any change to the C++ engine; any
real-money order routing. These are separate future specs.

---

## 2. Scope decisions (locked during brainstorming)

| Decision | Choice | Consequence |
|---|---|---|
| Harness goal | **Both** calibration + CLV in one harness | Unified around one `(prediction, market_price, outcome)` dataset schema |
| Market data source | **Forward-capture only** | CLV is **zero on day one**; it is infrastructure that accrues signal over coming slates. No paid API tier, no retroactive numbers |
| Recalibration wiring | **Wire into fair values** | `publish_fair_values.py` applies the fitted map behind a `RECALIBRATE=1` flag, so raw vs. recalibrated is A/B-comparable |
| Calibration dataset | **Recompute-from-cache primary + Supabase served-prediction cross-check** | Volume for reliability curves, plus a leakage-free sanity check |

---

## 3. Architecture

Offline, Python-only. New package `backend_ml/signal_research/`, kept separate from the
model files (so we neither bloat `predict.py` nor touch the C++ hot path).
Everything unifies around one tidy dataset schema:

```
(game_id, date, home_team_id, away_team_id, p_model, [market_price], outcome)
```

Calibration uses the `p_model` + `outcome` columns. CLV adds the `market_price`
column, sourced by forward capture. The two halves run independently — neither
blocks the other.

### 3.1 Module map

Seven deep modules, each with one clear responsibility, each testable in
isolation without real models or API keys.

| Module | Responsibility | Depends on | Tested via |
|---|---|---|---|
| `signal/dataset.py` | Assemble the labeled eval set. `build_recompute_dataset()` (as-of features + models, primary) and `build_served_dataset()` (Supabase `game_predictions` ⋈ `games`, cross-check) | `data_engine`, joblib models / Supabase | synthetic frames + mocked models |
| `signal/calibration.py` | Pure metrics on `(p, y)`: Brier, log-loss, reliability table, ECE, `calibration_report()` | none (pure) | synthetic — the TDD core |
| `signal/recalibration.py` | Fit isotonic (primary) / Platt map, **out-of-sample** eval, save/load JSON artifact | scikit-learn | synthetic miscalibrated set |
| `signal/market_capture.py` | Forward snapshotter: Kalshi mid/close + de-vigged book consensus → `market_snapshots` store. Owns a pure `devig()` | new `fetch_two_way_odds()`, Kalshi REST GET | pure de-vig unit tests; API mocked |
| `signal/clv.py` | Pure: from snapshots + settlement → CLV and edge-vs-close **after fees** | none (pure) | synthetic snapshots |
| `signal/report.py` (CLI) | `evaluate` / `capture` / `clv_report` entrypoints; human-readable + JSON output | above | smoke tests |
| `publish_fair_values.py` (edit) | Optional recalibration step behind `RECALIBRATE=1`; recompute `confidence` from recalibrated `p` | `recalibration.py` | flag-off passthrough == current behavior |

**Deliberately NOT reused:** `predict_games()` — it builds features from
`get_latest_team_stats()` (today's stats), which leaks future information into
past games. See §5.1.

**Deliberately NOT broken:** `fetch_live_odds()` keeps its current single-value
shape (`predict.py` depends on `odds_map[home_id] = price`). De-vig gets a new,
separate `fetch_two_way_odds()`.

---

## 4. Data flow

Three independent flows.

### Flow A — Calibration + recalibration (can-do-now)

```
build_training_dataset()  →  as-of features per game (groupby.shift(1), no leak)
   → load xgb/ridge/scaler .pkl → ensemble p_model → dataset[p_model, home_win]
   → calibration_report()  → {brier, log_loss, reliability_table, ece}
   → fit_recalibrator() on TRAIN split → eval on TEST split → report Δbrier, Δlog_loss
   → save signal/artifacts/recalibrator.json
   → (cross-check) build_served_dataset() → calibration_report() on served predictions
```

The ensemble math must match `backtest.py` exactly:
`p_model = xgb_weight · p_xgb + ridge_weight · p_ridge`, weights from
`ensemble_weights.json` (fallback 0.5/0.5). Ridge probability is
`sigmoid(decision_function)`.

### Flow B — Market capture (forward infra; you trigger it on a schedule)

```
watchlist (ticker ↔ game)  →  at defined moments (T-60min, tip-off):
   Kalshi REST GET mid/close   +   fetch_two_way_odds() → devig() → book consensus
   → append row to market_snapshots store
        (ticker, game_id, moment, kalshi_p, book_p, asof)
```

### Flow C — CLV report (forward; once games settle)

```
market_snapshots  +  settlement outcomes  →  clv.py
   → CLV     = f(entry_price, closing_price)
   → edge-vs-close = |p_model − market_p| − fee_cents_per_contract/100   (after fee)
   → report accrued over captured slates; refuse to summarize below min sample
```

---

## 5. Correctness landmines (must be enforced, not just noted)

### 5.1 Leakage (Flow A) — the headline calibration risk

- The recompute path uses **only** `build_training_dataset()`'s as-of features
  (built via `groupby.shift(1)` and rolling windows — strictly pre-game) and the
  same ensemble weighting as `backtest.py`.
- `dataset.py` must **never** import or call `predict_games()`. A guard test
  asserts `predict_games` is absent from `dataset.py`'s import graph.
- Recalibration improvement is reported **out-of-sample only** (train/test split
  or time-series CV). An in-sample Brier drop is meaningless and must not be
  reported as the headline number.

### 5.2 Forward-capture-only CLV — set expectations in the output itself

- CLV is **zero on day one by construction.** `clv_report` prints the captured
  slate/snapshot count and **refuses to summarize below a stated minimum sample**
  (config constant), so an empty report reads as "not enough captured slates
  yet," never "broken."

### 5.3 Fees are load-bearing

- Edge-vs-close subtracts `fee_cents_per_contract` — the *same* value the C++
  engine uses in `engine.json` (single source of truth; the harness reads it,
  does not redefine it). An unfee'd "edge" is not real.
- De-vig is mandatory before any book comparison. Unit invariant:
  `devig(-110, -110) == (0.5, 0.5)`; the two de-vigged probabilities sum to 1.

---

## 6. Testing strategy (TDD, synthetic-first)

No real models or API keys in the test suite — `.pkl` models and `.csv` caches
are gitignored, and API calls need network/keys.

- **`calibration.py`** — the TDD core. Known inputs → known outputs: all `p=0.5`
  with half winning → Brier = 0.25; a perfectly-calibrated synthetic set →
  ECE ≈ 0; a systematically-overconfident set → ECE above threshold. Reliability
  bins verified by construction.
- **`recalibration.py`** — deliberately miscalibrated synthetic set (e.g.
  `p_observed = p_raw²`); assert isotonic **reduces out-of-sample Brier**; assert
  save → load round-trips; assert `transform` is monotonic.
- **`market_capture.devig()` / `clv.py`** — pure unit tests: `devig(-110,-110) =
  (0.5, 0.5)`; favorite/dog asymmetry sums to 1; CLV sign correct on a
  hand-worked entry-vs-close example; edge flips sign when the fee exceeds raw
  edge.
- **`dataset.py`** — schema + join logic on synthetic frames with **mocked**
  models (inject a fake `predict_proba`); guard test asserts `predict_games` is
  not in the import graph.
- **`publish_fair_values.py`** — flag **off** → output byte-identical to current
  behavior (protects the working engine); flag **on** with a known recalibrator
  artifact → `p_yes` transformed and `confidence` recomputed as `max(p, 1−p)`.
- **Deferred to live verification (like the C++ gateway):** the real Kalshi REST
  GET and live `fetch_two_way_odds` need network/keys — scoped compile-and-mock
  here, live-verified by the user, captured in the before-live checklist.

---

## 7. Security & secrets

- `ODDS_API_KEY` and any Kalshi read key stay in environment variables — never
  committed.
- The `market_snapshots` output path and `signal/artifacts/` are gitignored,
  like `fair_values.json`.
- This harness is analysis + data capture only. It places **no orders** and does
  not touch the live venue. The recalibration wiring changes only the *fair
  values* the paper engine already consumes.

---

## 8. Deliverables

1. `backend_ml/signal_research/` package (7 modules above) with unit tests.
2. `signal/artifacts/recalibrator.json` produced by the `evaluate` CLI.
3. `market_snapshots` store populated by the `capture` CLI (forward).
4. `publish_fair_values.py` recalibration flag (`RECALIBRATE=1`), off by default.
5. A short calibration + recalibration report (Brier/log-loss/ECE, out-of-sample
   Δ) — the evidence that tells the user whether to trust the recalibrated
   signal before relying on it live.
6. Before-live-checklist additions for the deferred live-capture items.

---

## 9. Open follow-on specs (not this one)

- In-play / live win-probability model into the `FairValueProvider` seam the C++
  spec left open.
- Retroactive CLV via paid historical market APIs, if a data budget appears.
- Feature/model upgrades (recalibrated confidence as a real uncertainty estimate,
  lineup-latency edge) once calibration evidence shows where the model is weak.
