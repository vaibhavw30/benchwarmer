# Design: C++ Low-Latency Kalshi Trading Engine for NBA Game Markets

**Date:** 2026-07-10
**Status:** Draft for review
**Repo:** `vaibhavw30/benchwarmer` (this repo — the NBA Holistic Prediction Platform)

---

## 1. Goal

Build a new, self-contained **C++ trading engine** that market-makes and hunts edge on
**Kalshi NBA game-winner markets**, using this repo's existing NBA win-probability
ensemble as its fair-value signal. Version 1 runs against **live Kalshi market data**
but routes all orders to an **internal paper simulator** — no real money until a later,
explicitly gated milestone.

The engine lives beside the existing Python model as a new top-level module
(`trading_engine/`). The Python model is **not rewritten**; it gains one thin
"publish fair values" step (Section 6).

---

## 2. Context: what already exists in this repo

This design deliberately reuses the existing model rather than rebuilding it. Concrete
anchors the engine depends on:

| Existing asset | Location | Role in this design |
|---|---|---|
| Ensemble win-prob prediction | `backend_ml/predict.py` → `predict_games(day_offset=0)` | Produces `home_win_probability` / `away_win_probability` per game. **The fair-value source.** |
| XGBoost + Ridge ensemble | `backend_ml/predict.py:421-478`, weighted by `ensemble_weights.json` via `backend_ml/ensemble_config.py` | Produces `ensemble_prob_home` — the number we quote around. |
| Elo engine | `backend_ml/elo_engine.py` → `get_win_prob(elo_a, elo_b)`, `HOME_ADVANTAGE=100`, `BASE_ELO=1500` | Feeds the ensemble; also a standalone sanity check on fair value. |
| Injury adjustment | `backend_ml/player_impact_engine.py` → `calculate_injury_impact(hid, aid)` | Already folded into the published probability; no engine work needed. |
| Feature engineering | `backend_ml/data_engine.py` (Four-Factor EWMA, `TEAM_ALTITUDES`, `initialize_supabase`) | Runs upstream; engine never touches raw features. |
| Model feature reference | `MODEL_PARAMETERS.md` | Documents the 13 features; informs the confidence/haircut logic in Section 8. |
| Sportsbook odds fetch | `backend_ml/predict.py` → `fetch_live_odds()` + `ODDS_TEAM_MAPPING` (team-name → NBA team id) | Reused two ways: (a) a *reference* fair value cross-check, (b) the team-name mapping seeds the Kalshi `MarketMap` (Section 7). |
| Prediction storage | `SUPABASE_SCHEMA.sql` → `game_predictions` table (`home_win_probability DECIMAL(5,3)`, `confidence_score`, `model_version`, `UNIQUE(game_id, model_version)`); upserted at `backend_ml/predict.py:516` with `on_conflict='game_id'` | The transport for fair values into the engine (Section 6, Approach A). |

**Key property of the existing model:** it is a **pre-game** model. Its features
(Four-Factor EWMA, Elo, fatigue, injuries) update on the order of *once per day / on
lineup news*, not second-by-second. This is why fair value is treated as quasi-static
and the model stays out of the C++ hot path.

---

## 3. Scope

**In scope (v1):**
- Pre-game NBA game-winner (YES/NO) markets on Kalshi for the current slate.
- Live Kalshi market-data ingestion (WebSocket `orderbook_delta`).
- Three strategy behaviors: intra-market arbitrage, market-making around fair value,
  and edge-taking when the book crosses fair value.
- Risk manager + hard kill switch.
- Paper execution venue (simulated fills against the live book) with P&L / position
  tracking.
- Deterministic record/replay test harness.

**Out of scope (v1), explicitly deferred:**
- In-play / live-score trading (needs a new in-play model — the current one is pre-game).
- Live money order routing (`LiveKalshiVenue` is designed but gated off).
- Cross-market / correlated-market arbitrage (e.g., game vs series markets).
- Any change to the ML model's math, features, or training.
- Player-prop, futures, or non-basketball markets.

**Non-goals / honest constraints:**
- **Not HFT.** Kalshi is reached over the public internet with API rate limits. C++
  buys a fast, deterministic, GC-free decision loop — not colocated speed. The edge is
  correctness and model quality, not winning a latency race.
- **Riskless arb is thin.** True YES+NO<100 locks are rare and fleeting; the durable
  edge is model-vs-market. The arb detector is a bonus, not the thesis.

---

## 4. Architecture overview

```
                    Supabase game_predictions  (or fair_values.json)
                              │  ticker -> P_yes(home), confidence, ts
   Kalshi WSS ──► [1] MarketDataGateway ──► local OrderBook(s)
                              │                    │
                    [3] MarketMap                  ▼
              (NBA game/team ⇄ Kalshi ticker) [4] StrategyEngine ◄── [2] FairValueProvider
                                                   │   • arb: yes_ask+no_ask<100
                                                   │   • market-make around fair ± edge
                                                   │   • take when book crosses fair
                                                   ▼
                                            [5] RiskManager / KillSwitch
                                             (pos limits, max loss, rate budget)
                                                   ▼
                                            [6] OrderVenue (interface)
                                             ├─ PaperVenue      (v1)
                                             └─ LiveKalshiVenue  (deferred, gated)
                                                   ▼
                                            [7] Telemetry (P&L, positions, event log)
```

Each numbered component is a **deep module**: one responsibility, a narrow interface,
internals swappable without touching consumers. The two seams that must stay clean:
`FairValueProvider` (so an in-play model drops in later) and `OrderVenue` (so paper→live
is a config flip, not a rewrite).

---

## 5. Components

### 5.1 MarketDataGateway
- **Does:** Connect to `wss://api.elections.kalshi.com/trade-api/ws/v2`, authenticate
  (RSA-PSS over `{timestamp}GET/trade-api/ws/v2`), subscribe to `orderbook_delta` for
  the watchlist tickers, apply snapshot + incremental deltas into per-ticker
  `OrderBook` objects, and emit an update event on each change. Handles reconnect +
  resnapshot.
- **Interface:** `subscribe(ticker)`, `on_book_update(callback)`, `book(ticker) -> const OrderBook&`.
- **Hides:** auth signing, sequence-gap detection, reconnection, JSON decode.

### 5.2 FairValueProvider
- **Does:** Load the model's published fair values (Section 6), refresh on an interval
  (default 60s; pre-game values rarely change faster), expose the current fair value.
- **Interface:** `fair_value(ticker) -> optional<FairValue>` where
  `FairValue { double p_yes; double confidence; timestamp asof; }`.
- **Seam:** v1 reads pre-game values; an in-play implementation later satisfies the same
  interface with a live model + scores feed.

### 5.3 MarketMap
- **Does:** Resolve which Kalshi ticker corresponds to which NBA game, and hold tonight's
  watchlist. Seeds from Kalshi's markets REST endpoint (NBA series) joined to the model's
  game list by team name → NBA team id (reusing `ODDS_TEAM_MAPPING` from
  `backend_ml/predict.py`) and game date.
- **Interface:** `watchlist() -> vector<ticker>`, `ticker_for(game_id) -> optional<ticker>`,
  `game_for(ticker) -> optional<GameRef>`.
- **Note:** Kalshi `game_id` and this repo's NBA official `game_id`
  (`games.game_id VARCHAR` in `SUPABASE_SCHEMA.sql`) differ from Supabase's internal
  UUIDs; the map keys on the NBA official id + team names + date, never UUIDs.

### 5.4 StrategyEngine
Single-threaded, deterministic. On each book update for a ticker with a known fair value:
1. **Arb detector** — if `yes_ask + no_ask < 100 - fees`, buy both sides to lock; if
   `yes_bid + no_bid > 100 + fees`, sell both. Pure orderbook math, no model needed.
2. **Market-maker** — quote a two-sided market around `fair_price = round(100 * p_yes)`:
   `bid = fair_price - half_spread - inventory_skew`,
   `ask = fair_price + half_spread - inventory_skew`, where `half_spread` widens with
   `(1 - confidence)` and `inventory_skew` pushes quotes to reduce current position.
3. **Edge-taker** — if best ask `< fair_price - edge_threshold`, lift it (buy YES cheap);
   if best bid `> fair_price + edge_threshold`, hit it (sell rich). `edge_threshold`
   accounts for fees + a confidence haircut (Section 8).

All three propose orders; the RiskManager decides what actually goes out.

### 5.5 RiskManager / KillSwitch
Every order passes through. Enforces: max contracts per market, max aggregate exposure
(sum |position| × price), max order size, **max daily realized loss → hard kill**, and a
Kalshi API **rate-limit budget** (orders/sec). Exposes a manual kill switch (file flag +
signal) that cancels resting quotes and halts new orders. Fails closed: if fair value is
stale (older than N minutes) or a book is crossed/invalid, it blocks quoting for that
market.

### 5.6 OrderVenue (interface) + PaperVenue / LiveKalshiVenue
- **Interface:** `place(order) -> order_id`, `cancel(order_id)`, `on_fill(callback)`,
  `positions()`, `pnl()`.
- **PaperVenue (v1):** simulates fills against the *live* order book from
  MarketDataGateway (marketable orders fill at the touch up to available size; resting
  quotes fill when the live book trades through them). Tracks positions, realized +
  mark-to-market P&L. No network writes.
- **LiveKalshiVenue (deferred):** signs and POSTs real orders to
  `/trade-api/v2/portfolio/orders` (RSA-PSS auth). Built to the same interface but
  compiled behind a config gate and left disabled until a separate go-live review.

### 5.7 Telemetry
Structured event log (JSON lines): every book update acted on, every quote, every fill,
running P&L and positions. Enough to reconstruct a session and to later feed the existing
`frontend_web/` React dashboard if desired.

---

## 6. The fair-value bridge (Approach A — chosen)

The C++ engine must never do model inference in the hot path. Instead the **Python model
publishes fair values** and the engine **consumes** them.

**Producer (Python, small addition to this repo):** a new function/script
`backend_ml/publish_fair_values.py` that:
1. Calls the existing `predict_games()` (`backend_ml/predict.py`) to get each game's
   `home_win_probability`.
2. Resolves each game to its Kalshi ticker via `MarketMap` inputs (team names + date).
3. Writes `{ ticker, p_yes, confidence_score, model_version, asof }` rows — either into
   the existing `game_predictions` table (add a nullable `kalshi_ticker` column) or to a
   simple `trading_engine/fair_values.json`. **Default: JSON file**, because it needs no
   schema migration and the engine reads it with zero external deps; Supabase remains an
   option for a shared/remote deployment.
4. Runs on the same cadence as predictions (pre-slate, and re-run on lineup news).

**Consumer (C++):** `FairValueProvider` polls the JSON file (or Supabase row) every 60s
and hot-reloads. `p_yes` for the YES(home-wins) contract; the NO contract fair is
`100 - fair_price`.

**Why JSON default:** pre-game values change slowly, the payload is tiny (~a dozen games),
and it keeps the C++ side dependency-free for v1. Supabase is a drop-in alternative behind
the same `FairValueProvider` interface.

---

## 7. NBA game ⇄ Kalshi ticker mapping

- Kalshi exposes NBA markets under basketball series (confirmed live: per-game winner
  markets exist and trade with real volume). The engine calls Kalshi's markets REST
  endpoint filtered to the NBA game series, gets the active tickers for today, and joins
  them to the model's game list.
- **Join key:** (home team, away team, game date). Team-name normalization reuses the
  30-team `ODDS_TEAM_MAPPING` already in `backend_ml/predict.py`.
- Ambiguities (doubleheaders, name variants) are logged and skipped rather than guessed —
  a mismapped ticker means trading the wrong game, so the map fails closed.

---

## 8. Fair value → tradable price (the conversion that makes or breaks it)

1. **Model prob → cents:** `fair_price = round(100 * p_yes)` for the YES(home) contract.
2. **Confidence haircut:** the effective edge required to trade scales with model
   uncertainty. Define `edge_threshold = base_edge + fee_buffer + k * (1 - confidence)`.
   Low-confidence games (near 0.5, or where XGB/Ridge disagree per `models_agree` in
   `predict.py`) demand a wider divergence before the engine acts.
3. **Fees:** Kalshi charges maker/taker fees; `fee_buffer` must exceed round-trip fees or
   "arb" and "edge" are illusory. Fee schedule is a config constant, validated against
   Kalshi's published fees at implementation.
4. **Reference cross-check (optional, uses existing code):** de-vig the sportsbook h2h
   odds from `fetch_live_odds()` and, if Kalshi diverges from *both* the model and the
   book consensus, treat that as a stronger signal; if the model and book disagree
   sharply, widen the threshold (don't fight two disagreeing sources).

---

## 9. Data flow & threading

- **Thread A (I/O):** MarketDataGateway WebSocket read loop → decode delta → push
  `BookUpdate` onto a lock-free SPSC queue.
- **Thread B (decision):** single consumer drains the queue, updates the `OrderBook`,
  runs StrategyEngine → RiskManager → OrderVenue. Single-threaded by design for
  determinism and easy replay.
- **Thread C (slow):** FairValueProvider refresh timer + Telemetry flush.
- Rationale: keep the decision path single-threaded and allocation-light; all
  cross-thread handoff via bounded lock-free queues.

---

## 10. Proposed repo layout (new module, existing code untouched)

```
benchwarmer/
├── backend_ml/                     # EXISTING — Python model (unchanged except one add)
│   ├── predict.py                  #   reused: predict_games(), fetch_live_odds()
│   ├── elo_engine.py               #   reused: get_win_prob()
│   ├── ensemble_config.py, ensemble_weights.json
│   ├── player_impact_engine.py, data_engine.py
│   └── publish_fair_values.py      # NEW — the only Python addition (Section 6)
├── frontend_web/                   # EXISTING — React/Vite (optional telemetry sink later)
├── trading_engine/                 # NEW — the C++ engine
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── market_data/            # [1] gateway, order book, Kalshi WS + auth
│   │   ├── fair_value/             # [2] provider (JSON/Supabase reader)
│   │   ├── market_map/             # [3] NBA game ⇄ ticker
│   │   ├── strategy/               # [4] arb, market-maker, edge-taker
│   │   ├── risk/                   # [5] risk manager + kill switch
│   │   ├── execution/              # [6] OrderVenue, PaperVenue, LiveKalshiVenue
│   │   ├── telemetry/              # [7] structured logging, P&L
│   │   └── main.cpp
│   ├── tests/                      # unit + record/replay
│   ├── config/                     # limits, fees, thresholds, watchlist
│   └── fair_values.json            # written by publish_fair_values.py (gitignored)
└── docs/superpowers/specs/         # this document
```

---

## 11. Tech stack

- **Language/build:** C++20, CMake. Single Linux/macOS box.
- **Crypto:** OpenSSL (RSA-PSS SHA-256 signing for Kalshi auth).
- **WebSocket + HTTP:** Boost.Beast (WS + REST) — one dependency for both; libcurl as a
  REST fallback.
- **JSON:** `simdjson` for the hot decode path (orderbook deltas), `nlohmann/json` for
  config/fair-value files.
- **Concurrency:** lock-free SPSC queues (e.g., `boost::lockfree::spsc_queue`).
- **Testing:** GoogleTest.
- **Secrets:** Kalshi API key id + RSA private key via env / untracked file. **Private key
  never committed; `trading_engine/config/*secret*` and `fair_values.json` gitignored.**

---

## 12. Testing strategy

1. **Unit tests** — arb math (fee-aware), MM quote construction (spread widening,
   inventory skew), edge-threshold + confidence haircut, order-book delta application.
2. **Record/replay** — capture real `orderbook_delta` streams to disk; replay them
   deterministically through StrategyEngine + PaperVenue. Same input → same orders,
   every run. This is the primary confidence mechanism.
3. **Paper session** — run live against Kalshi WS with PaperVenue; verify fills, P&L,
   and that RiskManager/kill switch behave under real market motion.
4. **Model-fair sanity** — assert published fair values match `predict_games()` output
   for the same slate (guards the Python↔C++ seam).

---

## 13. Milestones

- **M0 — Skeleton:** CMake project, config loading, Telemetry, empty component
  interfaces, one passing unit test.
- **M1 — Market data:** MarketDataGateway connects, authenticates, maintains a correct
  live order book for a hand-picked NBA ticker (verified against Kalshi's UI).
- **M2 — Fair value + map:** `publish_fair_values.py` emits JSON from `predict_games()`;
  FairValueProvider + MarketMap resolve tonight's games to tickers.
- **M3 — Strategy + paper:** arb detector, market-maker, edge-taker → RiskManager →
  PaperVenue; full record/replay harness green; first live paper session with P&L.
- **M4 — Hardening:** risk limits, kill switch drills, reconnection, fee validation.
- **M5 (gated, separate review):** LiveKalshiVenue enabled with minimal size — **not part
  of v1**, requires an explicit go-live checklist and sign-off.

---

## 14. Risks & open questions

- **Kalshi NBA per-game markets may be thin pre-game** → little to arb, wide spreads.
  Mitigation: market-making captures spread even when arb is absent; watchlist favors
  liquid games.
- **Exact Kalshi fee schedule** must be confirmed and encoded before any "edge" is
  trusted (Section 8).
- **Ticker mapping correctness** is safety-critical (Section 7); fails closed.
- **Fair-value staleness** on lineup news: RiskManager blocks quoting on stale values;
  `publish_fair_values.py` cadence must cover late scratches.
- **Open:** JSON file vs Supabase for the fair-value transport in the deployed setup
  (v1 defaults to JSON; revisit if the engine runs remote from the model).
- **Open:** exact set of Kalshi order types to use (limit/IOC/post-only) — determined at
  M1 against the live API.

---

## 15. Explicit "do not" list

- Do not run model inference in the C++ decision path.
- Do not enable `LiveKalshiVenue` in v1; real orders require the M5 gated review.
- Do not modify the model's features, training, or ensemble math.
- Do not commit the RSA private key, API key id, or `fair_values.json`.
- Do not guess ticker mappings — skip and log ambiguous ones.
