# Before-Live Checklist — C++ Kalshi Trading Engine

**Status:** v1 is **paper-only** and merge-ready. This checklist gates the path to trading real money (the plan's M5). Every item here is fine for paper and a **blocker for live**. Do not enable real order routing until all are addressed.

Branch: `feat/cpp-kalshi-trading-engine`. Engine: `trading_engine/`. 33/33 C++ tests, 3/3 Python tests pass at merge.

---

## A. Live hardware-in-the-loop verification (needs your Kalshi credentials — could not be run in the build)

These were built to compile+link but never run against the real API here:

- [ ] **M1 — live market data.** Run `te_engine` with `KALSHI_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH` and a populated `config/watchlist.json` (tonight's NBA tickers). Confirm the maintained order book for a ticker matches Kalshi's website. (`gateway_run.cpp`)
- [ ] **M2 — fair values.** Run `python -m backend_ml.publish_fair_values` to emit `trading_engine/fair_values.json` from the real `predict_games()`; confirm tickers resolve and the engine loads them (no "skip: has_fv=false").
- [ ] **M3 — paper session.** Run a full pre-game slate on `PaperVenue`; confirm via the Telemetry JSONL that books update, `skip`/`take`/`quote` fire sanely, and paper positions/P&L evolve reasonably. **Zero real orders** are placed in this mode.
- [ ] **Reconnect drill.** Kill the network mid-session; confirm the reconnect-with-backoff loop re-subscribes and rebuilds books from the fresh snapshot. (`gateway_run.cpp`)
- [ ] **Kill-switch drill.** `touch KILL` (from the engine's working dir) mid-session; confirm quoting halts within one update cycle and a `"killed"` event is logged. Note: `KILL` is a **CWD-relative** path today.

## B. Risk-manager gaps (config fields exist but are NOT enforced in v1)

- [ ] **Aggregate exposure cap.** `Config::max_aggregate_exposure_cents` is loaded but unused. Enforce a total `Σ |position| × price` cap in `RiskManager::check`.
- [ ] **Rate-limit budget.** `Config::orders_per_sec_budget` is loaded but unused. Enforce an orders/sec budget before hitting Kalshi's real rate limits.
- [ ] **Daily-loss kill vs settlement.** The daily-loss kill is now wired to `PaperVenue::realized_pnl_cents()` and trips correctly on realized round-trips (test: `DailyLossKillSwitchTripsFromRealizedPnl`). But pre-game positions realize P&L only on **settlement**, which the paper venue does not model — so for a buy-and-hold-to-settlement flow the kill won't see the loss until settlement. Add mark-to-market or settlement accounting before relying on it live.
- [ ] **Position-flip cap.** `RiskManager::check` skips the per-market cap when an order flips the position to the opposite side (`cur` opposite to order direction). Unreachable under v1 config (`max_order_size` ≤ `max_contracts_per_market`) but must be closed if that invariant ever changes.

## C. Arbitrage semantics (currently a directional bet, not a lock)

- [ ] **Two-legged arb execution.** On an arb signal, `StrategyEngine` places only the **YES leg** (disclosed via a code comment and `"leg":"yes_only"` telemetry). `PaperVenue` models signed-YES only, so the NO leg of the lock is not executed — this is a **naked directional fill, not riskless arbitrage**. Implement a two-legged venue (YES + NO contracts) before treating arb as a lock.
- [ ] **Dead `Sell` branch.** Under the OrderBook identity `yes_ask = 100 − no_bid`, `detect_arb`'s buy-both and sell-both conditions are algebraically identical and buy is checked first, so `Action::Sell` is unreachable. Remove it or rework the arb model when the two-legged venue lands.

## D. Robustness / operability

- [ ] **SIGINT handler.** No signal handler is wired; `gw.stop()` is never called, so Ctrl-C hard-kills the process rather than shutting down cleanly. Add a handler that calls `stop()`.
- [ ] **Interruptible backoff.** `stop()` during the reconnect backoff sleep isn't observed for up to ~30s. Use an interruptible (condition-variable) sleep.
- [ ] **Malformed-frame handling.** A single malformed WS frame throws out of `parse_ws_message` → unwinds to the gateway catch → full reconnect. Catch parse errors inside `handle_raw` and drop the frame, distinct from connection errors, to avoid a disconnect storm.
- [ ] **Denied-order telemetry.** Risk-denied orders currently emit no event. Add a `"denied"` event for full decision-audit coverage.
- [ ] **Config/watchlist paths.** `set_kill_file("KILL")`, `fair_values.json`, and `config/*` are resolved relative to CWD. Move to absolute/configured paths for a deployed run.

## E. The live venue itself (M5 — explicitly out of v1 scope)

- [ ] **`LiveKalshiVenue`.** Not built. Implement signed `POST /trade-api/v2/portfolio/orders` behind the existing `OrderVenue` interface, gated off by default.
- [ ] **Go-live gate.** A separate spec/plan + explicit sign-off. Start with minimal size. Verify fee schedule is encoded correctly (`engine.json: fee_cents_per_contract`) against Kalshi's published fees — an unfee'd "edge" is not real.

## F. Signal-research harness (forward market capture — needs live creds)

These were built to run on mocked fetchers; the live paths need network + keys:

- [ ] **Live market capture.** Wire `fetch_kalshi_price(ticker)` (Kalshi REST GET
  mid/close, cents) and `fetch_two_way_odds(watchlist_row)` (two-way American
  odds for de-vig) into `signal/report.py:_cmd_capture`. Run `capture` at T-60min
  and tip-off across a real slate; confirm rows land in the snapshots JSONL.
- [ ] **CLV accrual.** CLV is zero until enough slates are captured. After N
  captured slates, `signal clv-report` should stop reporting `insufficient` and
  produce mean CLV / edge-vs-close. Confirm the fee (`engine.json`) is applied.
  Note: the implemented edge metric is book-consensus-vs-Kalshi AT ENTRY (t-60),
  not model-vs-closing — the snapshot store carries no model p by design. Treat
  "edge-vs-close" in the report output accordingly, or extend the snapshot
  schema to carry p_model before relying on it.
- [ ] **Recalibration go/no-go.** Run `signal evaluate`; only enable
  `RECALIBRATE=1` in the publisher once the out-of-sample Brier delta is a
  genuine improvement AND the served-prediction cross-check agrees. An in-sample
  gain is not sufficient.
- [ ] **Served cross-check volume.** `build_served_dataset` needs enough Supabase
  `game_predictions` with settled results to be meaningful; until then the
  recompute path stands alone.
- [ ] **`signal` package shadows stdlib `signal` in the publisher.** `publish_fair_values.py:main()` does `sys.path.insert(0, <backend_ml dir>)` so `predict.py`'s script-style imports resolve. Now that `backend_ml/signal/` exists, any bare `import signal` by a downstream dependency (joblib, sklearn, numpy) after that insert would import THIS package instead of the stdlib module. Consequence: the `RECALIBRATE=1` flag-on path has never been exercised end-to-end. Before enabling `RECALIBRATE=1` live: run `python -m backend_ml.publish_fair_values` with `RECALIBRATE=1` and a real `recalibrator.json` on a working joblib environment, confirm no stdlib-`signal` shadow crash, or harden `main()`'s sys.path handling (append instead of insert-at-0, or scope the import). The same shadow hazard exists in `signal/report.py:_cmd_evaluate` (it also does `sys.path.insert(0, <backend_ml dir>)` then imports `data_engine`/`joblib`); since `signal evaluate` is what generates `recalibrator.json`, verify/harden that site too (append to sys.path instead of insert-at-0, or scope the import).
- [ ] **Capture-moment literals are load-bearing.** The live capture caller must pass exactly `market_capture.ENTRY_MOMENT` (`"t-60"`) and `market_capture.CLOSING_MOMENT` (`"tipoff"`) as the `moment` — `clv.clv_report` pairs legs by these. A mismatched string pairs zero legs and makes `clv-report` read `insufficient` forever with no error. Import the constants; never hand-type the strings.

---

**Bottom line:** v1 proves the pipeline end-to-end on paper with a real, tested fail-closed risk gate and kill switch. Section E + the money-relevant items in B/C are hard blockers before a single real cent.
