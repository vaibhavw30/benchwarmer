# Engine Ingest Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the C++ engine's ingest path a correct reference implementation: validated frames, integer fixed-point units, error values instead of exceptions, and a byte-exact book-state output the future Go port must reproduce.

**Architecture:** A stateless decoder (`decode_frame`, simdjson DOM, error codes only) rejects frames that are wrong on their own; `OrderBook::apply_*` rejects frames that are wrong given current state. `MarketDataGateway::handle_raw` returns an `IngestResult`, counts rejects, never throws on frame content, and optionally streams one hand-serialized book-state line per frame through `BookStateWriter`. Strategy, risk and venue migrate to fixed-point units with behaviour pinned to a pre-migration baseline.

**Tech Stack:** C++20, CMake ≥ 3.20, GoogleTest 1.15.2, simdjson 3.10.1 (DOM API), nlohmann/json 3.11.3 (telemetry and config only), Boost.Beast (live path only).

**Spec:** `docs/superpowers/specs/2026-09-19-engine-ingest-hardening-design.md`. Read it before starting any task. Section references below (§N) point into it.

## Global Constraints

- Work on branch `engine-ingest-hardening` off `main`. Do not commit to `main` and do not push; the user pushes.
- All build and test commands run from `~/benchwarmer-nba/trading_engine`.
- `CMakeLists.txt` collects sources with `file(GLOB …)`. **After adding any new `.cpp` file, re-run `cmake -S . -B build`** before building, or the new file is silently not compiled.
- Build: `cmake --build build -j`. Tests: `./build/te_tests` (optionally `--gtest_filter=…`).
- Units (§3): `Price` = 1/10,000 dollar, `Qty` = 1/100 contract, `Micros` = 1/1,000,000 dollar, all `int64_t`. `kOneDollar = 10'000`, `kCent = 100`, `kContract = 100`, `kMicrosPerCent = 10'000`. On-book prices are `1 … 9'999`.
- No `double` or `strtod` anywhere between wire bytes and book state. The only float → price conversion is `fair_price(p_yes)` in the strategy.
- The decoder uses simdjson's **DOM** API and **error-code** interface only (`.get(x)`). No throwing conversions (`int64_t(element)`, `std::string_view(element)`).
- Book-state lines are hand-serialized (never via nlohmann or any JSON library), in the exact grammar of §6.
- Ticker rule: 1–64 bytes, each in `[A-Za-z0-9._-]`.
- Strategy arithmetic is done in whole cents and whole contracts, then scaled (§3). Integer divisions must not change.
- Do not modify `src/market_data/kalshi_auth.*`, `src/fair_value/*`, `src/market_map/*`, or the TLS/handshake half of `gateway_run.cpp`.
- Never run `te_engine` or anything that opens the live WebSocket. The live path stays unverified.
- No new dependencies. No new CMake `FetchContent` entries.
- Build must add no new compiler warnings.
- `git add` exact file paths only. Never `git add -A`, `git add .`, or a directory. Never stage `build/`, any `.csv`, `.pkl`, or `backend_ml/ensemble_weights.json`.
- Every commit message ends with a blank line and `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Test counts are reported as measured from `./build/te_tests`, never estimated.

### Spec clarifications (binding; recorded here because implementation forced a choice)

1. **`seq` must be a non-negative integer.** A present `seq` that is negative, non-integer or a string is `MalformedJson`. Reason: `-1` is the book-state format's "no seq" sentinel, so a wire `seq` of `-1` would be indistinguishable from "none".
2. **A frame applied without a `seq` leaves the book's recorded seq unchanged.** This includes snapshots. The recorded seq is always "the seq of the most recent applied frame that carried one".
3. **Structural vs value type errors.** A snapshot side that is not an array, or a level that is not a 2-element array, is `MalformedJson`. A level or delta value of the wrong JSON type (a number where the documented shape wants a string, a string or non-integer where the legacy shape wants an integer) is `BadFixedPoint`.
4. **Shape selection.** Snapshot: documented shape iff `yes_dollars_fp` or `no_dollars_fp` is present. In documented shape, a side whose `*_dollars_fp` key is absent but whose legacy key is present is `MalformedJson`; legacy keys are otherwise ignored. Delta: documented shape iff `price_dollars` or `delta_fp` is present, and then both are required (`MissingField`).
5. **Legacy integers** share the 12-digit magnitude cap: `|v| ≤ 999'999'999'999`, else `BadFixedPoint`.
6. **Decode precedence**, first match wins: (a) not well-formed / root not an object / duplicate key at top level or in `msg` (when `msg` is an object) → `MalformedJson`; (b) `type` missing or not a string → `MalformedJson`; (c) `type` not `orderbook_snapshot`/`orderbook_delta` → `Ignored`; (d) bad `seq` → `MalformedJson`; (e) `msg` missing or not an object → `MissingField`; (f) `market_ticker` missing → `MissingField`, present but not a valid ticker string → `BadTicker`; (g) shape-specific rules. The `t` field is set independently of this order (§6).
7. **`book_replay` treats every input line as a frame**, including empty lines (which are `MalformedJson`), so `f` always equals the 0-based line number.

---

## File map

| File | Responsibility | Task |
|---|---|---|
| `src/core/fixed_point.{hpp,cpp}` (new) | Integer-only fixed-point string parsing | 1 |
| `src/execution/paper_venue.{hpp,cpp}` | Integer cost basis; then units | 2, 3 |
| `src/core/types.hpp`, `src/core/config.{hpp,cpp}` | Unit aliases and constants; config conversion | 3 |
| `src/strategy/*`, `src/risk/*`, `src/execution/order_venue.hpp`, `tools/paper_session.cpp` | Unit migration, behaviour preserved | 3 |
| `src/market_data/reject_reason.hpp` (new) | `RejectReason` vocabulary + `to_string` | 4 |
| `src/market_data/kalshi_messages.{hpp,cpp}` | `decode_frame`, dual shape, validation | 4, 7 |
| `src/market_data/order_book.{hpp,cpp}` | State-dependent validation, seq, level accessors | 3, 5 |
| `src/market_data/ingest.hpp` (new) | `IngestStatus`, `IngestResult` | 6 |
| `src/market_data/book_state.{hpp,cpp}` (new) | `BookStateWriter`, line formatting | 6 |
| `src/market_data/gateway.{hpp,cpp}`, `gateway_run.cpp` | Error separation, counters, writer hook, reject log limiter | 7 |
| `tests/test_ingest_audit.cpp`, `tests/test_ingest_no_throw.cpp` (new) | Audit regression; no-throw corpus | 8 |
| `tools/book_replay.cpp` (new), fixtures, goldens | Replay tool and hand-derived goldens | 9 |
| `README.md`, `trading_engine/README.md` | Docs from measured results | 10 |

## Before you start

```bash
cd ~/benchwarmer-nba
git checkout main && git pull --ff-only
git checkout -b engine-ingest-hardening
cd trading_engine
cmake -S . -B build && cmake --build build -j && ./build/te_tests
```

Expected: `[  PASSED  ] 33 tests.` If not, stop and report; the baseline must be green.

---

### Task 1: Integer-only fixed-point parsing

**Files:**
- Create: `src/core/fixed_point.hpp`, `src/core/fixed_point.cpp`
- Test: `tests/test_fixed_point.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces: `std::optional<int64_t> te::parse_fixed(std::string_view s, int max_decimals, bool allow_negative)`; inline wrappers `parse_price_dollars(sv)` (4 decimals, unsigned), `parse_qty_fp(sv)` (2, unsigned), `parse_delta_fp(sv)` (2, signed). Result is the value × 10^max_decimals.

- [ ] **Step 1: Write the failing test**

`tests/test_fixed_point.cpp`:

```cpp
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include "core/fixed_point.hpp"
using namespace te;

// Grammar (spec §5): [ "-" if allowed ] DIGITS [ "." 1..max_decimals DIGITS ],
// DIGITS = 1..12 ASCII digits. Anything else is rejected, never truncated.
struct FixedCase {
  std::string name;
  std::string input;
  int max_decimals;
  bool allow_negative;
  std::optional<int64_t> expected;
};

class FixedPointTable : public ::testing::TestWithParam<FixedCase> {};

TEST_P(FixedPointTable, ParsesExactlyOrRejects) {
  const auto& c = GetParam();
  EXPECT_EQ(parse_fixed(c.input, c.max_decimals, c.allow_negative), c.expected);
}

INSTANTIATE_TEST_SUITE_P(Price, FixedPointTable, ::testing::Values(
  FixedCase{"DocumentedFormat", "0.0800", 4, false, 800},
  FixedCase{"FewerDecimals", "0.96", 4, false, 9600},
  FixedCase{"NoFraction", "1", 4, false, 10000},
  FixedCase{"Smallest", "0.0001", 4, false, 1},
  FixedCase{"Largest", "0.9999", 4, false, 9999},
  FixedCase{"Zero", "0", 4, false, 0},
  FixedCase{"LeadingZerosAllowed", "000.5", 4, false, 5000},
  FixedCase{"TwelveDigits", "123456789012", 4, false, 1234567890120000},
  FixedCase{"ThirteenDigits", "1234567890123", 4, false, std::nullopt},
  FixedCase{"FiveDecimals", "0.00001", 4, false, std::nullopt},
  FixedCase{"Empty", "", 4, false, std::nullopt},
  FixedCase{"BareDot", ".", 4, false, std::nullopt},
  FixedCase{"LeadingDot", ".5", 4, false, std::nullopt},
  FixedCase{"TrailingDot", "1.", 4, false, std::nullopt},
  FixedCase{"NegativeNotAllowed", "-0.5", 4, false, std::nullopt},
  FixedCase{"PlusSign", "+0.5", 4, false, std::nullopt},
  FixedCase{"LeadingSpace", " 0.5", 4, false, std::nullopt},
  FixedCase{"TrailingSpace", "0.5 ", 4, false, std::nullopt},
  FixedCase{"Exponent", "1e2", 4, false, std::nullopt},
  FixedCase{"Hex", "0x1", 4, false, std::nullopt},
  FixedCase{"TwoDots", "0.5.1", 4, false, std::nullopt}
), [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(Qty, FixedPointTable, ::testing::Values(
  FixedCase{"Documented", "300.00", 2, false, 30000},
  FixedCase{"OneDecimal", "2.5", 2, false, 250},
  FixedCase{"Whole", "13", 2, false, 1300},
  FixedCase{"ThreeDecimals", "1.234", 2, false, std::nullopt},
  FixedCase{"NegativeNotAllowed", "-1.00", 2, false, std::nullopt}
), [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(Delta, FixedPointTable, ::testing::Values(
  FixedCase{"DocumentedNegative", "-54.00", 2, true, -5400},
  FixedCase{"Positive", "7.25", 2, true, 725},
  FixedCase{"NegativeZero", "-0", 2, true, 0},
  FixedCase{"DoubleMinus", "--1", 2, true, std::nullopt},
  FixedCase{"MinusOnly", "-", 2, true, std::nullopt},
  FixedCase{"MinusDot", "-.5", 2, true, std::nullopt}
), [](const auto& info) { return info.param.name; });

TEST(FixedPoint, WrappersUseSpecScales) {
  EXPECT_EQ(parse_price_dollars("0.5400"), 5400);
  EXPECT_EQ(parse_qty_fp("20.00"), 2000);
  EXPECT_EQ(parse_delta_fp("-54.00"), -5400);
  EXPECT_EQ(parse_qty_fp("-1"), std::nullopt);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: compile error, `core/fixed_point.hpp` not found.

- [ ] **Step 3: Write the implementation**

`src/core/fixed_point.hpp`:

```cpp
#pragma once
#include <cstdint>
#include <optional>
#include <string_view>
namespace te {
// Integer-only parsing of Kalshi fixed-point strings (spec §5). Never goes
// through double: "0.0800" is parsed digit by digit, because parsing it as a
// double and scaling reintroduces exactly the float divergence the integer
// book-state format exists to avoid.
//
// Grammar: [ "-" if allow_negative ] DIGITS [ "." 1..max_decimals DIGITS ]
// where DIGITS is 1..12 ASCII digits. Returns value × 10^max_decimals, or
// nullopt for anything outside the grammar. Never truncates.
std::optional<int64_t> parse_fixed(std::string_view s, int max_decimals, bool allow_negative);

inline std::optional<int64_t> parse_price_dollars(std::string_view s) { return parse_fixed(s, 4, false); }
inline std::optional<int64_t> parse_qty_fp(std::string_view s)        { return parse_fixed(s, 2, false); }
inline std::optional<int64_t> parse_delta_fp(std::string_view s)      { return parse_fixed(s, 2, true); }
}
```

`src/core/fixed_point.cpp`:

```cpp
#include "core/fixed_point.hpp"
namespace te {
namespace {
constexpr size_t kMaxIntDigits = 12;  // keeps every scaled value far inside int64_t
bool is_digit(char c) { return c >= '0' && c <= '9'; }
}

std::optional<int64_t> parse_fixed(std::string_view s, int max_decimals, bool allow_negative) {
  size_t i = 0;
  bool negative = false;
  if (i < s.size() && s[i] == '-') {
    if (!allow_negative) return std::nullopt;
    negative = true;
    ++i;
  }

  const size_t int_start = i;
  int64_t int_part = 0;
  while (i < s.size() && is_digit(s[i])) {
    if (i - int_start == kMaxIntDigits) return std::nullopt;
    int_part = int_part * 10 + (s[i] - '0');
    ++i;
  }
  if (i == int_start) return std::nullopt;  // no integer digits: "", ".5", "-"

  int64_t frac = 0;
  int frac_digits = 0;
  if (i < s.size() && s[i] == '.') {
    ++i;
    const size_t frac_start = i;
    while (i < s.size() && is_digit(s[i])) {
      if (frac_digits == max_decimals) return std::nullopt;  // too many decimals
      frac = frac * 10 + (s[i] - '0');
      ++frac_digits;
      ++i;
    }
    if (i == frac_start) return std::nullopt;  // "1." has no fraction digits
  }
  if (i != s.size()) return std::nullopt;  // sign, space, exponent, second dot...

  int64_t scale = 1;
  for (int k = 0; k < max_decimals; ++k) scale *= 10;
  for (int k = frac_digits; k < max_decimals; ++k) frac *= 10;
  const int64_t value = int_part * scale + frac;
  return negative ? -value : value;
}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake -S . -B build && cmake --build build -j && ./build/te_tests --gtest_filter='*FixedPoint*'`
Expected: all `FixedPoint*` tests PASS. Then `./build/te_tests` — every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/core/fixed_point.hpp src/core/fixed_point.cpp tests/test_fixed_point.cpp
git commit -F - <<'EOF'
feat(engine): integer-only fixed-point parsing for Kalshi price/qty strings

Parses "0.0800"-style dollar strings and "300.00"-style counts digit by
digit into int64 fixed-point, never through double (spec §5). Rejects
anything outside the grammar rather than truncating.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 2: Paper venue integer cost basis (current units)

Replaces the floating-point average entry price with an exact integer cost basis while units are still cents and whole contracts, so the P&L change is reviewed separately from the unit migration.

**Files:**
- Modify: `src/execution/paper_venue.hpp`, `src/execution/paper_venue.cpp`
- Test: `tests/test_paper_venue.cpp`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PaperVenue` with private `std::unordered_map<Ticker, int64_t> cost_basis_` (signed: + long, − short) and `int64_t realized_`. Public API unchanged: `int realized_pnl_cents() const`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_paper_venue.cpp`:

```cpp
// Three 1-lot buys at 60, 61, 61 (average 60.67), then three 1-lot sells at
// 62. True total P&L is 3*62 - (60+61+61) = 4 cents. The old double average
// rounds each close separately, lround(1.33) = 1, giving 3. An integer cost
// basis carries the truncated remainder forward, so it is exact once flat.
TEST(Paper, CostBasisExactOnceFlat) {
  OrderBook ask60; ask60.apply_snapshot({{{50,100}},{{40,100}}});  // yes_ask = 60
  OrderBook ask61; ask61.apply_snapshot({{{50,100}},{{39,100}}});  // yes_ask = 61
  OrderBook bid62; bid62.apply_snapshot({{{62,100}},{{30,100}}});  // yes_bid = 62
  PaperVenue v;
  v.place_against(ask60, {"T", Side::Yes, Action::Buy, 60, 1});
  v.place_against(ask61, {"T", Side::Yes, Action::Buy, 61, 1});
  v.place_against(ask61, {"T", Side::Yes, Action::Buy, 61, 1});
  ASSERT_EQ(v.position("T"), 3);
  for (int i = 0; i < 3; ++i) v.place_against(bid62, {"T", Side::Yes, Action::Sell, 62, 1});
  EXPECT_EQ(v.position("T"), 0);
  EXPECT_EQ(v.realized_pnl_cents(), 4);
}

// Short side: sell 5 @ 60 opens short, buy 5 @ 55 covers -> (60-55)*5 = 25.
TEST(Paper, ShortCoverRealizesPnl) {
  OrderBook bid60; bid60.apply_snapshot({{{60,100}},{{30,100}}});  // yes_bid = 60
  OrderBook ask55; ask55.apply_snapshot({{{40,100}},{{45,100}}});  // yes_ask = 55
  PaperVenue v;
  v.place_against(bid60, {"T", Side::Yes, Action::Sell, 60, 5});
  ASSERT_EQ(v.position("T"), -5);
  v.place_against(ask55, {"T", Side::Yes, Action::Buy, 55, 5});
  EXPECT_EQ(v.position("T"), 0);
  EXPECT_EQ(v.realized_pnl_cents(), 25);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j && ./build/te_tests --gtest_filter='Paper.*'`
Expected: `Paper.CostBasisExactOnceFlat` FAILS with `realized_pnl_cents()` = 3, expected 4. `Paper.ShortCoverRealizesPnl` may pass; that is fine. It pins the short path through the rewrite.

- [ ] **Step 3: Write the implementation**

In `src/execution/paper_venue.hpp`, replace the private section and the `realized_pnl_cents` accessor:

```cpp
  int realized_pnl_cents() const override { return static_cast<int>(realized_); }

 private:
  std::unordered_map<Ticker, int> pos_;              // ticker -> signed qty
  // Signed total paid for the open position (+ long, - short). Exact integer
  // replacement for a floating-point average entry price (spec §3).
  std::unordered_map<Ticker, int64_t> cost_basis_;
  int64_t realized_ = 0;
  long fill_id_ = 0;
};
```

Add `#include <cstdint>` to the header's includes. Replace the body of `src/execution/paper_venue.cpp` with:

```cpp
#include "execution/paper_venue.hpp"
#include <algorithm>
#include <cstdlib>
namespace te {

std::string PaperVenue::place_against(const OrderBook& book, const Order& o) {
  // Marketable YES order (Kalshi convention: yes_ask = 100 - best_no_bid;
  // buying YES at p consumes NO liquidity at 100-p; selling YES at p consumes
  // YES liquidity at p).
  Cents price;
  int avail;
  if (o.action == Action::Buy) {
    auto ask = book.best_yes_ask();
    if (!ask) return "noliq";
    price = *ask;
    avail = book.qty_at(Side::No, 100 - price);
  } else {
    auto bid = book.best_yes_bid();
    if (!bid) return "noliq";
    price = *bid;
    avail = book.qty_at(Side::Yes, price);
  }

  int qty = std::min(o.qty, std::max(0, avail));
  if (qty <= 0) return "noliq";

  const int dir = (o.action == Action::Buy) ? 1 : -1;
  int& pos = pos_[o.ticker];
  int64_t& basis = cost_basis_[o.ticker];

  // Closing part of an opposing position (buy while short, sell while long).
  const bool reduces = (pos > 0 && dir < 0) || (pos < 0 && dir > 0);
  if (reduces) {
    const int closing = std::min(qty, std::abs(pos));
    const int pos_sign = (pos > 0) ? 1 : -1;
    // Proportional share of the cost basis, truncated toward zero. Whatever
    // truncation leaves behind stays in `basis` and is realized by a later
    // close, so P&L is exact once the position is flat (spec §3).
    const int64_t removed = basis * closing / std::abs(pos);
    realized_ += int64_t(pos_sign) * price * closing - removed;
    basis -= removed;
    pos += dir * closing;
    qty -= closing;
    if (pos == 0 && qty > 0) {
      // Order exceeded the closed amount: the rest opens the other side.
      pos = dir * qty;
      basis = int64_t(dir) * price * qty;
      qty = 0;
    }
  }

  if (qty > 0) {
    // Opens from flat or extends the same-side position.
    basis += int64_t(dir) * price * qty;
    pos += dir * qty;
  }

  return "fill-" + std::to_string(fill_id_++);
}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build -j && ./build/te_tests`
Expected: all tests PASS, including the pre-existing `Paper.BuyThenSellRealizesPnl` (−50), `Paper.FlipToShortAveragesCorrectly` (position −5, −50), `StrategyEngine.DailyLossKillSwitchTripsFromRealizedPnl` (−450), and `Replay.Deterministic`.

- [ ] **Step 5: Commit**

```bash
git add src/execution/paper_venue.hpp src/execution/paper_venue.cpp tests/test_paper_venue.cpp
git commit -F - <<'EOF'
fix(engine): exact integer cost basis for paper P&L

Replaces the double average entry price and per-close lround with a
signed integer cost basis. Truncated remainders stay in the basis and are
realized by later closes, so cumulative P&L is exact once flat. The old
code reports 3 cents on a 1-lot 60/61/61 -> 62x3 sequence whose true P&L
is 4 (spec §3).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 3: Fixed-point unit migration across the engine

One atomic change: the type change cannot compile halfway. Behaviour is pinned first by a characterization test on the current engine, then the same test's expected values are scaled.

**Files:**
- Modify: `src/core/types.hpp`, `src/core/config.hpp`, `src/core/config.cpp`, `src/market_data/order_book.hpp`, `src/market_data/order_book.cpp`, `src/market_data/kalshi_messages.cpp`, `src/strategy/pricing.{hpp,cpp}`, `src/strategy/arb.{hpp,cpp}`, `src/strategy/edge_taker.{hpp,cpp}`, `src/strategy/market_maker.{hpp,cpp}`, `src/strategy/strategy_engine.{hpp,cpp}`, `src/risk/risk_manager.{hpp,cpp}`, `src/execution/order_venue.hpp`, `src/execution/paper_venue.{hpp,cpp}`, `tools/paper_session.cpp`
- Test: every existing test file except `test_telemetry.cpp`, `test_fair_value.cpp`, `test_kalshi_auth.cpp`, `test_market_map.cpp`, `test_fixed_point.cpp`

**Interfaces:**
- Consumes: Task 2's cost-basis algorithm.
- Produces (later tasks rely on these exact names):
  - `te::Price`, `te::Qty`, `te::Micros` (`int64_t`); `kOneDollar`, `kCent`, `kContract`, `kMicrosPerCent`, `kSettleYes`, `kSettleNo`; `PriceLevel{Price price; Qty qty;}`. `Cents` is removed.
  - `BookSnapshot{std::vector<PriceLevel> yes, no;}`, `BookDelta{Side side; Price price; Qty delta_qty;}`
  - `OrderBook::qty_at(Side, Price) -> Qty`, `best_*() -> std::optional<Price>`
  - `Config` fields: `max_qty_per_market` (Qty), `max_aggregate_exposure` (Micros), `max_order_qty` (Qty), `max_daily_loss` (Micros), `fee_per_contract` (Price), `base_edge` (Price), plus unchanged `confidence_k`, `fair_value_refresh_secs`, `fair_value_max_age_secs`, `orders_per_sec_budget`
  - `Price fair_price(double)`, `Price edge_threshold(double, const Config&)`
  - `OrderVenue::position -> Qty`, `OrderVenue::realized_pnl_micros -> Micros`

- [ ] **Step 1: Write the characterization test (current units) and see it pass**

In `tests/test_replay.cpp`, change `run_replay` to also return the venue's final state, and add a baseline test. Replace the file's anonymous namespace and tests with:

```cpp
namespace {

constexpr long kFixedNowMs = 1783641600000L + 60000L;

struct ReplayOut { std::string log; long long position; long long realized; };

ReplayOut run_replay() {
  Config c;
  c.base_edge_cents = 2;
  c.fee_cents_per_contract = 1;
  c.confidence_k = 8.0;
  c.max_order_size = 25;
  c.max_contracts_per_market = 100;
  c.max_aggregate_exposure_cents = 500000;
  c.max_daily_loss_cents = 20000;
  c.fair_value_max_age_secs = 1800;

  FairValueProvider fv;
  {
    std::ofstream f("replay_fv.json");
    f << R"([{"ticker":"T","p_yes":0.62,"confidence":0.95,"asof":"2026-07-10T00:00:00Z"}])";
  }
  fv.load_from_file("replay_fv.json");

  RiskManager risk(c);
  PaperVenue venue;
  std::ostringstream log;
  Telemetry tel(log);
  StrategyEngine eng(c, fv, risk, venue, tel);

  MarketDataGateway gw;
  gw.on_update([&](const Ticker& t, const OrderBook& b) {
    eng.on_book_update(t, b, kFixedNowMs);
  });

  std::ifstream fixture("tests/fixtures/replay_sample.jsonl");
  if (!fixture) {
    ADD_FAILURE() << "could not open tests/fixtures/replay_sample.jsonl "
                      "(expected working directory: trading_engine/)";
    return {};
  }
  std::string line;
  while (std::getline(fixture, line)) {
    if (line.empty()) continue;
    gw.handle_raw(line);
  }
  return {log.str(), venue.position("T"), venue.realized_pnl_cents()};
}

}  // namespace

TEST(Replay, Deterministic) {
  std::string first = run_replay().log;
  std::string second = run_replay().log;
  EXPECT_EQ(first, second);
  EXPECT_NE(first.find("\"type\":\"take\""), std::string::npos);
  EXPECT_NE(first.find("\"type\":\"quote\""), std::string::npos);
  EXPECT_NE(first.find("\"type\":\"skip\""), std::string::npos);
}

// Characterization of the pre-migration engine (spec §2 baseline). After the
// unit migration the SAME decisions must appear with prices and quantities
// x100 and P&L x10,000; any other difference means a strategy decision moved.
TEST(Replay, MatchesBaseline) {
  const std::string expected =
    R"({"price":48,"qty":25,"seq":0,"ticker":"T","type":"take"})" "\n"
    R"({"ask":62,"bid":60,"fair":62,"seq":1,"size":25,"ticker":"T","type":"quote"})" "\n"
    R"({"price":48,"qty":25,"seq":2,"ticker":"T","type":"take"})" "\n"
    R"({"crossed":true,"has_fv":true,"seq":3,"stale":false,"ticker":"T","type":"skip"})" "\n"
    R"({"price":48,"qty":25,"seq":4,"ticker":"T","type":"take"})" "\n"
    R"({"ask":60,"bid":58,"fair":62,"seq":5,"size":25,"ticker":"T","type":"quote"})" "\n"
    R"({"ask":60,"bid":58,"fair":62,"seq":6,"size":25,"ticker":"T","type":"quote"})" "\n"
    R"({"price":66,"qty":25,"seq":7,"ticker":"T","type":"take"})" "\n"
    R"({"ask":61,"bid":59,"fair":62,"seq":8,"size":25,"ticker":"T","type":"quote"})" "\n";
  ReplayOut out = run_replay();
  EXPECT_EQ(out.log, expected);
  EXPECT_EQ(out.position, 50);
  EXPECT_EQ(out.realized, 450);
}
```

Run: `cmake --build build -j && ./build/te_tests --gtest_filter='Replay.*'`
Expected: both PASS on the unmigrated code. Commit this test alone:

```bash
git add tests/test_replay.cpp
git commit -F - <<'EOF'
test(engine): pin replay telemetry baseline before the unit migration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

- [ ] **Step 2: Migrate core types and config**

`src/core/types.hpp`:

```cpp
// trading_engine/src/core/types.hpp
#pragma once
#include <cstdint>
#include <string>
namespace te {
using Ticker = std::string;

// Fixed-point units (spec §3). All integer so book state serializes exactly.
using Price  = int64_t;   // 1/10,000 dollar; on-book prices are 1..9'999
using Qty    = int64_t;   // 1/100 contract
using Micros = int64_t;   // 1/1,000,000 dollar; Price x Qty needs no scaling

constexpr Price  kOneDollar     = 10'000;
constexpr Price  kCent          = 100;      // one cent in Price units
constexpr Qty    kContract      = 100;      // one whole contract in Qty units
constexpr Micros kMicrosPerCent = 10'000;
constexpr Price  kSettleYes     = kOneDollar;
constexpr Price  kSettleNo      = 0;

enum class Side   { Yes, No };
enum class Action { Buy, Sell };
struct PriceLevel { Price price; Qty qty; };
}
```

`src/core/config.hpp`:

```cpp
#pragma once
#include <string>
#include "core/types.hpp"
namespace te {
// engine.json keeps its cent / whole-contract keys; load() converts them to
// fixed-point units so existing config files stay valid (spec §3).
struct Config {
  Qty    max_qty_per_market{};      // "max_contracts_per_market" x kContract
  Micros max_aggregate_exposure{};  // "max_aggregate_exposure_cents" x kMicrosPerCent (loaded, not enforced in v1)
  Qty    max_order_qty{};           // "max_order_size" x kContract
  Micros max_daily_loss{};          // "max_daily_loss_cents" x kMicrosPerCent
  Price  fee_per_contract{};        // "fee_cents_per_contract" x kCent
  Price  base_edge{};               // "base_edge_cents" x kCent
  double confidence_k{};
  int fair_value_refresh_secs{};
  int fair_value_max_age_secs{};
  int orders_per_sec_budget{};
  static Config load(const std::string& path);
};
}
```

`src/core/config.cpp`:

```cpp
#include "core/config.hpp"
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
namespace te {
Config Config::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("config not found: " + path);
  nlohmann::json j; f >> j;
  Config c;
  c.max_qty_per_market     = j.at("max_contracts_per_market").get<int64_t>() * kContract;
  c.max_aggregate_exposure = j.at("max_aggregate_exposure_cents").get<int64_t>() * kMicrosPerCent;
  c.max_order_qty          = j.at("max_order_size").get<int64_t>() * kContract;
  c.max_daily_loss         = j.at("max_daily_loss_cents").get<int64_t>() * kMicrosPerCent;
  c.fee_per_contract       = j.at("fee_cents_per_contract").get<int64_t>() * kCent;
  c.base_edge              = j.at("base_edge_cents").get<int64_t>() * kCent;
  c.confidence_k            = j.at("confidence_k");
  c.fair_value_refresh_secs = j.at("fair_value_refresh_secs");
  c.fair_value_max_age_secs = j.at("fair_value_max_age_secs");
  c.orders_per_sec_budget   = j.at("orders_per_sec_budget");
  return c;
}
}
```

- [ ] **Step 3: Migrate the book and the legacy decoder**

`src/market_data/order_book.hpp`:

```cpp
#pragma once
#include <map>
#include <optional>
#include <vector>
#include "core/types.hpp"
namespace te {
struct BookSnapshot { std::vector<PriceLevel> yes, no; };
struct BookDelta { Side side; Price price; Qty delta_qty; };
class OrderBook {
 public:
  void apply_snapshot(const BookSnapshot& s);
  void apply_delta(const BookDelta& d);
  std::optional<Price> best_yes_bid() const;
  std::optional<Price> best_no_bid() const;
  std::optional<Price> best_yes_ask() const; // kOneDollar - best_no_bid
  std::optional<Price> best_no_ask() const;  // kOneDollar - best_yes_bid
  Qty qty_at(Side s, Price p) const;
  bool crossed() const;
 private:
  std::map<Price,Qty> yes_;  // price -> qty (YES bids)
  std::map<Price,Qty> no_;   // price -> qty (NO bids)
  static std::optional<Price> best(const std::map<Price,Qty>& m);
};
}
```

`src/market_data/order_book.cpp`:

```cpp
#include "market_data/order_book.hpp"
namespace te {
void OrderBook::apply_snapshot(const BookSnapshot& s) {
  yes_.clear(); no_.clear();
  for (auto& l : s.yes) if (l.qty > 0) yes_[l.price] = l.qty;
  for (auto& l : s.no)  if (l.qty > 0) no_[l.price]  = l.qty;
}
void OrderBook::apply_delta(const BookDelta& d) {
  auto& m = (d.side == Side::Yes) ? yes_ : no_;
  Qty q = m[d.price] + d.delta_qty;
  if (q <= 0) m.erase(d.price); else m[d.price] = q;
}
std::optional<Price> OrderBook::best(const std::map<Price,Qty>& m) {
  if (m.empty()) return std::nullopt;
  return m.rbegin()->first; // highest bid
}
std::optional<Price> OrderBook::best_yes_bid() const { return best(yes_); }
std::optional<Price> OrderBook::best_no_bid()  const { return best(no_); }
std::optional<Price> OrderBook::best_yes_ask() const {
  auto nb = best_no_bid(); if (!nb) return std::nullopt; return kOneDollar - *nb;
}
std::optional<Price> OrderBook::best_no_ask()  const {
  auto yb = best_yes_bid(); if (!yb) return std::nullopt; return kOneDollar - *yb;
}
Qty OrderBook::qty_at(Side s, Price p) const {
  auto& m = (s == Side::Yes) ? yes_ : no_;
  auto it = m.find(p); return it == m.end() ? 0 : it->second;
}
bool OrderBook::crossed() const {
  auto yb = best_yes_bid(); auto ya = best_yes_ask();
  return yb && ya && *yb >= *ya;
}
}
```

In `src/market_data/kalshi_messages.cpp`, scale the legacy integer fields (this decoder is replaced in Tasks 4 and 7; it only has to keep working meanwhile). Replace the two snapshot level loops and the delta field reads:

```cpp
    for (auto lvl : msg["yes"].get_array())
      { auto a = lvl.get_array().begin(); Price p = int64_t(*a) * kCent; ++a; Qty q = int64_t(*a) * kContract;
        out.snapshot.yes.push_back({p,q}); }
    for (auto lvl : msg["no"].get_array())
      { auto a = lvl.get_array().begin(); Price p = int64_t(*a) * kCent; ++a; Qty q = int64_t(*a) * kContract;
        out.snapshot.no.push_back({p,q}); }
```

```cpp
    out.delta.price = int64_t(msg["price"]) * kCent;
    out.delta.delta_qty = int64_t(msg["delta"]) * kContract;
```

- [ ] **Step 4: Migrate strategy**

`src/strategy/pricing.hpp`:

```cpp
#pragma once
#include "core/types.hpp"
#include "core/config.hpp"
namespace te {
Price fair_price(double p_yes);
Price edge_threshold(double confidence, const Config& c);
}
```

`src/strategy/pricing.cpp`:

```cpp
#include "strategy/pricing.hpp"
#include <algorithm>
#include <cmath>
namespace te {
// The one place a float becomes a price. Stays on the whole-cent grid the
// strategy has always quoted on (spec §3).
Price fair_price(double p_yes) {
  const long cents = std::lround(100.0 * p_yes);
  return std::clamp<long>(cents, 1, 99) * kCent;
}
// Computed in whole cents exactly as before the unit change, then scaled, so
// downstream integer divisions (the market-maker half-spread) are unchanged.
Price edge_threshold(double confidence, const Config& c) {
  const double t = double(c.base_edge / kCent) + double(c.fee_per_contract / kCent)
                 + c.confidence_k * (1.0 - confidence);
  return static_cast<Price>(std::floor(t)) * kCent;
}
}
```

`src/strategy/arb.hpp`:

```cpp
#pragma once
#include "market_data/order_book.hpp"
#include "core/config.hpp"
namespace te {
struct ArbSignal { bool present{false}; Action action{Action::Buy}; Qty qty{0}; Price yes_price{0}; Price no_price{0}; };
ArbSignal detect_arb(const OrderBook& b, const Config& c);
}
```

`src/strategy/arb.cpp`:

```cpp
#include "strategy/arb.hpp"
#include <algorithm>
namespace te {
ArbSignal detect_arb(const OrderBook& b, const Config& c) {
  ArbSignal s;
  const Price fee2 = 2 * c.fee_per_contract;
  auto ya = b.best_yes_ask(); auto na = b.best_no_ask();
  if (ya && na && (*ya + *na + fee2) < kOneDollar) {
    Qty q = std::min(b.qty_at(Side::No, kOneDollar - *ya), b.qty_at(Side::Yes, kOneDollar - *na));
    s = {true, Action::Buy, std::max(q, kContract), *ya, *na}; return s;
  }
  auto yb = b.best_yes_bid(); auto nb = b.best_no_bid();
  if (yb && nb && (*yb + *nb - fee2) > kOneDollar) {
    Qty q = std::min(b.qty_at(Side::Yes, *yb), b.qty_at(Side::No, *nb));
    s = {true, Action::Sell, std::max(q, kContract), *yb, *nb}; return s;
  }
  return s;
}
}
```

`src/strategy/edge_taker.hpp`:

```cpp
#pragma once
#include "market_data/order_book.hpp"
#include "core/config.hpp"
namespace te {
struct TakeSignal { bool present{false}; Action action{Action::Buy}; Side side{Side::Yes}; Price price{0}; Qty size{0}; };
TakeSignal detect_take(const OrderBook& b, Price fair, Price threshold, const Config& c);
}
```

`src/strategy/edge_taker.cpp`:

```cpp
#include "strategy/edge_taker.hpp"
namespace te {
TakeSignal detect_take(const OrderBook& b, Price fair, Price threshold, const Config& c) {
  TakeSignal s;
  auto ya = b.best_yes_ask();
  if (ya && *ya <= fair - threshold) { s = {true, Action::Buy, Side::Yes, *ya, c.max_order_qty}; return s; }
  auto yb = b.best_yes_bid();
  if (yb && *yb >= fair + threshold) { s = {true, Action::Sell, Side::Yes, *yb, c.max_order_qty}; return s; }
  return s;
}
}
```

`src/strategy/market_maker.hpp`:

```cpp
#pragma once
#include "core/types.hpp"
#include "core/config.hpp"
namespace te {
struct Quote { Price bid; Price ask; Qty size; };
Quote make_quote(Price fair, double confidence, Qty inventory, const Config& c);
}
```

`src/strategy/market_maker.cpp`:

```cpp
#include "strategy/market_maker.hpp"
#include "strategy/pricing.hpp"
#include <algorithm>
namespace te {
Quote make_quote(Price fair, double confidence, Qty inventory, const Config& c) {
  // Half-spread and skew are whole-cent quantities computed exactly as before
  // the unit change (spec §3): a 3-cent threshold halves to 1 cent, not 1.5.
  const Price thr_cents = edge_threshold(confidence, c) / kCent;
  const Price half = std::max<Price>(1, thr_cents / 2) * kCent;
  // 1 cent per 20 whole contracts, capped at +/-5 cents.
  const Price skew = std::clamp<Qty>(inventory / (20 * kContract), -5, 5) * kCent;
  const Price bid = std::clamp<Price>(fair - half - skew, kCent, 99 * kCent);
  const Price ask = std::clamp<Price>(fair + half - skew, kCent, 99 * kCent);
  return {bid, ask, c.max_order_qty};
}
}
```

In `src/strategy/strategy_engine.hpp`, replace the last private member and its comment with:

```cpp
  // Running total of PaperVenue's cumulative realized P&L last reported to
  // RiskManager. Feeding RiskManager the delta each fill (rather than the
  // running total) keeps its own accumulator equal to the venue's cumulative
  // realized P&L, which is what actually trips the daily-loss kill switch.
  Micros last_realized_ = 0;
```

In `src/strategy/strategy_engine.cpp`, make these replacements (every other line unchanged):

```cpp
  Price fair = fair_price(fv->p_yes);
  Price thr = edge_threshold(fv->confidence, c_);
```

and, in **both** the arb branch and the edge-take branch:

```cpp
      Micros cur_realized = venue_.realized_pnl_micros();
      risk_.record_realized_pnl(cur_realized - last_realized_);
      last_realized_ = cur_realized;
```

- [ ] **Step 5: Migrate risk and execution**

`src/execution/order_venue.hpp`:

```cpp
#pragma once
#include <string>
#include "core/types.hpp"
namespace te {
struct Order { Ticker ticker; Side side; Action action; Price price; Qty qty; };
struct Fill  { Ticker ticker; Side side; Action action; Price price; Qty qty; };

class OrderVenue {
 public:
  virtual ~OrderVenue() = default;
  virtual std::string place(const Order&) = 0;
  virtual void cancel(const std::string&) = 0;
  virtual Qty position(const Ticker&) const = 0;
  virtual Micros realized_pnl_micros() const = 0;
};
}
```

`src/execution/paper_venue.hpp` public/private sections become:

```cpp
  std::string place(const Order&) override { return "noop"; }
  void cancel(const std::string&) override {}
  Qty position(const Ticker& t) const override {
    auto it = pos_.find(t);
    return it == pos_.end() ? 0 : it->second;
  }
  Micros realized_pnl_micros() const override { return realized_; }

 private:
  std::unordered_map<Ticker, Qty> pos_;           // ticker -> signed qty
  // Signed total paid for the open position (+ long, - short). Exact integer
  // replacement for a floating-point average entry price (spec §3).
  std::unordered_map<Ticker, Micros> cost_basis_;
  Micros realized_ = 0;
  long fill_id_ = 0;
};
```

`src/execution/paper_venue.cpp`:

```cpp
#include "execution/paper_venue.hpp"
#include <algorithm>
#include <cstdlib>
namespace te {

std::string PaperVenue::place_against(const OrderBook& book, const Order& o) {
  // Marketable YES order (Kalshi convention: yes_ask = $1 - best_no_bid;
  // buying YES at p consumes NO liquidity at $1-p; selling YES at p consumes
  // YES liquidity at p).
  Price price;
  Qty avail;
  if (o.action == Action::Buy) {
    auto ask = book.best_yes_ask();
    if (!ask) return "noliq";
    price = *ask;
    avail = book.qty_at(Side::No, kOneDollar - price);
  } else {
    auto bid = book.best_yes_bid();
    if (!bid) return "noliq";
    price = *bid;
    avail = book.qty_at(Side::Yes, price);
  }

  Qty qty = std::min(o.qty, std::max<Qty>(0, avail));
  if (qty <= 0) return "noliq";

  const int dir = (o.action == Action::Buy) ? 1 : -1;
  Qty& pos = pos_[o.ticker];
  Micros& basis = cost_basis_[o.ticker];

  // Closing part of an opposing position (buy while short, sell while long).
  const bool reduces = (pos > 0 && dir < 0) || (pos < 0 && dir > 0);
  if (reduces) {
    const Qty closing = std::min(qty, std::abs(pos));
    const int pos_sign = (pos > 0) ? 1 : -1;
    // Proportional share of the cost basis, truncated toward zero. Whatever
    // truncation leaves behind stays in `basis` and is realized by a later
    // close, so P&L is exact once the position is flat (spec §3).
    const Micros removed = basis * closing / std::abs(pos);
    realized_ += pos_sign * price * closing - removed;
    basis -= removed;
    pos += dir * closing;
    qty -= closing;
    if (pos == 0 && qty > 0) {
      // Order exceeded the closed amount: the rest opens the other side.
      pos = dir * qty;
      basis = dir * price * qty;
      qty = 0;
    }
  }

  if (qty > 0) {
    // Opens from flat or extends the same-side position.
    basis += dir * price * qty;
    pos += dir * qty;
  }

  return "fill-" + std::to_string(fill_id_++);
}
}
```

`src/risk/risk_manager.hpp` — replace the `RiskDecision` struct, the scope comment, and the class body:

```cpp
struct RiskDecision { bool allow; std::string reason; Qty approved_qty; };
// NOTE(v1 scope): Config::max_aggregate_exposure and
// Config::orders_per_sec_budget are loaded but NOT enforced by this class in
// v1 -- only max_order_qty, max_qty_per_market, and max_daily_loss gate
// orders below. Don't read the "fail-closed" behavior as covering aggregate
// exposure or order-rate limits.
class RiskManager {
 public:
  explicit RiskManager(const Config& c) : c_(c) {}
  void set_position(const Ticker& t, Qty pos) { pos_[t] = pos; }
  void record_realized_pnl(Micros delta);
  void trip_kill_switch() { killed_ = true; }
  bool killed() const { return killed_; }
  // Kill-switch file flag: if a file exists (openable) at `path`, trips the
  // kill switch. Cheap to call every book-update tick; a missing path is a
  // normal no-op (not an error) so operators can drop/remove the file live.
  void poll_kill_file(const std::string& path) {
    std::ifstream f(path);
    if (f.good()) trip_kill_switch();
  }
  RiskDecision check(const Order& o, bool fair_stale, bool book_crossed);
 private:
  const Config& c_;
  std::unordered_map<Ticker,Qty> pos_;
  Micros realized_pnl_ = 0;
  bool killed_ = false;
};
```

`src/risk/risk_manager.cpp`:

```cpp
#include "risk/risk_manager.hpp"
#include <algorithm>
#include <cstdlib>
namespace te {
void RiskManager::record_realized_pnl(Micros delta) {
  realized_pnl_ += delta;
  if (realized_pnl_ <= -c_.max_daily_loss) killed_ = true;
}
RiskDecision RiskManager::check(const Order& o, bool fair_stale, bool book_crossed) {
  if (killed_)       return {false, "kill_switch", 0};
  if (fair_stale)    return {false, "stale_fair_value", 0};
  if (book_crossed)  return {false, "book_crossed", 0};
  Qty qty = std::min(o.qty, c_.max_order_qty);
  Qty cur = pos_.count(o.ticker) ? pos_[o.ticker] : 0;
  Qty allowed = c_.max_qty_per_market - std::abs(cur);
  if (o.action == Action::Buy && cur >= 0) qty = std::min(qty, std::max<Qty>(0, allowed));
  if (o.action == Action::Sell && cur <= 0) qty = std::min(qty, std::max<Qty>(0, allowed));
  if (qty <= 0) return {false, "position_limit", 0};
  return {true, "ok", qty};
}
}
```

In `tools/paper_session.cpp`, replace the `Config` block and the `session_end` output:

```cpp
  Config c;
  c.base_edge = 2 * kCent;
  c.fee_per_contract = 1 * kCent;
  c.confidence_k = 8.0;
  c.max_order_qty = 25 * kContract;
  c.max_qty_per_market = 100 * kContract;
  c.max_aggregate_exposure = 500000 * kMicrosPerCent;
  c.max_daily_loss = 20000 * kMicrosPerCent;
  c.fair_value_max_age_secs = 1800;
```

```cpp
  std::cout << "{\"type\":\"session_end\",\"ticker\":\"" << ticker
            << "\",\"position\":" << venue.position(ticker)
            << ",\"realized_pnl_micros\":" << venue.realized_pnl_micros() << "}\n";
```

- [ ] **Step 6: Rescale the existing tests**

Every change below is a unit rescale: prices ×100, quantities ×100, money ×10,000, config fields renamed. No assertion changes meaning.

`tests/test_order_book.cpp`: replace the four tests' bodies with:

```cpp
TEST(Scaffold, TypesCompile) {
  te::PriceLevel lvl{5500, 1000};
  EXPECT_EQ(lvl.price, 5500);
  EXPECT_EQ(te::kSettleYes, 10000);
}

TEST(OrderBook, SnapshotThenBestPrices) {
  OrderBook b;
  b.apply_snapshot({/*yes*/{{5400,1000},{5300,500}}, /*no*/{{4400,800},{4300,300}}});
  EXPECT_EQ(b.best_yes_bid().value(), 5400);
  EXPECT_EQ(b.best_no_bid().value(), 4400);
  EXPECT_EQ(b.best_yes_ask().value(), 5600); // $1 - best_no_bid(0.44)
  EXPECT_EQ(b.best_no_ask().value(), 4600);  // $1 - best_yes_bid(0.54)
}

TEST(OrderBook, DeltaAddsAndRemovesLevels) {
  OrderBook b;
  b.apply_snapshot({{{5400,1000}},{{4400,800}}});
  b.apply_delta({Side::Yes, 5400, -1000}); // remove all qty at 0.54
  b.apply_delta({Side::Yes, 5500, 700});   // new best yes bid
  EXPECT_EQ(b.best_yes_bid().value(), 5500);
  EXPECT_EQ(b.qty_at(Side::Yes, 5400), 0);
}

TEST(OrderBook, DetectsCrossed) {
  OrderBook b;
  b.apply_snapshot({{{6000,500}},{{4500,500}}}); // yes_bid 0.60, yes_ask 0.55 -> crossed
  EXPECT_TRUE(b.crossed());
}
```

`tests/test_kalshi_messages.cpp`: change the expected values only: snapshot `yes[0].price` → `5400`, `no[0].qty` → `800`; delta `price` → `5500`, `delta_qty` → `-300`.

`tests/test_gateway.cpp`: `EXPECT_EQ(g.book("T").best_yes_bid().value(), 5500);`

`tests/test_config.cpp`:

```cpp
TEST(Config, LoadsEngineJsonAndConvertsUnits) {
  auto c = te::Config::load("config/engine.json");
  EXPECT_EQ(c.max_qty_per_market, 10000);          // 100 contracts
  EXPECT_EQ(c.max_order_qty, 2500);                // 25 contracts
  EXPECT_EQ(c.fee_per_contract, 100);              // 1 cent
  EXPECT_EQ(c.base_edge, 200);                     // 2 cents
  EXPECT_EQ(c.max_daily_loss, 200000000);          // 20000 cents
  EXPECT_EQ(c.max_aggregate_exposure, 5000000000); // 500000 cents
  EXPECT_GT(c.confidence_k, 0.0);
}
```

`tests/test_arb.cpp`:

```cpp
TEST(Arb, DetectsBuyBothWhenAsksUnderPar) {
  OrderBook b; // yes_ask = $1-no_bid, no_ask = $1-yes_bid
  b.apply_snapshot({/*yes bids*/{{4000,500}}, /*no bids*/{{4500,500}}});
  // yes_ask = 0.55, no_ask = 0.60 -> 1.15 (no arb)
  Config c; c.fee_per_contract = 100;
  EXPECT_FALSE(detect_arb(b,c).present);
  OrderBook b2;
  b2.apply_snapshot({{{5200,500}}, {{5300,500}}}); // yes_ask=0.47, no_ask=0.48 -> 0.95 < 1.00-0.02 -> arb
  auto s = detect_arb(b2,c);
  EXPECT_TRUE(s.present);
  EXPECT_EQ(s.action, Action::Buy);
  EXPECT_EQ(s.yes_price, 4700);
  EXPECT_EQ(s.no_price, 4800);
}
```

`tests/test_edge_taker.cpp`:

```cpp
TEST(EdgeTaker, BuysWhenAskBelowFairMinusThreshold) {
  OrderBook b; b.apply_snapshot({{{4000,500}},{{5200,700}}}); // yes_ask = 0.48
  Config c; c.max_order_qty = 2500;
  auto s = detect_take(b, /*fair*/6200, /*threshold*/500, c); // 0.48 <= 0.62-0.05 -> take
  EXPECT_TRUE(s.present);
  EXPECT_EQ(s.action, Action::Buy);
  EXPECT_EQ(s.side, Side::Yes);
  EXPECT_EQ(s.price, 4800);
}
TEST(EdgeTaker, NoTakeInsideThreshold) {
  OrderBook b; b.apply_snapshot({{{4000,500}},{{4000,500}}}); // yes_ask = 0.60
  Config c; c.max_order_qty = 2500;
  EXPECT_FALSE(detect_take(b, 6200, 500, c).present); // 0.60 > 0.57
}
```

`tests/test_market_maker.cpp`:

```cpp
TEST(MarketMaker, SymmetricWhenFlat) {
  Config c; c.base_edge = 200; c.fee_per_contract = 100; c.confidence_k = 8.0; c.max_order_qty = 2500;
  // Threshold 3 cents -> half-spread 3/2 = 1 cent, computed in cents. If the
  // division were done in Price units it would be 1.5 cents (bid 6050).
  auto q = make_quote(6200, 0.95, /*inventory*/0, c);
  EXPECT_EQ(q.bid, 6100);
  EXPECT_EQ(q.ask, 6300);
  EXPECT_EQ(q.size, 2500);
}
TEST(MarketMaker, SkewsDownWhenLong) {
  Config c; c.base_edge = 200; c.fee_per_contract = 100; c.confidence_k = 8.0; c.max_order_qty = 2500;
  auto flat = make_quote(6200, 0.95, 0, c);
  auto lng  = make_quote(6200, 0.95, 5000, c); // 50 contracts long -> quotes shift down
  EXPECT_LT(lng.bid, flat.bid);
  EXPECT_LT(lng.ask, flat.ask);
}
```

`tests/test_pricing.cpp`:

```cpp
TEST(Pricing, FairPriceRoundsAndClamps) {
  EXPECT_EQ(fair_price(0.624), 6200);
  EXPECT_EQ(fair_price(0.001), 100);   // clamp low: 1 cent
  EXPECT_EQ(fair_price(0.999), 9900);  // clamp high: 99 cents
}
TEST(Pricing, EdgeThresholdWidensWhenLessConfident) {
  Config c; c.base_edge = 200; c.fee_per_contract = 100; c.confidence_k = 8.0;
  Price hi = edge_threshold(0.95, c); // 2+1+8*0.05 = 3.4 -> 3 cents
  Price lo = edge_threshold(0.55, c); // 2+1+8*0.45 = 6.6 -> 6 cents
  EXPECT_LT(hi, lo);
  EXPECT_EQ(hi, 300);
  EXPECT_EQ(lo, 600);
}
```

`tests/test_paper_venue.cpp` — rescale all five tests:

```cpp
TEST(Paper, BuyThenSellRealizesPnl) {
  OrderBook book; book.apply_snapshot({{{6000,10000}},{{3500,10000}}}); // yes_ask=0.65, yes_bid=0.60
  PaperVenue v;
  v.place_against(book, {"T", Side::Yes, Action::Buy, 6500, 1000});
  EXPECT_EQ(v.position("T"), 1000);
  // (0.60-0.65) x 10 contracts = -$0.50 = -500,000 micros
  v.place_against(book, {"T", Side::Yes, Action::Sell, 6000, 1000});
  EXPECT_EQ(v.position("T"), 0);
  EXPECT_EQ(v.realized_pnl_micros(), -500000);
}

TEST(Paper, PartialFillToBookQty) {
  OrderBook book; book.apply_snapshot({{{6000,300}},{{3500,300}}}); // only 3 contracts at touch
  PaperVenue v;
  v.place_against(book, {"T", Side::Yes, Action::Buy, 6500, 1000});
  EXPECT_EQ(v.position("T"), 300);
}

TEST(Paper, FlipToShortAveragesCorrectly) {
  OrderBook book; book.apply_snapshot({{{6000,10000}},{{3500,10000}}});
  PaperVenue v;
  v.place_against(book, {"T", Side::Yes, Action::Buy, 6500, 1000});
  EXPECT_EQ(v.position("T"), 1000);
  v.place_against(book, {"T", Side::Yes, Action::Sell, 6000, 1500});
  EXPECT_EQ(v.position("T"), -500);
  EXPECT_EQ(v.realized_pnl_micros(), -500000);
}

TEST(Paper, CostBasisExactOnceFlat) {
  OrderBook ask60; ask60.apply_snapshot({{{5000,10000}},{{4000,10000}}});  // yes_ask = 0.60
  OrderBook ask61; ask61.apply_snapshot({{{5000,10000}},{{3900,10000}}});  // yes_ask = 0.61
  OrderBook bid62; bid62.apply_snapshot({{{6200,10000}},{{3000,10000}}});  // yes_bid = 0.62
  PaperVenue v;
  v.place_against(ask60, {"T", Side::Yes, Action::Buy, 6000, 100});
  v.place_against(ask61, {"T", Side::Yes, Action::Buy, 6100, 100});
  v.place_against(ask61, {"T", Side::Yes, Action::Buy, 6100, 100});
  ASSERT_EQ(v.position("T"), 300);
  for (int i = 0; i < 3; ++i) v.place_against(bid62, {"T", Side::Yes, Action::Sell, 6200, 100});
  EXPECT_EQ(v.position("T"), 0);
  EXPECT_EQ(v.realized_pnl_micros(), 40000);  // exactly 4 cents
}

TEST(Paper, ShortCoverRealizesPnl) {
  OrderBook bid60; bid60.apply_snapshot({{{6000,10000}},{{3000,10000}}});  // yes_bid = 0.60
  OrderBook ask55; ask55.apply_snapshot({{{4000,10000}},{{4500,10000}}});  // yes_ask = 0.55
  PaperVenue v;
  v.place_against(bid60, {"T", Side::Yes, Action::Sell, 6000, 500});
  ASSERT_EQ(v.position("T"), -500);
  v.place_against(ask55, {"T", Side::Yes, Action::Buy, 5500, 500});
  EXPECT_EQ(v.position("T"), 0);
  EXPECT_EQ(v.realized_pnl_micros(), 250000);  // 25 cents
}
```

`tests/test_risk_manager.cpp`:

```cpp
static Order buy(const char* t, Qty qty){ return {t, Side::Yes, Action::Buy, 5000, qty}; }
static Config risk_config() {
  Config c;
  c.max_qty_per_market = 10000; c.max_order_qty = 2500;
  c.max_daily_loss = 200000000; c.max_aggregate_exposure = 5000000000;
  return c;
}
TEST(Risk, BlocksWhenStaleOrCrossedOrKilled) {
  Config c = risk_config();
  RiskManager r(c);
  EXPECT_FALSE(r.check(buy("T",1000), /*stale*/true,  false).allow);
  EXPECT_FALSE(r.check(buy("T",1000), false, /*crossed*/true ).allow);
  r.trip_kill_switch();
  EXPECT_TRUE(r.killed());
  EXPECT_FALSE(r.check(buy("T",1000), false, false).allow);
}
TEST(Risk, CapsQtyToPositionLimit) {
  Config c = risk_config();
  RiskManager r(c);
  r.set_position("T", 9500);                  // only 5 contracts more allowed
  auto d = r.check(buy("T",2500), false, false);
  EXPECT_TRUE(d.allow);
  EXPECT_EQ(d.approved_qty, 500);
}
TEST(Risk, KillsOnMaxDailyLoss) {
  Config c = risk_config();
  RiskManager r(c);
  r.record_realized_pnl(-200000001);
  EXPECT_TRUE(r.killed());
}
```

Keep `Risk.KillFileTripsSwitch` as is, replacing its four config assignment lines with `Config c = risk_config();`.

`tests/test_strategy_engine.cpp`: in all three tests replace the config assignments with the scaled fields:

```cpp
  c.base_edge = 200;
  c.fee_per_contract = 100;
  c.confidence_k = 8.0;
  c.max_order_qty = 2500;
  c.max_qty_per_market = 10000;
  c.max_aggregate_exposure = 5000000000;
  c.max_daily_loss = 200000000;   // third test: 400000 (40 cents)
  c.fair_value_max_age_secs = 1800;
```

Scale every snapshot ×100 on both elements: `{{{40, 50}}, {{52, 50}}}` → `{{{4000, 5000}}, {{5200, 5000}}}`, `{{{30, 50}}, {{50, 50}}}` → `{{{3000, 5000}}, {{5000, 5000}}}`. Then: `ASSERT_EQ(venue.position("T"), 2500);` and `EXPECT_EQ(venue.realized_pnl_micros(), -4500000);`. Update the explanatory comments' cent values to match.

`tests/test_replay.cpp`: in `run_replay`, replace the config block with the scaled one used in `paper_session`, change `ReplayOut` to `struct ReplayOut { std::string log; Qty position; Micros realized; };`, return `venue.realized_pnl_micros()`, rename `MatchesBaseline` to `MatchesPreMigrationBaselineScaled`, and replace its expectations with:

```cpp
  // Pre-migration baseline (spec §2) with prices and quantities x100 and
  // P&L x10,000. Same nine decisions, same order.
  const std::string expected =
    R"({"price":4800,"qty":2500,"seq":0,"ticker":"T","type":"take"})" "\n"
    R"({"ask":6200,"bid":6000,"fair":6200,"seq":1,"size":2500,"ticker":"T","type":"quote"})" "\n"
    R"({"price":4800,"qty":2500,"seq":2,"ticker":"T","type":"take"})" "\n"
    R"({"crossed":true,"has_fv":true,"seq":3,"stale":false,"ticker":"T","type":"skip"})" "\n"
    R"({"price":4800,"qty":2500,"seq":4,"ticker":"T","type":"take"})" "\n"
    R"({"ask":6000,"bid":5800,"fair":6200,"seq":5,"size":2500,"ticker":"T","type":"quote"})" "\n"
    R"({"ask":6000,"bid":5800,"fair":6200,"seq":6,"size":2500,"ticker":"T","type":"quote"})" "\n"
    R"({"price":6600,"qty":2500,"seq":7,"ticker":"T","type":"take"})" "\n"
    R"({"ask":6100,"bid":5900,"fair":6200,"seq":8,"size":2500,"ticker":"T","type":"quote"})" "\n";
  ReplayOut out = run_replay();
  EXPECT_EQ(out.log, expected);
  EXPECT_EQ(out.position, 5000);      // 50 contracts
  EXPECT_EQ(out.realized, 4500000);   // 450 cents
```

- [ ] **Step 7: Build and run everything**

Run: `cmake --build build -j 2>&1 | grep -iE "warning|error" ; ./build/te_tests`
Expected: no warnings or errors from `src/` or `tests/`; every test PASSES, including `Replay.MatchesPreMigrationBaselineScaled`. Also run `./build/paper_session <fv> tests/fixtures/replay_sample.jsonl T` with a `p_yes=0.62, confidence=0.95` file and confirm the last line is `{"type":"session_end","ticker":"T","position":5000,"realized_pnl_micros":4500000}`.

If the baseline test fails, a strategy decision changed. Find it; do not edit the expected string.

- [ ] **Step 8: Commit**

```bash
git add src/core/types.hpp src/core/config.hpp src/core/config.cpp \
  src/market_data/order_book.hpp src/market_data/order_book.cpp src/market_data/kalshi_messages.cpp \
  src/strategy/pricing.hpp src/strategy/pricing.cpp src/strategy/arb.hpp src/strategy/arb.cpp \
  src/strategy/edge_taker.hpp src/strategy/edge_taker.cpp src/strategy/market_maker.hpp src/strategy/market_maker.cpp \
  src/strategy/strategy_engine.hpp src/strategy/strategy_engine.cpp \
  src/risk/risk_manager.hpp src/risk/risk_manager.cpp \
  src/execution/order_venue.hpp src/execution/paper_venue.hpp src/execution/paper_venue.cpp \
  tools/paper_session.cpp \
  tests/test_order_book.cpp tests/test_kalshi_messages.cpp tests/test_gateway.cpp tests/test_config.cpp \
  tests/test_arb.cpp tests/test_edge_taker.cpp tests/test_market_maker.cpp tests/test_pricing.cpp \
  tests/test_paper_venue.cpp tests/test_risk_manager.cpp tests/test_strategy_engine.cpp tests/test_replay.cpp
git commit -F - <<'EOF'
refactor(engine): migrate to integer fixed-point units

Price 1/10,000 dollar, Qty 1/100 contract, money in micro-dollars, all
int64 (spec §3). Config keeps its cent-denominated keys and converts on
load. Strategy arithmetic stays in whole cents and whole contracts and is
scaled at the end, so integer divisions are unchanged.

Behaviour pinned by Replay.MatchesPreMigrationBaselineScaled: the same
nine decisions as before the migration, with values scaled.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 4: `decode_frame` — validating dual-shape decoder

Adds the new decoder alongside the legacy `parse_ws_message`, which the gateway keeps using until Task 7.

**Files:**
- Create: `src/market_data/reject_reason.hpp`
- Modify: `src/market_data/kalshi_messages.hpp`, `src/market_data/kalshi_messages.cpp`
- Test: `tests/test_kalshi_messages.cpp` (append)

**Interfaces:**
- Consumes: `parse_price_dollars`, `parse_qty_fp`, `parse_delta_fp` (Task 1); `Price`, `Qty`, `kCent`, `kContract`, `kOneDollar`, `BookSnapshot`, `BookDelta` (Task 3).
- Produces:
  - `enum class te::RejectReason : uint8_t { None, MalformedJson, MissingField, BadTicker, BadSide, PriceOutOfRange, QtyOutOfRange, BadFixedPoint, DuplicateLevel, DeltaBeforeSnapshot, LevelUnderflow, SeqGap };`
  - `constexpr std::size_t te::kRejectReasonCount = 12;`
  - `std::string_view te::to_string(RejectReason)`
  - `enum class te::FrameKind : uint8_t { Snapshot, Delta, Ignored };`
  - `struct te::DecodedFrame { FrameKind kind; Ticker ticker; std::optional<int64_t> seq; BookSnapshot snapshot; BookDelta delta; };`
  - `struct te::DecodeResult { DecodedFrame frame; RejectReason reject; };`
  - `DecodeResult te::decode_frame(std::string_view json);`
  - `bool te::valid_ticker(std::string_view);`

- [ ] **Step 1: Create the reject vocabulary**

`src/market_data/reject_reason.hpp`:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
namespace te {
// Why a frame was not applied (spec §4). Counted C++-side only; the
// cross-language book-state line carries just ok/rej, never the reason.
enum class RejectReason : uint8_t {
  None,
  MalformedJson,
  MissingField,
  BadTicker,
  BadSide,
  PriceOutOfRange,
  QtyOutOfRange,
  BadFixedPoint,
  DuplicateLevel,
  DeltaBeforeSnapshot,
  LevelUnderflow,
  SeqGap,  // defined for spec 2; nothing produces it yet
};
inline constexpr std::size_t kRejectReasonCount = 12;

inline std::string_view to_string(RejectReason r) {
  switch (r) {
    case RejectReason::None:                return "none";
    case RejectReason::MalformedJson:       return "malformed_json";
    case RejectReason::MissingField:        return "missing_field";
    case RejectReason::BadTicker:           return "bad_ticker";
    case RejectReason::BadSide:             return "bad_side";
    case RejectReason::PriceOutOfRange:     return "price_out_of_range";
    case RejectReason::QtyOutOfRange:       return "qty_out_of_range";
    case RejectReason::BadFixedPoint:       return "bad_fixed_point";
    case RejectReason::DuplicateLevel:      return "duplicate_level";
    case RejectReason::DeltaBeforeSnapshot: return "delta_before_snapshot";
    case RejectReason::LevelUnderflow:      return "level_underflow";
    case RejectReason::SeqGap:              return "seq_gap";
  }
  return "unknown";
}
}
```

- [ ] **Step 2: Write the failing tests**

Add `#include <string>` and `#include "market_data/reject_reason.hpp"` to `tests/test_kalshi_messages.cpp` and append:

```cpp
// ---- decode_frame (spec §4-§5) -------------------------------------------

// Verbatim from Kalshi's docs (docs.kalshi.com/websockets/orderbook-updates).
static const std::string kDocSnapshot =
  R"({"type":"orderbook_snapshot","id":0,"sid":2,"seq":2,"msg":{"market_ticker":"FED-23DEC-T3.00","market_id":"9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1","yes_dollars_fp":[["0.0800","300.00"]],"no_dollars_fp":[["0.5400","20.00"]]}})";
static const std::string kDocDelta =
  R"({"type":"orderbook_delta","sid":2,"seq":3,"msg":{"market_ticker":"FED-23DEC-T3.00","market_id":"9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1","price_dollars":"0.9600","delta_fp":"-54.00","side":"yes","ts_ms":1669149841000,"client_order_id":"optional","subaccount":0}})";

TEST(Decode, DocumentedSnapshotVerbatim) {
  auto r = decode_frame(kDocSnapshot);
  ASSERT_EQ(r.reject, RejectReason::None) << to_string(r.reject);
  EXPECT_EQ(r.frame.kind, FrameKind::Snapshot);
  EXPECT_EQ(r.frame.ticker, "FED-23DEC-T3.00");
  EXPECT_EQ(r.frame.seq, 2);
  ASSERT_EQ(r.frame.snapshot.yes.size(), 1u);
  EXPECT_EQ(r.frame.snapshot.yes[0].price, 800);
  EXPECT_EQ(r.frame.snapshot.yes[0].qty, 30000);
  ASSERT_EQ(r.frame.snapshot.no.size(), 1u);
  EXPECT_EQ(r.frame.snapshot.no[0].price, 5400);
  EXPECT_EQ(r.frame.snapshot.no[0].qty, 2000);
}

TEST(Decode, DocumentedDeltaVerbatim) {
  auto r = decode_frame(kDocDelta);
  ASSERT_EQ(r.reject, RejectReason::None) << to_string(r.reject);
  EXPECT_EQ(r.frame.kind, FrameKind::Delta);
  EXPECT_EQ(r.frame.seq, 3);
  EXPECT_EQ(r.frame.delta.side, Side::Yes);
  EXPECT_EQ(r.frame.delta.price, 9600);
  EXPECT_EQ(r.frame.delta.delta_qty, -5400);
}

TEST(Decode, LegacySnapshotScaled) {
  auto r = decode_frame(R"({"type":"orderbook_snapshot","msg":{"market_ticker":"KXNBA-25-ABC","yes":[[54,10],[53,5]],"no":[[44,8]]}})");
  ASSERT_EQ(r.reject, RejectReason::None);
  EXPECT_FALSE(r.frame.seq.has_value());
  ASSERT_EQ(r.frame.snapshot.yes.size(), 2u);
  EXPECT_EQ(r.frame.snapshot.yes[0].price, 5400);
  EXPECT_EQ(r.frame.snapshot.yes[0].qty, 1000);
  EXPECT_EQ(r.frame.snapshot.no[0].qty, 800);
}

TEST(Decode, LegacyDeltaScaled) {
  auto r = decode_frame(R"({"type":"orderbook_delta","msg":{"market_ticker":"KXNBA-25-ABC","price":55,"delta":-3,"side":"no"}})");
  ASSERT_EQ(r.reject, RejectReason::None);
  EXPECT_EQ(r.frame.delta.side, Side::No);
  EXPECT_EQ(r.frame.delta.price, 5500);
  EXPECT_EQ(r.frame.delta.delta_qty, -300);
}

TEST(Decode, OmittedSideIsEmptyNotMissing) {
  // Kalshi omits an empty side's key entirely.
  auto r = decode_frame(R"({"type":"orderbook_snapshot","seq":4,"msg":{"market_ticker":"T","yes_dollars_fp":[["0.4000","1.00"]]}})");
  ASSERT_EQ(r.reject, RejectReason::None);
  EXPECT_EQ(r.frame.snapshot.yes.size(), 1u);
  EXPECT_TRUE(r.frame.snapshot.no.empty());
}

TEST(Decode, SnapshotWithNoSidesIsEmptyBook) {
  auto r = decode_frame(R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T"}})");
  ASSERT_EQ(r.reject, RejectReason::None);
  EXPECT_TRUE(r.frame.snapshot.yes.empty());
  EXPECT_TRUE(r.frame.snapshot.no.empty());
}

TEST(Decode, DocumentedShapeWinsWhenBothPresent) {
  auto s = decode_frame(R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[1,1]],"yes_dollars_fp":[["0.4000","2.00"]]}})");
  ASSERT_EQ(s.reject, RejectReason::None);
  ASSERT_EQ(s.frame.snapshot.yes.size(), 1u);
  EXPECT_EQ(s.frame.snapshot.yes[0].price, 4000);
  EXPECT_EQ(s.frame.snapshot.yes[0].qty, 200);
  auto d = decode_frame(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":1,"delta":1,"price_dollars":"0.5500","delta_fp":"3.00","side":"yes"}})");
  ASSERT_EQ(d.reject, RejectReason::None);
  EXPECT_EQ(d.frame.delta.price, 5500);
  EXPECT_EQ(d.frame.delta.delta_qty, 300);
}

TEST(Decode, PriceBoundariesAccepted) {
  auto lo = decode_frame(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":"0.0001","delta_fp":"1","side":"yes"}})");
  auto hi = decode_frame(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":"0.9999","delta_fp":"1","side":"yes"}})");
  auto l1 = decode_frame(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":1,"delta":1,"side":"yes"}})");
  auto l99 = decode_frame(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":99,"delta":1,"side":"yes"}})");
  EXPECT_EQ(lo.frame.delta.price, 1);
  EXPECT_EQ(hi.frame.delta.price, 9999);
  EXPECT_EQ(l1.frame.delta.price, 100);
  EXPECT_EQ(l99.frame.delta.price, 9900);
  EXPECT_EQ(lo.reject, RejectReason::None);
  EXPECT_EQ(hi.reject, RejectReason::None);
}

TEST(Decode, UnknownTypesAreIgnoredNotRejected) {
  auto sub = decode_frame(R"({"type":"subscribed","id":1,"msg":{"channel":"orderbook_delta","sid":1}})");
  EXPECT_EQ(sub.frame.kind, FrameKind::Ignored);
  EXPECT_EQ(sub.reject, RejectReason::None);
  auto err = decode_frame(R"({"type":"error","msg":"not an object"})");
  EXPECT_EQ(err.frame.kind, FrameKind::Ignored);
  EXPECT_EQ(err.reject, RejectReason::None);
}

TEST(Decode, TickerRule) {
  EXPECT_TRUE(valid_ticker("KXNBA-25-ABC"));
  EXPECT_TRUE(valid_ticker("FED-23DEC-T3.00"));
  EXPECT_TRUE(valid_ticker("a_b"));
  EXPECT_TRUE(valid_ticker(std::string(64, 'A')));
  EXPECT_FALSE(valid_ticker(""));
  EXPECT_FALSE(valid_ticker(std::string(65, 'A')));
  EXPECT_FALSE(valid_ticker("bad ticker"));
  EXPECT_FALSE(valid_ticker("T\"x"));
  EXPECT_FALSE(valid_ticker("caf\xC3\xA9"));
}

// Every reject: the reason, and the ticker per spec §6's `t` rule. The ticker
// is set iff the frame is well-formed, has no duplicate keys, has a `msg`
// object, and msg.market_ticker passes the ticker rule -- regardless of which
// other rule rejected it.
struct RejectCase {
  std::string name;
  std::string json;
  RejectReason reason;
  std::string ticker;
};

class DecodeRejects : public ::testing::TestWithParam<RejectCase> {};

TEST_P(DecodeRejects, ReasonAndTickerRule) {
  const auto& c = GetParam();
  auto r = decode_frame(c.json);
  EXPECT_EQ(r.reject, c.reason) << "got " << to_string(r.reject);
  EXPECT_EQ(r.frame.ticker, c.ticker);
}

INSTANTIATE_TEST_SUITE_P(Table, DecodeRejects, ::testing::Values(
  // MalformedJson
  RejectCase{"Empty", "", RejectReason::MalformedJson, ""},
  RejectCase{"RootArray", "[1,2]", RejectReason::MalformedJson, ""},
  RejectCase{"TruncatedSnapshot", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[10,1],[11)", RejectReason::MalformedJson, ""},
  RejectCase{"TrailingGarbage", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T"}} x)", RejectReason::MalformedJson, ""},
  RejectCase{"DuplicateTopKey", R"({"type":"orderbook_delta","type":"orderbook_delta","msg":{"market_ticker":"T","price":42,"delta":1,"side":"yes"}})", RejectReason::MalformedJson, ""},
  RejectCase{"DuplicateMsgKey", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":42,"price":43,"delta":1,"side":"yes"}})", RejectReason::MalformedJson, ""},
  RejectCase{"MissingTypeKeepsTicker", R"({"msg":{"market_ticker":"T","price":42,"delta":1,"side":"yes"}})", RejectReason::MalformedJson, "T"},
  RejectCase{"NonStringType", R"({"type":5,"msg":{"market_ticker":"T"}})", RejectReason::MalformedJson, "T"},
  RejectCase{"StringSeq", R"({"type":"orderbook_delta","seq":"3","msg":{"market_ticker":"T","price":42,"delta":1,"side":"yes"}})", RejectReason::MalformedJson, "T"},
  RejectCase{"FloatSeq", R"({"type":"orderbook_delta","seq":3.5,"msg":{"market_ticker":"T","price":42,"delta":1,"side":"yes"}})", RejectReason::MalformedJson, "T"},
  RejectCase{"NegativeSeq", R"({"type":"orderbook_delta","seq":-1,"msg":{"market_ticker":"T","price":42,"delta":1,"side":"yes"}})", RejectReason::MalformedJson, "T"},
  RejectCase{"SideNotArray", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":5}})", RejectReason::MalformedJson, "T"},
  RejectCase{"LevelThreeElements", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[40,1,2]]}})", RejectReason::MalformedJson, "T"},
  RejectCase{"LevelNotArray", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[40]}})", RejectReason::MalformedJson, "T"},
  RejectCase{"MixedShapesAcrossSides", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[40,1]],"no_dollars_fp":[["0.5200","1.00"]]}})", RejectReason::MalformedJson, "T"},
  // MissingField
  RejectCase{"SnapshotNoMsg", R"({"type":"orderbook_snapshot"})", RejectReason::MissingField, ""},
  RejectCase{"MsgNotObject", R"({"type":"orderbook_snapshot","msg":[1]})", RejectReason::MissingField, ""},
  RejectCase{"NoTicker", R"({"type":"orderbook_snapshot","msg":{"yes":[[40,1]]}})", RejectReason::MissingField, ""},
  RejectCase{"DeltaNoSide", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":1}})", RejectReason::MissingField, "T"},
  RejectCase{"LegacyDeltaNoDelta", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"side":"no"}})", RejectReason::MissingField, "T"},
  RejectCase{"DocDeltaNoDeltaFp", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":"0.5000","side":"yes"}})", RejectReason::MissingField, "T"},
  // BadTicker
  RejectCase{"TickerWithSpace", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"bad ticker"}})", RejectReason::BadTicker, ""},
  RejectCase{"TickerEmpty", R"({"type":"orderbook_snapshot","msg":{"market_ticker":""}})", RejectReason::BadTicker, ""},
  RejectCase{"TickerTooLong", std::string(R"({"type":"orderbook_snapshot","msg":{"market_ticker":")") + std::string(65, 'A') + R"("}})", RejectReason::BadTicker, ""},
  RejectCase{"TickerNotString", R"({"type":"orderbook_snapshot","msg":{"market_ticker":5}})", RejectReason::BadTicker, ""},
  RejectCase{"TickerNonAscii", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"café"}})", RejectReason::BadTicker, ""},
  // BadSide
  RejectCase{"SideBanana", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":9,"side":"banana"}})", RejectReason::BadSide, "T"},
  RejectCase{"SideUppercase", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":9,"side":"YES"}})", RejectReason::BadSide, "T"},
  RejectCase{"SideNotString", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":9,"side":1}})", RejectReason::BadSide, "T"},
  // PriceOutOfRange
  RejectCase{"LegacyPrice500", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":500,"delta":9,"side":"yes"}})", RejectReason::PriceOutOfRange, "T"},
  RejectCase{"LegacyPrice0", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":0,"delta":9,"side":"yes"}})", RejectReason::PriceOutOfRange, "T"},
  RejectCase{"LegacyPrice100", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":100,"delta":9,"side":"yes"}})", RejectReason::PriceOutOfRange, "T"},
  RejectCase{"DocPriceOneDollar", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":"1.0000","delta_fp":"1","side":"yes"}})", RejectReason::PriceOutOfRange, "T"},
  RejectCase{"DocPriceZero", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":"0.0000","delta_fp":"1","side":"yes"}})", RejectReason::PriceOutOfRange, "T"},
  RejectCase{"SnapshotLevelPrice0", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[0,1]]}})", RejectReason::PriceOutOfRange, "T"},
  // BadFixedPoint
  RejectCase{"DocPriceFiveDecimals", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":"0.00001","delta_fp":"1","side":"yes"}})", RejectReason::BadFixedPoint, "T"},
  RejectCase{"DocPriceAsNumber", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":0.96,"delta_fp":"1","side":"yes"}})", RejectReason::BadFixedPoint, "T"},
  RejectCase{"LegacyPriceFraction", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":42.5,"delta":1,"side":"yes"}})", RejectReason::BadFixedPoint, "T"},
  RejectCase{"LegacyPriceString", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":"42","delta":1,"side":"yes"}})", RejectReason::BadFixedPoint, "T"},
  RejectCase{"LegacyOverCap", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":42,"delta":10000000000000,"side":"yes"}})", RejectReason::BadFixedPoint, "T"},
  RejectCase{"SnapshotSignedQty", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes_dollars_fp":[["0.4000","-1.00"]]}})", RejectReason::BadFixedPoint, "T"},
  // QtyOutOfRange
  RejectCase{"LegacyDeltaZero", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":0,"side":"no"}})", RejectReason::QtyOutOfRange, "T"},
  RejectCase{"DocDeltaZero", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":"0.5200","delta_fp":"0.00","side":"no"}})", RejectReason::QtyOutOfRange, "T"},
  RejectCase{"DocDeltaNegativeZero", R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price_dollars":"0.5200","delta_fp":"-0.00","side":"no"}})", RejectReason::QtyOutOfRange, "T"},
  RejectCase{"SnapshotQtyZero", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[40,0]]}})", RejectReason::QtyOutOfRange, "T"},
  RejectCase{"SnapshotLegacyNegativeQty", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[40,-5]]}})", RejectReason::QtyOutOfRange, "T"},
  // DuplicateLevel
  RejectCase{"LegacyDuplicatePrice", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[40,1],[40,2]]}})", RejectReason::DuplicateLevel, "T"},
  RejectCase{"DocDuplicateAfterNormalizing", R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes_dollars_fp":[["0.4000","1.00"],["0.40","2.00"]]}})", RejectReason::DuplicateLevel, "T"}
), [](const auto& info) { return info.param.name; });
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cmake --build build -j`
Expected: compile errors — `decode_frame`, `FrameKind`, `valid_ticker` not declared.

- [ ] **Step 4: Declare the interface**

`src/market_data/kalshi_messages.hpp`:

```cpp
#pragma once
#include <cstdint>
#include <optional>
#include <string_view>
#include "market_data/order_book.hpp"
#include "market_data/reject_reason.hpp"
namespace te {

enum class FrameKind : uint8_t { Snapshot, Delta, Ignored };

struct DecodedFrame {
  FrameKind kind{FrameKind::Ignored};
  // Set iff the frame is well-formed JSON with no duplicate keys, has a `msg`
  // object, and msg.market_ticker passes valid_ticker (spec §6) -- including
  // on some rejected frames. Empty otherwise.
  Ticker ticker;
  std::optional<int64_t> seq;  // present only if the frame carried one
  BookSnapshot snapshot;       // Price/Qty units
  BookDelta delta{};           // Price/Qty units, signed delta
};

struct DecodeResult {
  DecodedFrame frame;
  RejectReason reject{RejectReason::None};
};

// Stateless validation and decoding of one WebSocket frame (spec §4-§5).
// Accepts both the legacy integer shape and Kalshi's documented
// *_dollars / *_fp string shape. Never throws on frame content.
DecodeResult decode_frame(std::string_view json);

// 1..64 bytes, each in [A-Za-z0-9._-]. Rules out every character that would
// need escaping in the book-state line.
bool valid_ticker(std::string_view t);

// Legacy decoder, still used by the gateway until Task 7 removes it.
enum class MsgKind { Snapshot, Delta, Other };
struct ParsedMsg { MsgKind kind{MsgKind::Other}; Ticker ticker; BookSnapshot snapshot; BookDelta delta{}; };
ParsedMsg parse_ws_message(std::string_view json);
}
```

- [ ] **Step 5: Implement `decode_frame`**

Add `#include "core/fixed_point.hpp"`, `#include <algorithm>` and `#include <vector>` to the top of `src/market_data/kalshi_messages.cpp` (keep the existing simdjson include block and pragmas). Inside `namespace te {`, before `parse_ws_message`, add:

```cpp
namespace {
using simdjson::dom::array;
using simdjson::dom::element;
using simdjson::dom::object;

// Same magnitude cap as parse_fixed's 12 integer digits.
constexpr int64_t kMaxLegacyMagnitude = 999'999'999'999;

bool has_key(const object& o, std::string_view k) {
  element e;
  return !o.at_key(k).get(e);
}

// JSON allows duplicate keys and implementations disagree about them (Go
// keeps the last, a DOM lookup returns the first), so they are rejected
// outright rather than resolved (spec §5).
bool has_duplicate_keys(const object& o) {
  std::vector<std::string_view> keys;
  for (auto field : o) keys.push_back(field.key);
  std::sort(keys.begin(), keys.end());
  return std::adjacent_find(keys.begin(), keys.end()) != keys.end();
}

std::optional<int64_t> legacy_scaled(const element& e, int64_t scale) {
  int64_t v;
  if (e.get_int64().get(v)) return std::nullopt;  // not a JSON integer
  if (v > kMaxLegacyMagnitude || v < -kMaxLegacyMagnitude) return std::nullopt;
  return v * scale;
}

bool price_in_range(Price p) { return p >= 1 && p <= kOneDollar - 1; }

RejectReason parse_level(const element& lvl, bool documented, PriceLevel& out) {
  array a;
  if (lvl.get_array().get(a) || a.size() != 2) return RejectReason::MalformedJson;
  element pe, qe;
  if (a.at(0).get(pe) || a.at(1).get(qe)) return RejectReason::MalformedJson;
  std::optional<int64_t> price, qty;
  if (documented) {
    std::string_view ps, qs;
    if (pe.get_string().get(ps) || qe.get_string().get(qs)) return RejectReason::BadFixedPoint;
    price = parse_price_dollars(ps);
    qty = parse_qty_fp(qs);
  } else {
    price = legacy_scaled(pe, kCent);
    qty = legacy_scaled(qe, kContract);
  }
  if (!price || !qty) return RejectReason::BadFixedPoint;
  if (!price_in_range(*price)) return RejectReason::PriceOutOfRange;
  if (*qty <= 0) return RejectReason::QtyOutOfRange;
  out = {*price, *qty};
  return RejectReason::None;
}

RejectReason parse_side(const object& msg, std::string_view key, bool documented,
                        std::vector<PriceLevel>& out) {
  element e;
  if (msg.at_key(key).get(e)) return RejectReason::None;  // omitted side == empty (Kalshi docs)
  array levels;
  if (e.get_array().get(levels)) return RejectReason::MalformedJson;
  for (element lvl : levels) {
    PriceLevel pl{};
    if (RejectReason r = parse_level(lvl, documented, pl); r != RejectReason::None) return r;
    out.push_back(pl);
  }
  // Compared after normalizing, so "0.40" and "0.4000" collide.
  std::vector<Price> prices;
  for (const auto& l : out) prices.push_back(l.price);
  std::sort(prices.begin(), prices.end());
  if (std::adjacent_find(prices.begin(), prices.end()) != prices.end())
    return RejectReason::DuplicateLevel;
  return RejectReason::None;
}

RejectReason parse_snapshot(const object& msg, BookSnapshot& out) {
  const bool documented = has_key(msg, "yes_dollars_fp") || has_key(msg, "no_dollars_fp");
  if (documented) {
    // A side given only in legacy form next to a documented side is a mixed
    // frame; no real frame does this (plan clarification 4).
    if ((!has_key(msg, "yes_dollars_fp") && has_key(msg, "yes")) ||
        (!has_key(msg, "no_dollars_fp") && has_key(msg, "no")))
      return RejectReason::MalformedJson;
    if (RejectReason r = parse_side(msg, "yes_dollars_fp", true, out.yes); r != RejectReason::None) return r;
    return parse_side(msg, "no_dollars_fp", true, out.no);
  }
  if (RejectReason r = parse_side(msg, "yes", false, out.yes); r != RejectReason::None) return r;
  return parse_side(msg, "no", false, out.no);
}

RejectReason parse_delta(const object& msg, BookDelta& out) {
  const bool documented = has_key(msg, "price_dollars") || has_key(msg, "delta_fp");
  element pe, de, se;
  if (msg.at_key(documented ? "price_dollars" : "price").get(pe) ||
      msg.at_key(documented ? "delta_fp" : "delta").get(de) ||
      msg.at_key("side").get(se))
    return RejectReason::MissingField;
  std::string_view side;
  if (se.get_string().get(side) || (side != "yes" && side != "no")) return RejectReason::BadSide;
  std::optional<int64_t> price, delta;
  if (documented) {
    std::string_view ps, ds;
    if (pe.get_string().get(ps) || de.get_string().get(ds)) return RejectReason::BadFixedPoint;
    price = parse_price_dollars(ps);
    delta = parse_delta_fp(ds);
  } else {
    price = legacy_scaled(pe, kCent);
    delta = legacy_scaled(de, kContract);
  }
  if (!price || !delta) return RejectReason::BadFixedPoint;
  if (!price_in_range(*price)) return RejectReason::PriceOutOfRange;
  if (*delta == 0) return RejectReason::QtyOutOfRange;
  out = {side == "yes" ? Side::Yes : Side::No, *price, *delta};
  return RejectReason::None;
}
}  // namespace

bool valid_ticker(std::string_view t) {
  if (t.empty() || t.size() > 64) return false;
  for (char ch : t) {
    const bool ok = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                    (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-';
    if (!ok) return false;
  }
  return true;
}

DecodeResult decode_frame(std::string_view json) {
  static thread_local simdjson::dom::parser parser;
  DecodeResult out;
  auto reject = [&out](RejectReason r) { out.reject = r; return out; };

  // DOM rather than on-demand: validates the whole document up front, as Go's
  // encoding/json does, so both engines agree on which frames are well-formed
  // (spec §5).
  element root;
  object top;
  if (parser.parse(json.data(), json.size()).get(root) ||
      root.get_object().get(top) || has_duplicate_keys(top))
    return reject(RejectReason::MalformedJson);

  element msg_el;
  object msg;
  const bool has_msg = !top.at_key("msg").get(msg_el) && !msg_el.get_object().get(msg);
  if (has_msg && has_duplicate_keys(msg)) return reject(RejectReason::MalformedJson);

  // The ticker is fixed by the frame's properties alone (spec §6), before any
  // other rule runs, so it cannot depend on how far validation got.
  if (has_msg) {
    element tick_el;
    std::string_view tick;
    if (!msg.at_key("market_ticker").get(tick_el) && !tick_el.get_string().get(tick) &&
        valid_ticker(tick))
      out.frame.ticker = std::string(tick);
  }

  element type_el;
  std::string_view type;
  if (top.at_key("type").get(type_el) || type_el.get_string().get(type))
    return reject(RejectReason::MalformedJson);
  if (type == "orderbook_snapshot") {
    out.frame.kind = FrameKind::Snapshot;
  } else if (type == "orderbook_delta") {
    out.frame.kind = FrameKind::Delta;
  } else {
    out.frame.kind = FrameKind::Ignored;  // normal traffic (subscribed, ok, error...)
    return out;
  }

  element seq_el;
  if (!top.at_key("seq").get(seq_el)) {
    int64_t s;
    // Non-negative only: -1 is the book-state line's "no seq" sentinel
    // (plan clarification 1).
    if (seq_el.get_int64().get(s) || s < 0) return reject(RejectReason::MalformedJson);
    out.frame.seq = s;
  }

  if (!has_msg) return reject(RejectReason::MissingField);
  element tick_el;
  if (msg.at_key("market_ticker").get(tick_el)) return reject(RejectReason::MissingField);
  if (out.frame.ticker.empty()) return reject(RejectReason::BadTicker);

  return reject(out.frame.kind == FrameKind::Snapshot
                    ? parse_snapshot(msg, out.frame.snapshot)
                    : parse_delta(msg, out.frame.delta));
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cmake --build build -j && ./build/te_tests --gtest_filter='Decode*:Parse*'`
Expected: every `Decode.*` and `Table/DecodeRejects.*` case PASSES; the legacy `Parse.*` tests still pass. Then `./build/te_tests`: all pass.

If a `DecodeRejects` row fails, fix the decoder, not the row, unless the row contradicts the spec or a plan clarification. In that case stop and report.

- [ ] **Step 7: Commit**

```bash
git add src/market_data/reject_reason.hpp src/market_data/kalshi_messages.hpp src/market_data/kalshi_messages.cpp tests/test_kalshi_messages.cpp
git commit -F - <<'EOF'
feat(engine): validating decoder for both Kalshi wire shapes

decode_frame uses simdjson's DOM API and error codes only, accepts the
legacy integer shape and the documented *_dollars/*_fp shape, and returns
a RejectReason instead of throwing. Duplicate keys, bad sides, out-of-range
prices, zero deltas and duplicate levels are rejected; omitted snapshot
sides are empty (spec §4-§5). The gateway switches to it in a later task.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 5: Book-level validation and sequence recording

**Files:**
- Modify: `src/market_data/order_book.hpp`, `src/market_data/order_book.cpp`
- Test: `tests/test_order_book.cpp` (append)

**Interfaces:**
- Consumes: `RejectReason` (Task 4).
- Produces:
  - `RejectReason OrderBook::apply_snapshot(const BookSnapshot& s, std::optional<int64_t> seq = std::nullopt)`
  - `RejectReason OrderBook::apply_delta(const BookDelta& d, std::optional<int64_t> seq = std::nullopt)`
  - `bool OrderBook::has_snapshot() const`
  - `std::optional<int64_t> OrderBook::last_seq() const`
  - `const std::map<Price,Qty>& OrderBook::yes_levels() const`, `no_levels() const`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_order_book.cpp` (add `#include <optional>` and `#include <string>` at the top):

```cpp
// ---- state-dependent validation (spec §4) ---------------------------------

TEST(OrderBook, DeltaBeforeSnapshotRejectedAndBookUntouched) {
  OrderBook b;
  EXPECT_EQ(b.apply_delta({Side::No, 5200, 500}), RejectReason::DeltaBeforeSnapshot);
  EXPECT_FALSE(b.has_snapshot());
  EXPECT_TRUE(b.yes_levels().empty());
  EXPECT_TRUE(b.no_levels().empty());
}

// Each row starts from yes {0.52 x 30} and applies one delta.
struct DeltaCase {
  std::string name;
  BookDelta delta;
  RejectReason expected;
  Qty yes_52_after;   // qty at YES 0.52 afterwards
  Qty no_10_after;    // qty at NO 0.10 afterwards
};
class OrderBookDelta : public ::testing::TestWithParam<DeltaCase> {};

TEST_P(OrderBookDelta, AppliesOrRejectsLeavingBookUnchanged) {
  const auto& c = GetParam();
  OrderBook b;
  b.apply_snapshot({{{5200, 3000}}, {}});
  const auto yes_before = b.yes_levels();
  const auto no_before = b.no_levels();
  EXPECT_EQ(b.apply_delta(c.delta), c.expected);
  EXPECT_EQ(b.qty_at(Side::Yes, 5200), c.yes_52_after);
  EXPECT_EQ(b.qty_at(Side::No, 1000), c.no_10_after);
  if (c.expected != RejectReason::None) {
    EXPECT_EQ(b.yes_levels(), yes_before);
    EXPECT_EQ(b.no_levels(), no_before);
  }
}

INSTANTIATE_TEST_SUITE_P(Table, OrderBookDelta, ::testing::Values(
  DeltaCase{"AddToLevel", {Side::Yes, 5200, 500}, RejectReason::None, 3500, 0},
  DeltaCase{"PartialRemove", {Side::Yes, 5200, -1000}, RejectReason::None, 2000, 0},
  DeltaCase{"ExactZeroErases", {Side::Yes, 5200, -3000}, RejectReason::None, 0, 0},
  DeltaCase{"UnderflowRejected", {Side::Yes, 5200, -99900}, RejectReason::LevelUnderflow, 3000, 0},
  DeltaCase{"NegativeOnAbsentLevel", {Side::No, 1000, -100}, RejectReason::LevelUnderflow, 3000, 0},
  DeltaCase{"NewLevel", {Side::No, 1000, 100}, RejectReason::None, 3000, 100}
), [](const auto& info) { return info.param.name; });

TEST(OrderBook, ExactZeroEraseRemovesTheLevelEntirely) {
  OrderBook b;
  b.apply_snapshot({{{5200, 3000}}, {}});
  ASSERT_EQ(b.apply_delta({Side::Yes, 5200, -3000}), RejectReason::None);
  EXPECT_TRUE(b.yes_levels().empty());  // no zero-qty level left behind
}

TEST(OrderBook, SeqRecordedOnlyOnApply) {
  OrderBook b;
  EXPECT_FALSE(b.last_seq().has_value());
  b.apply_snapshot({{{5200, 3000}}, {}}, 1);
  EXPECT_EQ(b.last_seq(), 1);
  b.apply_delta({Side::Yes, 5200, 100}, 2);
  EXPECT_EQ(b.last_seq(), 2);
  EXPECT_EQ(b.apply_delta({Side::Yes, 5200, -999900}, 3), RejectReason::LevelUnderflow);
  EXPECT_EQ(b.last_seq(), 2);               // rejected frames do not advance it
  b.apply_delta({Side::Yes, 5200, 100});    // no seq on the frame
  EXPECT_EQ(b.last_seq(), 2);               // plan clarification 2
  b.apply_snapshot({{{4000, 100}}, {}});    // snapshot without seq
  EXPECT_EQ(b.last_seq(), 2);
  EXPECT_TRUE(b.has_snapshot());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build -j`
Expected: compile errors — `apply_delta` returns `void`, `has_snapshot`/`yes_levels`/`last_seq` not members.

- [ ] **Step 3: Implement**

`src/market_data/order_book.hpp`:

```cpp
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <vector>
#include "core/types.hpp"
#include "market_data/reject_reason.hpp"
namespace te {
struct BookSnapshot { std::vector<PriceLevel> yes, no; };
struct BookDelta { Side side; Price price; Qty delta_qty; };
class OrderBook {
 public:
  // Both return RejectReason::None when applied. A rejected call leaves the
  // book exactly as it was (spec §4). `seq`, when present, is recorded only
  // when the frame applies; a frame without one leaves it unchanged.
  // Stateless checks (price range, zero delta, duplicate levels) belong to
  // decode_frame; these methods check only what needs book state.
  RejectReason apply_snapshot(const BookSnapshot& s, std::optional<int64_t> seq = std::nullopt);
  RejectReason apply_delta(const BookDelta& d, std::optional<int64_t> seq = std::nullopt);

  std::optional<Price> best_yes_bid() const;
  std::optional<Price> best_no_bid() const;
  std::optional<Price> best_yes_ask() const; // kOneDollar - best_no_bid
  std::optional<Price> best_no_ask() const;  // kOneDollar - best_yes_bid
  Qty qty_at(Side s, Price p) const;
  bool crossed() const;

  bool has_snapshot() const { return has_snapshot_; }
  std::optional<int64_t> last_seq() const { return last_seq_; }
  // Ascending by price; the book-state serializer relies on this order.
  const std::map<Price,Qty>& yes_levels() const { return yes_; }
  const std::map<Price,Qty>& no_levels() const { return no_; }

 private:
  std::map<Price,Qty> yes_;  // price -> qty (YES bids)
  std::map<Price,Qty> no_;   // price -> qty (NO bids)
  bool has_snapshot_ = false;
  std::optional<int64_t> last_seq_;
  static std::optional<Price> best(const std::map<Price,Qty>& m);
};
}
```

In `src/market_data/order_book.cpp`, replace `apply_snapshot` and `apply_delta`:

```cpp
RejectReason OrderBook::apply_snapshot(const BookSnapshot& s, std::optional<int64_t> seq) {
  yes_.clear(); no_.clear();
  for (auto& l : s.yes) if (l.qty > 0) yes_[l.price] = l.qty;
  for (auto& l : s.no)  if (l.qty > 0) no_[l.price]  = l.qty;
  has_snapshot_ = true;
  if (seq) last_seq_ = seq;
  return RejectReason::None;
}

RejectReason OrderBook::apply_delta(const BookDelta& d, std::optional<int64_t> seq) {
  if (!has_snapshot_) return RejectReason::DeltaBeforeSnapshot;
  auto& m = (d.side == Side::Yes) ? yes_ : no_;
  auto it = m.find(d.price);
  const Qty current = (it == m.end()) ? 0 : it->second;
  const Qty next = current + d.delta_qty;
  // Checked before any mutation, so a rejected delta changes nothing.
  if (next < 0) return RejectReason::LevelUnderflow;
  if (next == 0) {
    if (it != m.end()) m.erase(it);
  } else {
    m[d.price] = next;
  }
  if (seq) last_seq_ = seq;
  return RejectReason::None;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build -j && ./build/te_tests`
Expected: new `OrderBook.*` and `Table/OrderBookDelta.*` PASS; every other test still passes (callers ignore the new return value, and every fixture snapshots before its first delta).

- [ ] **Step 5: Commit**

```bash
git add src/market_data/order_book.hpp src/market_data/order_book.cpp tests/test_order_book.cpp
git commit -F - <<'EOF'
feat(engine): book rejects deltas before snapshot and level underflow

apply_snapshot/apply_delta return RejectReason and leave the book
untouched on reject. The book records the seq of the last applied frame
that carried one, and exposes its levels for the book-state serializer
(spec §4).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 6: Book-state line format and writer

**Files:**
- Create: `src/market_data/ingest.hpp`, `src/market_data/book_state.hpp`, `src/market_data/book_state.cpp`
- Test: `tests/test_book_state.cpp`

**Interfaces:**
- Consumes: `OrderBook::yes_levels/no_levels/last_seq` (Task 5); `RejectReason` (Task 4).
- Produces:
  - `enum class te::IngestStatus : uint8_t { Applied, Rejected, Ignored };`
  - `struct te::IngestResult { IngestStatus status; RejectReason reason; Ticker ticker; };`
  - `std::string te::format_book_state_line(int64_t f, const IngestResult& r, const OrderBook* book);`
  - `class te::BookStateWriter { explicit BookStateWriter(std::ostream&); void on_frame(const IngestResult&, const OrderBook* book); };`

- [ ] **Step 1: Write the failing tests**

`tests/test_book_state.cpp`:

```cpp
#include <gtest/gtest.h>
#include <sstream>
#include "market_data/book_state.hpp"
using namespace te;

// Exact strings: this format is the cross-language contract (spec §6).
// Keys f,t,s,seq,yes,no; no whitespace; levels ascending; "\n" terminated.

TEST(BookState, AppliedBookSortedAscendingWithSeq) {
  OrderBook b;
  // Deliberately unsorted input: output order must come from the book.
  b.apply_snapshot({{{900, 1000}, {800, 30000}}, {{5400, 2000}}}, 2);
  EXPECT_EQ(format_book_state_line(0, {IngestStatus::Applied, RejectReason::None, "KXNBA-25-ABC"}, &b),
            "{\"f\":0,\"t\":\"KXNBA-25-ABC\",\"s\":\"ok\",\"seq\":2,"
            "\"yes\":[[800,30000],[900,1000]],\"no\":[[5400,2000]]}\n");
}

TEST(BookState, OneSidedBookEmptySideIsEmptyArray) {
  OrderBook b;
  b.apply_snapshot({{{4000, 100}}, {}});
  EXPECT_EQ(format_book_state_line(7, {IngestStatus::Applied, RejectReason::None, "T"}, &b),
            "{\"f\":7,\"t\":\"T\",\"s\":\"ok\",\"seq\":-1,\"yes\":[[4000,100]],\"no\":[]}\n");
}

TEST(BookState, RejectedFrameShowsUnchangedBook) {
  OrderBook b;
  b.apply_snapshot({{{4000, 100}}, {{5500, 725}}}, 2);
  EXPECT_EQ(format_book_state_line(3, {IngestStatus::Rejected, RejectReason::LevelUnderflow, "T"}, &b),
            "{\"f\":3,\"t\":\"T\",\"s\":\"rej\",\"seq\":2,\"yes\":[[4000,100]],\"no\":[[5500,725]]}\n");
}

TEST(BookState, RejectWithoutTickerIsEmptyBook) {
  EXPECT_EQ(format_book_state_line(5, {IngestStatus::Rejected, RejectReason::MalformedJson, ""}, nullptr),
            "{\"f\":5,\"t\":\"\",\"s\":\"rej\",\"seq\":-1,\"yes\":[],\"no\":[]}\n");
}

TEST(BookState, KnownTickerWithNoBookIsEmptyBook) {
  EXPECT_EQ(format_book_state_line(4, {IngestStatus::Rejected, RejectReason::DeltaBeforeSnapshot, "KXOTHER"}, nullptr),
            "{\"f\":4,\"t\":\"KXOTHER\",\"s\":\"rej\",\"seq\":-1,\"yes\":[],\"no\":[]}\n");
}

TEST(BookState, ZeroSeqPrintsZero) {
  OrderBook b;
  b.apply_snapshot({{}, {}}, 0);
  EXPECT_EQ(format_book_state_line(0, {IngestStatus::Applied, RejectReason::None, "T"}, &b),
            "{\"f\":0,\"t\":\"T\",\"s\":\"ok\",\"seq\":0,\"yes\":[],\"no\":[]}\n");
}

TEST(BookState, WriterCountsIgnoredFramesButDoesNotEmitThem) {
  std::ostringstream os;
  BookStateWriter w(os);
  OrderBook b;
  b.apply_snapshot({{{4000, 100}}, {}});
  w.on_frame({IngestStatus::Ignored, RejectReason::None, ""}, nullptr);  // f = 0, no line
  w.on_frame({IngestStatus::Applied, RejectReason::None, "T"}, &b);      // f = 1
  EXPECT_EQ(os.str(), "{\"f\":1,\"t\":\"T\",\"s\":\"ok\",\"seq\":-1,\"yes\":[[4000,100]],\"no\":[]}\n");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: compile error, `market_data/book_state.hpp` not found.

- [ ] **Step 3: Implement**

`src/market_data/ingest.hpp`:

```cpp
#pragma once
#include <cstdint>
#include "core/types.hpp"
#include "market_data/reject_reason.hpp"
namespace te {
enum class IngestStatus : uint8_t { Applied, Rejected, Ignored };

// Outcome of one frame through the gateway (spec §7). `ticker` follows the
// decoder's rule: empty when the frame never established a valid one.
struct IngestResult {
  IngestStatus status{IngestStatus::Ignored};
  RejectReason reason{RejectReason::None};
  Ticker ticker;
};
}
```

`src/market_data/book_state.hpp`:

```cpp
#pragma once
#include <cstdint>
#include <ostream>
#include <string>
#include "market_data/ingest.hpp"
#include "market_data/order_book.hpp"
namespace te {

// One book-state line (spec §6), the contract the Go port must reproduce byte
// for byte. Written by hand rather than through a JSON library, because
// library formatting is where two correct implementations diverge. `book` is
// null when no book exists for the ticker (or the ticker is empty).
std::string format_book_state_line(int64_t f, const IngestResult& r, const OrderBook* book);

// Streams one line per non-ignored frame. Counts every frame, ignored ones
// included, so `f` always equals the frame's 0-based input index.
class BookStateWriter {
 public:
  explicit BookStateWriter(std::ostream& out) : out_(out) {}
  void on_frame(const IngestResult& r, const OrderBook* book);
 private:
  std::ostream& out_;
  int64_t frame_ = 0;
};
}
```

`src/market_data/book_state.cpp`:

```cpp
#include "market_data/book_state.hpp"
namespace te {
namespace {
void write_side(std::string& s, const OrderBook* book, bool yes) {
  s += '[';
  if (book) {
    bool first = true;
    // std::map iterates in ascending price order.
    for (const auto& [price, qty] : yes ? book->yes_levels() : book->no_levels()) {
      if (!first) s += ',';
      first = false;
      s += '[';
      s += std::to_string(price);
      s += ',';
      s += std::to_string(qty);
      s += ']';
    }
  }
  s += ']';
}
}  // namespace

std::string format_book_state_line(int64_t f, const IngestResult& r, const OrderBook* book) {
  // Tickers are restricted to [A-Za-z0-9._-] (valid_ticker), so no escaping.
  std::string s = "{\"f\":";
  s += std::to_string(f);
  s += ",\"t\":\"";
  s += r.ticker;
  s += "\",\"s\":\"";
  s += (r.status == IngestStatus::Applied) ? "ok" : "rej";
  s += "\",\"seq\":";
  s += std::to_string((book && book->last_seq()) ? *book->last_seq() : int64_t{-1});
  s += ",\"yes\":";
  write_side(s, book, true);
  s += ",\"no\":";
  write_side(s, book, false);
  s += "}\n";
  return s;
}

void BookStateWriter::on_frame(const IngestResult& r, const OrderBook* book) {
  if (r.status != IngestStatus::Ignored) out_ << format_book_state_line(frame_, r, book);
  ++frame_;
}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake -S . -B build && cmake --build build -j && ./build/te_tests --gtest_filter='BookState.*'`
Expected: all 7 PASS. Then `./build/te_tests`: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/market_data/ingest.hpp src/market_data/book_state.hpp src/market_data/book_state.cpp tests/test_book_state.cpp
git commit -F - <<'EOF'
feat(engine): hand-serialized book-state line and writer

One integer-only line per frame with fixed key order, ascending levels
and an ok/rej flag: the contract the Go port must reproduce byte for byte
(spec §6). Ignored frames advance the frame counter without a line.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 7: Gateway error separation, counters and writer hook

Switches the gateway to `decode_frame`, removes the legacy parser, and makes a bad frame a counted value instead of an exception that reaches the reconnect handler.

**Files:**
- Modify: `src/market_data/gateway.hpp`, `src/market_data/gateway.cpp`, `src/market_data/gateway_run.cpp`, `src/market_data/kalshi_messages.hpp`, `src/market_data/kalshi_messages.cpp`
- Test: `tests/test_gateway.cpp`, `tests/test_kalshi_messages.cpp`

**Interfaces:**
- Consumes: `decode_frame`, `DecodeResult`, `FrameKind` (Task 4); `OrderBook::apply_*` returning `RejectReason` (Task 5); `IngestResult`, `BookStateWriter` (Task 6).
- Produces:
  - `IngestResult MarketDataGateway::handle_raw(std::string_view json)`
  - `void MarketDataGateway::set_book_state_writer(BookStateWriter* w)`
  - `bool MarketDataGateway::has_book(const Ticker&) const`
  - `const std::array<uint64_t, kRejectReasonCount>& MarketDataGateway::reject_counts() const`
  - `class te::RejectLogLimiter { bool should_log(RejectReason r, int64_t now_ms); };`
  - `parse_ws_message`, `MsgKind`, `ParsedMsg` are **deleted**.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_gateway.cpp` with:

```cpp
#include <gtest/gtest.h>
#include <sstream>
#include "market_data/book_state.hpp"
#include "market_data/gateway.hpp"
using namespace te;

static const char* kSnap = R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[54,10]],"no":[[44,8]]}})";

TEST(Gateway, RoutesSnapshotThenDeltaToBookAndFiresCallback) {
  MarketDataGateway g;
  int calls = 0; Ticker last;
  g.on_update([&](const Ticker& t, const OrderBook&){ ++calls; last = t; });
  EXPECT_EQ(g.handle_raw(kSnap).status, IngestStatus::Applied);
  auto r = g.handle_raw(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":55,"delta":7,"side":"yes"}})");
  EXPECT_EQ(r.status, IngestStatus::Applied);
  EXPECT_EQ(r.ticker, "T");
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(last, "T");
  EXPECT_EQ(g.book("T").best_yes_bid().value(), 5500);
}

TEST(Gateway, RejectedFrameIsCountedNotThrownAndDoesNotFireCallback) {
  MarketDataGateway g;
  int calls = 0;
  g.on_update([&](const Ticker&, const OrderBook&){ ++calls; });
  g.handle_raw(kSnap);
  IngestResult r;
  EXPECT_NO_THROW(r = g.handle_raw(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":9,"side":"banana"}})"));
  EXPECT_EQ(r.status, IngestStatus::Rejected);
  EXPECT_EQ(r.reason, RejectReason::BadSide);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(g.reject_counts()[static_cast<size_t>(RejectReason::BadSide)], 1u);
  EXPECT_EQ(g.book("T").best_yes_bid().value(), 5400);  // unchanged
}

TEST(Gateway, MalformedFrameDoesNotThrow) {
  MarketDataGateway g;
  IngestResult r;
  EXPECT_NO_THROW(r = g.handle_raw(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"side":"no"}})"));
  EXPECT_EQ(r.status, IngestStatus::Rejected);
  EXPECT_EQ(r.reason, RejectReason::MissingField);
}

TEST(Gateway, DeltaForUnknownTickerCreatesNoBook) {
  MarketDataGateway g;
  auto r = g.handle_raw(R"({"type":"orderbook_delta","msg":{"market_ticker":"X","price":52,"delta":5,"side":"no"}})");
  EXPECT_EQ(r.status, IngestStatus::Rejected);
  EXPECT_EQ(r.reason, RejectReason::DeltaBeforeSnapshot);
  EXPECT_FALSE(g.has_book("X"));
}

TEST(Gateway, IgnoredFrameCountsNothingAndFiresNothing) {
  MarketDataGateway g;
  int calls = 0;
  g.on_update([&](const Ticker&, const OrderBook&){ ++calls; });
  auto r = g.handle_raw(R"({"type":"subscribed","id":1,"msg":{"channel":"orderbook_delta","sid":1}})");
  EXPECT_EQ(r.status, IngestStatus::Ignored);
  EXPECT_EQ(calls, 0);
  for (auto n : g.reject_counts()) EXPECT_EQ(n, 0u);
}

TEST(Gateway, WriterSeesEveryFrame) {
  std::ostringstream os;
  BookStateWriter w(os);
  MarketDataGateway g;
  g.set_book_state_writer(&w);
  g.handle_raw(R"({"type":"subscribed","id":1,"msg":{}})");  // f = 0, no line
  g.handle_raw(kSnap);                                       // f = 1
  g.handle_raw("{");                                         // f = 2
  EXPECT_EQ(os.str(),
            "{\"f\":1,\"t\":\"T\",\"s\":\"ok\",\"seq\":-1,\"yes\":[[5400,1000]],\"no\":[[4400,800]]}\n"
            "{\"f\":2,\"t\":\"\",\"s\":\"rej\",\"seq\":-1,\"yes\":[],\"no\":[]}\n");
}

TEST(RejectLogLimiter, OneLinePerReasonPerSecond) {
  RejectLogLimiter lim;
  EXPECT_TRUE(lim.should_log(RejectReason::BadSide, 1000));
  EXPECT_FALSE(lim.should_log(RejectReason::BadSide, 1999));
  EXPECT_TRUE(lim.should_log(RejectReason::MalformedJson, 1500));  // reasons are independent
  EXPECT_TRUE(lim.should_log(RejectReason::BadSide, 2000));
  EXPECT_FALSE(lim.should_log(RejectReason::BadSide, 2500));
}
```

In `tests/test_kalshi_messages.cpp`, delete the three legacy tests `TEST(Parse, Snapshot)`, `TEST(Parse, Delta)`, `TEST(Parse, Other)`. `Decode.LegacySnapshotScaled`, `Decode.LegacyDeltaScaled` and `Decode.UnknownTypesAreIgnoredNotRejected` cover the same behaviour through the new decoder.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build -j`
Expected: compile errors — `handle_raw` returns `void`, `has_book`, `set_book_state_writer`, `reject_counts`, `RejectLogLimiter` undeclared.

- [ ] **Step 3: Implement the gateway**

`src/market_data/gateway.hpp`:

```cpp
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "market_data/book_state.hpp"
#include "market_data/ingest.hpp"
#include "market_data/order_book.hpp"
#include "market_data/reject_reason.hpp"
namespace te {

class MarketDataGateway {
 public:
  using UpdateCb = std::function<void(const Ticker&, const OrderBook&)>;
  void on_update(UpdateCb cb) { cb_ = std::move(cb); }
  // Optional book-state sink (spec §6). Off by default; the live engine never sets one.
  void set_book_state_writer(BookStateWriter* w) { writer_ = w; }

  // Decodes and applies one frame. Never throws on frame content: every
  // outcome is a value, and a rejected frame changes no book (spec §7). The
  // update callback fires only on Applied.
  IngestResult handle_raw(std::string_view json);

  const OrderBook& book(const Ticker& t) const { return books_.at(t); }
  bool has_book(const Ticker& t) const { return books_.count(t) != 0; }
  const std::array<uint64_t, kRejectReasonCount>& reject_counts() const { return reject_counts_; }

  void run(const std::vector<Ticker>& watchlist); // live WS connect+auth+subscribe; see gateway_run.cpp
  void stop() { stop_ = true; }                   // request run()'s read loop to halt

 private:
  std::unordered_map<Ticker, OrderBook> books_;
  UpdateCb cb_;
  BookStateWriter* writer_ = nullptr;
  std::array<uint64_t, kRejectReasonCount> reject_counts_{};
  std::atomic<bool> stop_{false};
};

// At most one log line per reject reason per second, so a feed that turns
// bad cannot flood stderr (spec §7). Clock injected so it is testable.
class RejectLogLimiter {
 public:
  RejectLogLimiter() { last_ms_.fill(kNever); }
  bool should_log(RejectReason r, int64_t now_ms) {
    int64_t& last = last_ms_[static_cast<size_t>(r)];
    if (last != kNever && now_ms - last < 1000) return false;
    last = now_ms;
    return true;
  }
 private:
  static constexpr int64_t kNever = std::numeric_limits<int64_t>::min();
  std::array<int64_t, kRejectReasonCount> last_ms_{};
};
}
```

`src/market_data/gateway.cpp`:

```cpp
#include "market_data/gateway.hpp"
#include "market_data/kalshi_messages.hpp"
namespace te {
IngestResult MarketDataGateway::handle_raw(std::string_view json) {
  DecodeResult d = decode_frame(json);
  IngestResult r;
  r.ticker = d.frame.ticker;

  if (d.reject != RejectReason::None) {
    r.status = IngestStatus::Rejected;
    r.reason = d.reject;
  } else if (d.frame.kind == FrameKind::Ignored) {
    r.status = IngestStatus::Ignored;
  } else if (d.frame.kind == FrameKind::Snapshot) {
    books_[d.frame.ticker].apply_snapshot(d.frame.snapshot, d.frame.seq);  // a snapshot creates the book
    r.status = IngestStatus::Applied;
  } else {
    // A delta must never create a book (spec §4): find, not operator[].
    auto it = books_.find(d.frame.ticker);
    const RejectReason rr = (it == books_.end())
        ? RejectReason::DeltaBeforeSnapshot
        : it->second.apply_delta(d.frame.delta, d.frame.seq);
    r.status = (rr == RejectReason::None) ? IngestStatus::Applied : IngestStatus::Rejected;
    r.reason = rr;
  }

  if (r.status == IngestStatus::Rejected) ++reject_counts_[static_cast<size_t>(r.reason)];

  if (writer_) {
    const OrderBook* book = nullptr;
    if (!r.ticker.empty()) {
      auto it = books_.find(r.ticker);
      if (it != books_.end()) book = &it->second;
    }
    writer_->on_frame(r, book);
  }

  if (r.status == IngestStatus::Applied && cb_) cb_(r.ticker, books_.at(r.ticker));
  return r;
}
}
```

In `src/market_data/kalshi_messages.hpp`, delete the legacy block (the comment line, `enum class MsgKind`, `struct ParsedMsg`, and `parse_ws_message`). In `src/market_data/kalshi_messages.cpp`, delete the whole `ParsedMsg parse_ws_message(...) { ... }` function.

- [ ] **Step 4: Separate bad frames from dead sockets in the live loop**

In `src/market_data/gateway_run.cpp`, replace the read loop (from `beast::flat_buffer buffer;` through the closing brace of the inner `while (!stop_)`) with:

```cpp
      RejectLogLimiter limiter;
      beast::flat_buffer buffer;
      while (!stop_) {
        buffer.clear();
        ws.read(buffer);
        if (!ws.got_text()) continue;
        // A bad frame is counted and logged, never treated as a dead socket:
        // handle_raw cannot throw on frame content, so the catch below only
        // ever sees Beast/Asio/SSL failures (spec §7).
        const IngestResult r = handle_raw(std::string_view(
            static_cast<const char*>(buffer.data().data()), buffer.size()));
        if (r.status == IngestStatus::Rejected && limiter.should_log(r.reason, now_ms())) {
          std::cerr << "[gateway] rejected frame: " << to_string(r.reason);
          if (!r.ticker.empty()) std::cerr << " ticker=" << r.ticker;
          std::cerr << " total=" << reject_counts_[static_cast<size_t>(r.reason)] << '\n';
        }
      }
```

Do not touch anything above `beast::flat_buffer` (credentials, TLS, handshake, subscribe) or the `catch` block. The live path has no automated test and must not be run; compiling cleanly is this step's bar.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j 2>&1 | grep -iE "warning|error" ; ./build/te_tests`
Expected: no warnings or errors from our sources; all `Gateway.*` and `RejectLogLimiter.*` tests PASS; `Replay.*` still passes (the fixture is all valid legacy frames). Confirm the legacy parser is gone: `grep -rn "parse_ws_message\|ParsedMsg\|MsgKind" src tests tools` prints nothing.

- [ ] **Step 6: Commit**

```bash
git add src/market_data/gateway.hpp src/market_data/gateway.cpp src/market_data/gateway_run.cpp \
  src/market_data/kalshi_messages.hpp src/market_data/kalshi_messages.cpp \
  tests/test_gateway.cpp tests/test_kalshi_messages.cpp
git commit -F - <<'EOF'
feat(engine): bad frames are counted, never treated as a dead socket

handle_raw decodes via decode_frame, returns an IngestResult and cannot
throw on frame content. Rejected frames change no book, are counted per
reason, and do not fire the strategy callback; a delta for an unknown
ticker no longer creates a book. The live read loop logs rejects (at most
one line per reason per second) and continues, so its catch block now only
sees transport failures (spec §7). Removes the legacy parse_ws_message.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 8: Audit regression table and no-throw corpus

Pins the eight inputs from the September audit (spec §2) to their spec-1 outcomes, and proves no input shape can make `handle_raw` throw.

**Files:**
- Create: `tests/test_ingest_audit.cpp`, `tests/test_ingest_no_throw.cpp`

**Interfaces:**
- Consumes: `MarketDataGateway::handle_raw`, `set_book_state_writer`, `book`, `has_book` (Task 7); `BookStateWriter`, `format_book_state_line` (Task 6).
- Produces: nothing new.

- [ ] **Step 1: Write the audit table**

`tests/test_ingest_audit.cpp`:

```cpp
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>
#include "market_data/book_state.hpp"
#include "market_data/gateway.hpp"
using namespace te;

// The eight inputs a source-level probe of the pre-hardening engine showed
// being mishandled (spec §2), each with its spec-1 outcome (spec §9). Every
// row asserts the exact book-state line the probe frame emits, which shows
// the book after the frame rather than merely asserting it is unchanged.

static const std::string kSnapT =
  R"({"type":"orderbook_snapshot","sid":1,"seq":1,"msg":{"market_ticker":"T","yes":[[40,50]],"no":[[52,50]]}})";

struct AuditCase {
  std::string name;
  std::vector<std::string> setup;   // frames fed before the probe
  std::string probe;
  IngestStatus status;
  RejectReason reason;
  std::string line;                 // exact book-state line the probe emits
};

class IngestAudit : public ::testing::TestWithParam<AuditCase> {};

TEST_P(IngestAudit, ProbeHasSpecOneOutcome) {
  const auto& c = GetParam();
  std::ostringstream os;
  BookStateWriter w(os);
  MarketDataGateway g;
  g.set_book_state_writer(&w);
  for (const auto& f : c.setup) ASSERT_EQ(g.handle_raw(f).status, IngestStatus::Applied);

  // Book T before the probe, in the contract format, for the rows whose own
  // line cannot show it (t:"" or another ticker).
  const std::string before = g.has_book("T")
      ? format_book_state_line(0, {IngestStatus::Applied, RejectReason::None, "T"}, &g.book("T")) : "";

  const std::size_t mark = os.str().size();
  IngestResult r;
  ASSERT_NO_THROW(r = g.handle_raw(c.probe));
  EXPECT_EQ(r.status, c.status);
  EXPECT_EQ(r.reason, c.reason) << to_string(r.reason);
  EXPECT_EQ(os.str().substr(mark), c.line);

  if (c.status == IngestStatus::Rejected && g.has_book("T")) {
    EXPECT_EQ(format_book_state_line(0, {IngestStatus::Applied, RejectReason::None, "T"}, &g.book("T")), before)
        << "a rejected frame changed book T";
  }
}

INSTANTIATE_TEST_SUITE_P(Table, IngestAudit, ::testing::Values(
  AuditCase{"DeltaBeforeSnapshot", {},
    R"({"type":"orderbook_delta","msg":{"market_ticker":"X","price":52,"delta":5,"side":"no"}})",
    IngestStatus::Rejected, RejectReason::DeltaBeforeSnapshot,
    "{\"f\":0,\"t\":\"X\",\"s\":\"rej\",\"seq\":-1,\"yes\":[],\"no\":[]}\n"},
  // Applied under spec 1: gap DETECTION is spec 2. This row records today's
  // behaviour so spec 2 has a failing row to flip.
  AuditCase{"SeqGapAppliedUntilSpec2", {kSnapT},
    R"({"type":"orderbook_delta","sid":1,"seq":7,"msg":{"market_ticker":"T","price":52,"delta":-20,"side":"no"}})",
    IngestStatus::Applied, RejectReason::None,
    "{\"f\":1,\"t\":\"T\",\"s\":\"ok\",\"seq\":7,\"yes\":[[4000,5000]],\"no\":[[5200,3000]]}\n"},
  AuditCase{"LevelUnderflow",
    {kSnapT, R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":-20,"side":"no"}})"},
    R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":-999,"side":"no"}})",
    IngestStatus::Rejected, RejectReason::LevelUnderflow,
    "{\"f\":2,\"t\":\"T\",\"s\":\"rej\",\"seq\":1,\"yes\":[[4000,5000]],\"no\":[[5200,3000]]}\n"},
  AuditCase{"SideBanana", {kSnapT},
    R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"delta":9,"side":"banana"}})",
    IngestStatus::Rejected, RejectReason::BadSide,
    "{\"f\":1,\"t\":\"T\",\"s\":\"rej\",\"seq\":1,\"yes\":[[4000,5000]],\"no\":[[5200,5000]]}\n"},
  AuditCase{"Price500", {kSnapT},
    R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":500,"delta":9,"side":"yes"}})",
    IngestStatus::Rejected, RejectReason::PriceOutOfRange,
    "{\"f\":1,\"t\":\"T\",\"s\":\"rej\",\"seq\":1,\"yes\":[[4000,5000]],\"no\":[[5200,5000]]}\n"},
  AuditCase{"TruncatedSnapshot", {kSnapT},
    R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[10,1],[11)",
    IngestStatus::Rejected, RejectReason::MalformedJson,
    "{\"f\":1,\"t\":\"\",\"s\":\"rej\",\"seq\":-1,\"yes\":[],\"no\":[]}\n"},
  AuditCase{"DeltaMissingDelta", {kSnapT},
    R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":52,"side":"no"}})",
    IngestStatus::Rejected, RejectReason::MissingField,
    "{\"f\":1,\"t\":\"T\",\"s\":\"rej\",\"seq\":1,\"yes\":[[4000,5000]],\"no\":[[5200,5000]]}\n"},
  // Applied: Kalshi omits an empty side's key, so this is a valid one-sided
  // snapshot. Before spec 1 it threw and forced a reconnect.
  AuditCase{"SnapshotOmittedNoSide", {kSnapT},
    R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[41,1]]}})",
    IngestStatus::Applied, RejectReason::None,
    "{\"f\":1,\"t\":\"T\",\"s\":\"ok\",\"seq\":1,\"yes\":[[4100,100]],\"no\":[]}\n"}
), [](const auto& info) { return info.param.name; });
```

- [ ] **Step 2: Write the no-throw corpus**

`tests/test_ingest_no_throw.cpp`:

```cpp
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>
#include "market_data/gateway.hpp"
using namespace te;

// Every strict prefix and a set of single-byte corruptions of every real
// frame we have, fed through handle_raw (spec §9). Nothing may throw, and a
// strict prefix of a frame may never apply. Deterministic, so it runs in CI
// unlike a fuzzer.
namespace {
std::vector<std::string> corpus() {
  std::vector<std::string> frames;
  std::ifstream f("tests/fixtures/replay_sample.jsonl");
  std::string line;
  while (std::getline(f, line)) frames.push_back(line);
  frames.push_back(R"({"type":"orderbook_snapshot","id":0,"sid":2,"seq":2,"msg":{"market_ticker":"FED-23DEC-T3.00","market_id":"9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1","yes_dollars_fp":[["0.0800","300.00"]],"no_dollars_fp":[["0.5400","20.00"]]}})");
  frames.push_back(R"({"type":"orderbook_delta","sid":2,"seq":3,"msg":{"market_ticker":"FED-23DEC-T3.00","market_id":"9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1","price_dollars":"0.9600","delta_fp":"-54.00","side":"yes","ts_ms":1669149841000,"client_order_id":"optional","subaccount":0}})");
  return frames;
}

// Seeds books so deltas in the corpus have something to apply against. Takes
// the gateway by reference: MarketDataGateway holds a std::atomic and cannot
// be copied or moved.
void seed(MarketDataGateway& g) {
  g.handle_raw(R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[40,50],[95,50],[66,50]],"no":[[52,50],[30,50]]}})");
  g.handle_raw(R"({"type":"orderbook_snapshot","msg":{"market_ticker":"FED-23DEC-T3.00","yes":[[96,100]],"no":[[4,100]]}})");
}
}  // namespace

TEST(IngestNoThrow, CorpusLoaded) {
  EXPECT_EQ(corpus().size(), 11u);  // 9 fixture frames + 2 documented frames
}

TEST(IngestNoThrow, StrictPrefixesNeverThrowOrApply) {
  for (const auto& frame : corpus()) {
    MarketDataGateway g;
    seed(g);
    for (std::size_t n = 0; n < frame.size(); ++n) {
      const std::string prefix = frame.substr(0, n);
      IngestResult r;
      ASSERT_NO_THROW(r = g.handle_raw(prefix)) << prefix;
      EXPECT_NE(r.status, IngestStatus::Applied) << prefix;
    }
  }
}

TEST(IngestNoThrow, SingleByteCorruptionsNeverThrow) {
  const std::string replacements = std::string("\"{}[],:0x-.") + '\0' + '\xff';
  for (const auto& frame : corpus()) {
    for (std::size_t i = 0; i < frame.size(); ++i) {
      for (char ch : replacements) {
        std::string bad = frame;
        bad[i] = ch;
        MarketDataGateway g;
        seed(g);
        ASSERT_NO_THROW(g.handle_raw(bad)) << "byte " << i << " -> " << int(static_cast<unsigned char>(ch));
      }
    }
  }
}
```

- [ ] **Step 3: Run the tests**

Run: `cmake -S . -B build && cmake --build build -j && ./build/te_tests --gtest_filter='Table/IngestAudit.*:IngestNoThrow.*'`
Expected: all PASS. These tests pin behaviour already implemented in Tasks 4–7; if one fails, the implementation disagrees with the spec. Fix the implementation; change a row only if it contradicts the spec, and then stop and report.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ingest_audit.cpp tests/test_ingest_no_throw.cpp
git commit -F - <<'EOF'
test(engine): pin the ingest audit's eight inputs and a no-throw corpus

Each audit input asserts its exact book-state line: six rejected with the
book unchanged, the omitted-side snapshot applied, and the seq gap applied
pending spec 2. Every strict prefix and single-byte corruption of every
real frame goes through handle_raw without throwing, and no strict prefix
applies (spec §9).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 9: `book_replay` tool and hand-derived golden outputs

**Files:**
- Create: `tools/book_replay.cpp`, `tests/fixtures/replay_sample.book_state`, `tests/fixtures/ingest_mixed.jsonl`, `tests/fixtures/ingest_mixed.book_state`
- Modify: `CMakeLists.txt`, `tests/test_replay.cpp`

**Why a second fixture.** The 9-frame `replay_sample.jsonl` is all valid legacy frames for one ticker, so its golden never exercises ignored frames, rejects, `t:""`, the documented shape, or `seq`. `ingest_mixed.jsonl` is a 10-frame **synthetic** fixture covering exactly those, so the spec-4 Go port has an end-to-end target for every line shape. It is labeled synthetic in its test comment.

**Interfaces:**
- Consumes: `MarketDataGateway`, `BookStateWriter`, `reject_counts`, `to_string` (Tasks 4–7).
- Produces: `build/book_replay <frames.jsonl>` — book-state lines on stdout, nonzero reject counts on stderr. The Go port (spec 4) ships an equivalent.

- [ ] **Step 1: Write the golden files by hand**

These are derived by tracing the fixture frame by frame, **not** by running the code. A golden produced by the implementation under test only proves the code agrees with itself.

`tests/fixtures/replay_sample.book_state` (from `replay_sample.jsonl`; legacy shape, no `seq`, prices ×100, qty ×100):

```
{"f":0,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000]],"no":[[5200,5000]]}
{"f":1,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000]],"no":[]}
{"f":2,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000]],"no":[[5200,5000]]}
{"f":3,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000],[9500,1000]],"no":[[5200,5000]]}
{"f":4,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000]],"no":[[5200,5000]]}
{"f":5,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000]],"no":[]}
{"f":6,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000]],"no":[[3000,4000]]}
{"f":7,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000],[6600,3000]],"no":[[3000,4000]]}
{"f":8,"t":"T","s":"ok","seq":-1,"yes":[[4000,5000]],"no":[[3000,4000]]}
```

`tests/fixtures/ingest_mixed.jsonl` (exactly these 10 lines; line 5 is deliberately truncated, line 8 deliberately has a duplicate key):

```
{"type":"subscribed","id":1,"msg":{"channel":"orderbook_delta","sid":1}}
{"type":"orderbook_snapshot","sid":1,"seq":1,"msg":{"market_ticker":"KXNBA-25-ABC","yes_dollars_fp":[["0.4000","10.00"],["0.3950","2.50"]]}}
{"type":"orderbook_delta","sid":1,"seq":2,"msg":{"market_ticker":"KXNBA-25-ABC","price_dollars":"0.5500","delta_fp":"7.25","side":"no"}}
{"type":"orderbook_delta","sid":1,"seq":3,"msg":{"market_ticker":"KXNBA-25-ABC","price_dollars":"0.4000","delta_fp":"-20.00","side":"yes"}}
{"type":"orderbook_delta","sid":1,"seq":4,"msg":{"market_ticker":"KXOTHER","price_dollars":"0.1000","delta_fp":"1.00","side":"yes"}}
{"type":"orderbook_delta","sid":1,"seq":5,"msg":{"market_ti
{"type":"orderbook_delta","msg":{"market_ticker":"KXNBA-25-ABC","price":40,"delta":-10,"side":"yes"}}
{"type":"orderbook_delta","sid":1,"seq":7,"msg":{"market_ticker":"KXNBA-25-ABC","price_dollars":"0.5500","delta_fp":"1.00","side":"banana"}}
{"type":"orderbook_delta","sid":1,"seq":8,"msg":{"market_ticker":"KXNBA-25-ABC","price_dollars":"0.5500","price_dollars":"0.5600","delta_fp":"1.00","side":"no"}}
{"type":"orderbook_snapshot","sid":1,"seq":9,"msg":{"market_ticker":"KXNBA-25-ABC","no_dollars_fp":[["0.6000","1.00"]]}}
```

`tests/fixtures/ingest_mixed.book_state`:

```
{"f":1,"t":"KXNBA-25-ABC","s":"ok","seq":1,"yes":[[3950,250],[4000,1000]],"no":[]}
{"f":2,"t":"KXNBA-25-ABC","s":"ok","seq":2,"yes":[[3950,250],[4000,1000]],"no":[[5500,725]]}
{"f":3,"t":"KXNBA-25-ABC","s":"rej","seq":2,"yes":[[3950,250],[4000,1000]],"no":[[5500,725]]}
{"f":4,"t":"KXOTHER","s":"rej","seq":-1,"yes":[],"no":[]}
{"f":5,"t":"","s":"rej","seq":-1,"yes":[],"no":[]}
{"f":6,"t":"KXNBA-25-ABC","s":"ok","seq":2,"yes":[[3950,250]],"no":[[5500,725]]}
{"f":7,"t":"KXNBA-25-ABC","s":"rej","seq":2,"yes":[[3950,250]],"no":[[5500,725]]}
{"f":8,"t":"","s":"rej","seq":-1,"yes":[],"no":[]}
{"f":9,"t":"KXNBA-25-ABC","s":"ok","seq":9,"yes":[],"no":[[6000,100]]}
```

Every file ends with exactly one trailing newline and has no blank lines. Check with `tail -c 1 FILE | xxd` (expect `0a`) and `grep -c '^$' FILE` (expect `0`).

- [ ] **Step 2: Write the failing golden tests**

Add to `tests/test_replay.cpp` (add `#include "market_data/book_state.hpp"` and `#include <iterator>`):

```cpp
namespace {
std::string read_file(const std::string& path) {
  std::ifstream f(path);
  EXPECT_TRUE(f.good()) << "cannot open " << path << " (expected working directory: trading_engine/)";
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Same loop as tools/book_replay.cpp: every line is a frame (plan
// clarification 7), so `f` equals the 0-based line number.
std::string book_replay(const std::string& path) {
  std::ifstream in(path);
  EXPECT_TRUE(in.good()) << "cannot open " << path;
  std::ostringstream out;
  BookStateWriter w(out);
  MarketDataGateway gw;
  gw.set_book_state_writer(&w);
  std::string line;
  while (std::getline(in, line)) gw.handle_raw(line);
  return out.str();
}
}  // namespace

/* Hand derivation of replay_sample.book_state (legacy shape: price x100,
   qty x100; no frame carries seq, so seq stays -1 throughout):
   f0 snapshot yes{40:50} no{52:50}  -> yes[[4000,5000]] no[[5200,5000]]
   f1 no 52 -50: 5000-5000=0 erase   -> no[]
   f2 no 52 +50                      -> no[[5200,5000]]
   f3 yes 95 +10                     -> yes[[4000,5000],[9500,1000]]
   f4 yes 95 -10: erase              -> yes[[4000,5000]]
   f5 no 52 -50: erase               -> no[]
   f6 no 30 +40                      -> no[[3000,4000]]
   f7 yes 66 +30                     -> yes[[4000,5000],[6600,3000]]
   f8 yes 66 -30: erase              -> yes[[4000,5000]]                  */
TEST(Replay, BookStateMatchesHandDerivedGolden) {
  EXPECT_EQ(book_replay("tests/fixtures/replay_sample.jsonl"),
            read_file("tests/fixtures/replay_sample.book_state"));
}

/* SYNTHETIC fixture (hand-written, not captured from Kalshi). Derivation:
   f0 "subscribed"                     -> ignored, no line (f still advances)
   f1 doc snapshot, no side omitted    -> yes[[3950,250],[4000,1000]] no[] seq 1
   f2 doc delta no 0.55 +7.25          -> no[[5500,725]] seq 2
   f3 yes 0.40: 1000-2000 < 0          -> rej level_underflow, book + seq unchanged
   f4 KXOTHER never snapshotted        -> rej delta_before_snapshot, no book created
   f5 truncated                        -> rej malformed_json, t ""
   f6 legacy delta yes 40 -10: erase   -> yes[[3950,250]]; frame has no seq -> seq stays 2
   f7 side "banana"                    -> rej bad_side, book unchanged
   f8 duplicate key "price_dollars"    -> rej malformed_json, t "" (dup keys void the ticker)
   f9 doc snapshot, yes omitted        -> yes[] no[[6000,100]] seq 9               */
TEST(Replay, MixedFixtureMatchesHandDerivedGolden) {
  EXPECT_EQ(book_replay("tests/fixtures/ingest_mixed.jsonl"),
            read_file("tests/fixtures/ingest_mixed.book_state"));
}
```

Run: `cmake --build build -j && ./build/te_tests --gtest_filter='Replay.*'`
Expected: both new tests PASS if Tasks 4–7 match the spec. If one fails, compare line by line against the derivation comment. The derivation is the spec; fix the code, not the golden. If you believe the derivation itself is wrong, stop and report rather than regenerating the golden from the code's output.

- [ ] **Step 3: Write the tool**

`tools/book_replay.cpp`:

```cpp
// book_replay: feed a .jsonl file of raw WebSocket frames through the real
// ingest path (MarketDataGateway::handle_raw) and print one book-state line
// per frame on stdout (spec §6). Nonzero reject counts go to stderr. No
// network, no credentials. The spec-4 Go port ships an equivalent, and the
// cross-implementation diff runs both over the same files.
//
//   usage: book_replay <frames.jsonl>
#include <fstream>
#include <iostream>
#include <string>

#include "market_data/book_state.hpp"
#include "market_data/gateway.hpp"

using namespace te;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: book_replay <frames.jsonl>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open: " << argv[1] << "\n";
    return 1;
  }

  BookStateWriter writer(std::cout);
  MarketDataGateway gw;
  gw.set_book_state_writer(&writer);

  // Every line is a frame, empty ones included (an empty line is
  // malformed_json), so `f` always equals the 0-based line number.
  std::string line;
  while (std::getline(in, line)) gw.handle_raw(line);

  const auto& counts = gw.reject_counts();
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] != 0) std::cerr << to_string(static_cast<RejectReason>(i)) << ' ' << counts[i] << '\n';
  }
  return 0;
}
```

Append to `CMakeLists.txt`:

```cmake

# Book-state replay: one serialized book-state line per input frame (spec §6).
# The cross-implementation diff (spec 4) runs this against the Go equivalent.
add_executable(book_replay tools/book_replay.cpp)
target_link_libraries(book_replay PRIVATE te_lib)
target_include_directories(book_replay PRIVATE src)
```

- [ ] **Step 4: Verify the binary matches the goldens**

```bash
cmake -S . -B build && cmake --build build -j
./build/book_replay tests/fixtures/replay_sample.jsonl | diff - tests/fixtures/replay_sample.book_state && echo SAMPLE_OK
./build/book_replay tests/fixtures/ingest_mixed.jsonl | diff - tests/fixtures/ingest_mixed.book_state && echo MIXED_OK
./build/book_replay tests/fixtures/ingest_mixed.jsonl 2>&1 >/dev/null
./build/te_tests
```

Expected: `SAMPLE_OK` and `MIXED_OK`; the third command's stderr is exactly these four lines, in `RejectReason` enum order:

```
malformed_json 2
bad_side 1
delta_before_snapshot 1
level_underflow 1
```

All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/book_replay.cpp CMakeLists.txt tests/test_replay.cpp \
  tests/fixtures/replay_sample.book_state tests/fixtures/ingest_mixed.jsonl tests/fixtures/ingest_mixed.book_state
git commit -F - <<'EOF'
feat(engine): book_replay tool with hand-derived golden outputs

book_replay prints one book-state line per frame: the binary the Go
port's cross-implementation diff will run against. Goldens for the
existing fixture and a new synthetic mixed fixture (ignored, rejected,
documented-shape and seq-bearing frames) are derived by hand, with the
derivations committed next to the tests (spec §9).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

### Task 10: Documentation from measured results

Every number in this task comes from a command run in this task. Nothing is copied from the spec or this plan.

**Files:**
- Modify: `trading_engine/README.md`, `README.md` (repo root)

**Interfaces:** none.

- [ ] **Step 1: Measure**

From `trading_engine/`:

```bash
cmake -S . -B build && cmake --build build -j 2>&1 | grep -ciE "warning" ; ./build/te_tests 2>&1 | tail -3
./build/te_tests --gtest_list_tests | grep -c '^  '
./build/te_tests --gtest_list_tests | grep -c '^[^ ]'
```

Record: warning count (must be 0), the `[  PASSED  ] N tests.` line, N tests, and S suites. The two `--gtest_list_tests` counts must agree with the run's summary line. Below, `N` and `S` mean these measured values.

- [ ] **Step 2: Update `trading_engine/README.md`**

1. **`### Parsing`** — replace the section body with:

```markdown
`decode_frame` (`src/market_data/kalshi_messages.cpp`) parses each frame with
simdjson's **DOM** API, which validates the whole document before anything is
read, and uses its error-code interface only, so nothing on this path throws.
It accepts both wire shapes, chosen per frame: the legacy integer-cent shape
(`yes`/`no`, `price`/`delta`) and the documented string shape
(`yes_dollars_fp`/`no_dollars_fp`, `price_dollars`/`delta_fp`). Kalshi's docs
show the second; which one the `/trade-api/ws/v2` endpoint actually sends is
unconfirmed until a real session is captured.

Fixed-point strings are parsed digit by digit into integers
(`src/core/fixed_point.cpp`), never through `double`. Units are integer
fixed-point throughout: price in 1/10,000 dollar, quantity in 1/100 contract,
money in micro-dollars.

A frame is rejected, with the book untouched, if it is not entirely well-formed
JSON, has a duplicate key, is missing a required field, has a ticker outside
`[A-Za-z0-9._-]{1,64}`, a side other than `yes`/`no`, a price outside
`0.0001–0.9999`, a malformed number, a zero delta, a non-positive snapshot
quantity, or the same price twice on one side of a snapshot. An omitted
snapshot side means that side is empty, as Kalshi documents.
```

2. **`### Applying incremental deltas`** — replace the section body with:

```markdown
`OrderBook::apply_delta` (`src/market_data/order_book.cpp`) rejects a delta for
a book that has never been snapshotted, and a delta that would take a level
below zero, including a negative delta on a level that does not exist. Both
checks run before any mutation. A delta that lands a level on exactly zero
erases it. The gateway never creates a book from a delta. The `seq` of the
last applied frame that carried one is recorded per book.
```

3. **Failure-closed §1** — replace the heading and everything through the end of the "What this does not cover" paragraph with:

```markdown
**1. A bad frame changes nothing, is counted, and does not reconnect.**
`MarketDataGateway::handle_raw` returns an `IngestResult` (applied, rejected
with a reason, or ignored) and cannot throw on frame content. A rejected frame
leaves every book unchanged, does not reach the strategy, and increments a
per-reason counter (`reject_counts()`). In the live loop a reject is logged at
most once per reason per second and the loop continues, so the reconnect
handler only ever sees transport failures. `tests/test_ingest_audit.cpp` pins
the eight inputs an earlier audit found mishandled, and
`tests/test_ingest_no_throw.cpp` feeds every prefix and single-byte corruption
of every real frame through `handle_raw` without an exception.

**What this does not cover yet.** Sequence gaps are recorded but not detected:
a jump in `seq` is applied. Gap detection and recovery are the next piece of
work.
```

4. **New section**, inserted immediately before `## Deterministic replay harness`:

````markdown
## Book-state output

`BookStateWriter` (`src/market_data/book_state.cpp`) emits one line per input
frame, integers only, written by hand rather than through a JSON library:

```
{"f":2,"t":"KXNBA-25-ABC","s":"ok","seq":2,"yes":[[3950,250],[4000,1000]],"no":[[5500,725]]}
```

`f` is the input line number, `s` is `ok` or `rej`, `seq` is the last applied
sequence number or `-1`, and levels are `[price, qty]` in fixed-point units,
ascending. A rejected frame still emits a line showing the unchanged book.
`build/book_replay <frames.jsonl>` prints these lines for any file of frames.
The format is specified exactly in
`docs/superpowers/specs/2026-09-19-engine-ingest-hardening-design.md` §6 so a
second implementation can reproduce it byte for byte.

Both fixtures in `tests/fixtures/` are hand-written, not captured from Kalshi.
Their expected outputs (`*.book_state`) were derived by hand; the derivations
are in `tests/test_replay.cpp`.
````

5. **`## Tests`** — replace the count line and the pasted run output with the Step 1 measurements (`N tests across S suites, all passing.` and the actual last lines of `./build/te_tests`). If the section has a per-suite table, regenerate it from `./build/te_tests --gtest_list_tests`, one row per suite, with the test count per suite as listed. Delete any row whose suite no longer exists, such as `Parse`.

6. **`### Running the replay harness offline`** — append:

````markdown
To print the book state after every frame:

```bash
./build/book_replay tests/fixtures/ingest_mixed.jsonl
```
````

7. **`## Known limitations`** — delete these three bullets:
   - `No sequence/gap checking and no frame validation …`
   - `A malformed frame is handled as a connection failure and triggers a reconnect.`
   - `No serialized book-state output; replay compares strategy telemetry only.`

   and add, as the second bullet:
   - `**Sequence gaps are not detected**: a jump in \`seq\` is recorded and applied.`

   Keep the live-path, recorder/journal, risk-limit, arb, market-making, P&L and fee bullets exactly as they are.

8. Search the file for `33` and `realized_pnl_cents` and update any remaining occurrence to the measured count or `realized_pnl_micros`.

- [ ] **Step 3: Update the root `README.md`**

Replace every C++ test count with the measured `N`. Leave the Python count (`193`) alone; it belongs to a different suite.

- Line 16 table cell: `33 tests passing` → `N tests passing`
- `- **33 tests across 17 suites, all passing.**` → `- **N tests across S suites, all passing.**`
- `# 33 GoogleTest tests` → `# N GoogleTest tests`
- `./build/te_tests        # 33 tests` → `./build/te_tests        # N tests`
- `cd trading_engine && ./build/te_tests    # 33 C++ tests` → `… # N C++ tests`
- `- [x] Deterministic replay harness (33 tests passing)` → `- [x] Deterministic replay harness (N tests passing)`
- Footer `(paper-only, 33 + 193 tests passing)` → `(paper-only, N + 193 tests passing)`

In the trading-engine known-limits sentence, replace `there is no sequence/gap checking or validation of incoming frames;` with `sequence gaps are recorded but not yet detected;`.

In the Phase 5 roadmap, replace `- [ ] Sequence/gap checking and frame validation` with two lines:

```markdown
- [x] Frame validation, fixed-point units and byte-exact book-state output
- [ ] Sequence-gap detection and recovery
```

- [ ] **Step 4: Check every claim you wrote**

For each sentence changed in Steps 2–3, point to the file and line or the test that makes it true. In particular confirm: `grep -rn "throw" src/market_data/kalshi_messages.cpp src/market_data/gateway.cpp src/market_data/book_state.cpp src/core/fixed_point.cpp` finds no `throw`; the fixture files are the hand-written ones from Task 9; neither README says "live", "recorded", or "captured" about anything the engine has not actually done.

- [ ] **Step 5: Commit**

```bash
cd ~/benchwarmer-nba
git add README.md trading_engine/README.md
git commit -F - <<'EOF'
docs: describe validated ingest, fixed-point units and book-state output

Test counts are the measured values from this branch. Sequence-gap
detection, the recorder and the live path are still stated as not done.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

---

## Carry-forward for spec 4 (Go port)

Parity hazards found while planning. None affect this plan; each must be settled in the spec-4 design so the Go decoder agrees with this one byte for byte.

- **Invalid UTF-8.** simdjson rejects it (→ `MalformedJson`). Go's `encoding/json` silently replaces invalid bytes with U+FFFD. Go must validate with `utf8.Valid` before decoding.
- **Duplicate keys.** Go must detect them explicitly (a `json.Decoder` token walk over the top-level object and `msg`); `json.Unmarshal` keeps the last value silently.
- **Integer vs float.** The legacy shape requires JSON integers. Go must decode with `UseNumber()` and reject any `json.Number` that has a fraction or exponent, including `42.0`, as simdjson's `get_int64` does.
- **Large integers.** Values outside int64 are errors in simdjson. Go must match, and must apply the same ±999,999,999,999 legacy cap.
- **Whole-document validation.** Trailing content after the root object (`{…} x`) is `MalformedJson`. `json.Decoder` stops after the first value, so Go must check that nothing but whitespace follows it.
- **Ticker rule timing.** `t` is set by the frame's properties (§6), not by parse progress. Port the property check, not the C++ control flow.
- **Precedence** (plan clarification 6) decides only reject *reasons*, which are not compared; `ok`/`rej` must not depend on it.

---

## Self-review record

Checked against the spec after writing:

- §1 goals → Tasks 3 (units), 4 (dual decoder, validation), 5 (state validation), 6 (format), 7 (error separation), 9 (tool). §3 units/config/strategy/P&L → Tasks 2–3. §4 rule table → Tasks 4–5, every row has a test in `DecodeRejects` or `OrderBookDelta`. §5 DOM, duplicate keys, fixed-point, seq → Tasks 1, 4. §6 grammar, `t` rule, writer, tool → Tasks 4, 6, 9. §7 → Task 7. §8 files → file map. §9 tests → Tasks 1–9; baseline equivalence in Task 3; hand-derived goldens in Task 9. §10 exclusions respected (no seq enforcement, no journal, no Go). §11 criteria 1–7 → Tasks 8–10.
- Additions beyond the spec, flagged: plan clarifications 1–7 (choices the spec left open), and the synthetic `ingest_mixed` fixture (the only end-to-end coverage of ignored/rejected/`t:""`/seq lines).
- Types and names are consistent across tasks: `Price`/`Qty`/`Micros`, `RejectReason`, `FrameKind`, `DecodedFrame`, `DecodeResult`, `IngestStatus`, `IngestResult`, `BookStateWriter`, `format_book_state_line`, `realized_pnl_micros`, `max_order_qty`, `max_qty_per_market`, `max_daily_loss`, `fee_per_contract`, `base_edge`.
