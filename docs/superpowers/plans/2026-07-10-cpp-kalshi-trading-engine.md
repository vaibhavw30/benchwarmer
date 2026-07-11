# C++ Low-Latency Kalshi NBA Trading Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-trading C++ engine that maintains live Kalshi NBA game-winner order books and market-makes / takes edge around the fair values produced by this repo's existing `backend_ml` win-probability ensemble.

**Architecture:** Greenfield C++20 module `trading_engine/` beside the untouched `backend_ml/`. The Python model publishes per-ticker fair values to `trading_engine/fair_values.json` (via new `backend_ml/publish_fair_values.py` calling the existing `predict_games()`); the C++ engine consumes that file, ingests Kalshi `orderbook_delta` over WebSocket, and runs a single-threaded deterministic decision loop (arb → market-make → edge-take) gated by a RiskManager into a `PaperVenue`. No live order routing in this plan.

**Tech Stack:** C++20, CMake, GoogleTest, OpenSSL (RSA-PSS auth), Boost.Beast (WebSocket + REST), simdjson (hot-path decode), nlohmann/json (config/fair values), Python 3 + pytest (fair-value publisher).

## Global Constraints

- **Language floor:** C++20; CMake ≥ 3.20. Python ≥ 3.10 for the publisher.
- **Paper only:** `LiveKalshiVenue` is NOT implemented in this plan. No real orders.
- **Pre-game only:** fair value is quasi-static; no live-score / in-play logic.
- **Prices are integer cents** in `[1, 99]`; contracts settle at `0` or `100`. Never use floats for on-book prices.
- **Model is read-only:** do not modify features, training, or ensemble math in `backend_ml/`. The only Python addition is `backend_ml/publish_fair_values.py`.
- **Secrets never committed:** RSA private key + API key id come from env / untracked files. `trading_engine/fair_values.json`, `trading_engine/config/*secret*`, and `**/.env*` are gitignored.
- **Kalshi endpoints:** REST base `https://api.elections.kalshi.com/trade-api/v2`; WS `wss://api.elections.kalshi.com/trade-api/ws/v2`. Auth = RSA-PSS(SHA-256, MGF1-SHA-256, salt=32) over `timestamp_ms + METHOD + path` (REST) or `timestamp_ms + "GET" + "/trade-api/ws/v2"` (WS); base64 signature in header `KALSHI-ACCESS-SIGNATURE`, key id in `KALSHI-ACCESS-KEY`, timestamp in `KALSHI-ACCESS-TIMESTAMP`.
- **Fees are real:** every "edge"/"arb" comparison must subtract the configured Kalshi fee; an unfee'd edge is a bug.
- **Fail closed:** stale fair value, crossed/invalid book, or unmapped ticker ⇒ do not quote that market.
- **Branch:** all work on `feat/cpp-kalshi-trading-engine` (already created).

---

## File Structure

```
trading_engine/
├── CMakeLists.txt                       # build + GoogleTest + deps
├── .gitignore                           # fair_values.json, *secret*, .env*
├── config/
│   ├── engine.json                      # limits, fees, thresholds, refresh intervals
│   └── watchlist.json                   # ticker -> {home_team_id, away_team_id, game_date}
├── src/
│   ├── core/types.hpp                   # Cents, Side, Ticker, value types
│   ├── core/config.hpp / .cpp           # Config loader (nlohmann)
│   ├── telemetry/telemetry.hpp / .cpp   # JSONL structured logger
│   ├── market_data/order_book.hpp/.cpp  # snapshot/delta application, best bid/ask
│   ├── market_data/kalshi_messages.hpp/.cpp  # parse WS JSON -> structs (simdjson)
│   ├── market_data/kalshi_auth.hpp/.cpp # RSA-PSS signer (OpenSSL)
│   ├── market_data/gateway.hpp/.cpp     # Boost.Beast WS client -> OrderBook + callback
│   ├── fair_value/fair_value.hpp/.cpp   # FairValueProvider (reads fair_values.json)
│   ├── market_map/market_map.hpp/.cpp   # ticker <-> game resolution from watchlist
│   ├── strategy/pricing.hpp/.cpp        # prob->cents, edge threshold, fee math
│   ├── strategy/arb.hpp/.cpp            # arb detector (pure)
│   ├── strategy/market_maker.hpp/.cpp   # quote construction (pure)
│   ├── strategy/edge_taker.hpp/.cpp     # edge-take decision (pure)
│   ├── strategy/strategy_engine.hpp/.cpp# orchestrates arb+mm+edge per book update
│   ├── risk/risk_manager.hpp/.cpp       # order gate + kill switch
│   ├── execution/order_venue.hpp        # OrderVenue interface + Order/Fill types
│   ├── execution/paper_venue.hpp/.cpp   # simulated fills, positions, P&L
│   └── main.cpp                         # assemble + run; also replay entrypoint
├── tests/
│   ├── test_order_book.cpp
│   ├── test_kalshi_messages.cpp
│   ├── test_kalshi_auth.cpp
│   ├── test_pricing.cpp
│   ├── test_arb.cpp
│   ├── test_market_maker.cpp
│   ├── test_risk_manager.cpp
│   ├── test_paper_venue.cpp
│   ├── test_strategy_engine.cpp
│   └── test_replay.cpp
└── recordings/                          # captured orderbook_delta streams (gitignored)

backend_ml/
└── publish_fair_values.py               # NEW: predict_games() -> fair_values.json
tests/ (python)
└── test_publish_fair_values.py
```

---

## PHASE M0 — Skeleton

### Task 1: Project scaffold + core value types

**Files:**
- Create: `trading_engine/CMakeLists.txt`, `trading_engine/.gitignore`
- Create: `trading_engine/src/core/types.hpp`
- Test: `trading_engine/tests/test_order_book.cpp` (placeholder compile-only test first)

**Interfaces:**
- Produces: `Cents` (int, 1..99 on book), `Side` (`Yes`/`No`), `Action` (`Buy`/`Sell`), `Ticker` (std::string alias), `PriceLevel {Cents price; int qty;}`.

- [ ] **Step 1: Write `types.hpp`**

```cpp
// trading_engine/src/core/types.hpp
#pragma once
#include <string>
#include <cstdint>
namespace te {
using Ticker = std::string;
using Cents  = int;            // on-book price, 1..99; settle 0/100
enum class Side   { Yes, No };
enum class Action { Buy, Sell };
struct PriceLevel { Cents price; int qty; };
constexpr Cents kSettleYes = 100;
constexpr Cents kSettleNo  = 0;
}
```

- [ ] **Step 2: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.20)
project(trading_engine CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.15.2)
FetchContent_MakeAvailable(googletest)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)
FetchContent_Declare(simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git GIT_TAG v3.10.1)
FetchContent_MakeAvailable(simdjson)
find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED)   # header-only Beast; system Boost ok

enable_testing()
add_library(te_core INTERFACE)
target_include_directories(te_core INTERFACE src)

file(GLOB TEST_SRCS tests/*.cpp)
add_executable(te_tests ${TEST_SRCS})
target_link_libraries(te_tests PRIVATE te_core GTest::gtest_main
  nlohmann_json::nlohmann_json simdjson OpenSSL::Crypto OpenSSL::SSL)
target_include_directories(te_tests PRIVATE src)
include(GoogleTest)
gtest_discover_tests(te_tests)
```

- [ ] **Step 3: Write a trivial passing test**

```cpp
// trading_engine/tests/test_order_book.cpp
#include <gtest/gtest.h>
#include "core/types.hpp"
TEST(Scaffold, TypesCompile) {
  te::PriceLevel lvl{55, 10};
  EXPECT_EQ(lvl.price, 55);
  EXPECT_EQ(te::kSettleYes, 100);
}
```

- [ ] **Step 4: Write `.gitignore`**

```
# trading_engine/.gitignore
fair_values.json
recordings/
config/*secret*
**/.env*
build/
```

- [ ] **Step 5: Build + run**

Run: `cd trading_engine && cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: `TypesCompile` PASSES.

- [ ] **Step 6: Commit**

```bash
git add trading_engine/CMakeLists.txt trading_engine/.gitignore trading_engine/src/core/types.hpp trading_engine/tests/test_order_book.cpp
git commit -m "feat(engine): project scaffold, core value types, gtest wired"
```

---

### Task 2: Config loader

**Files:**
- Create: `trading_engine/src/core/config.hpp`, `trading_engine/src/core/config.cpp`
- Create: `trading_engine/config/engine.json`
- Test: `trading_engine/tests/test_config.cpp`

**Interfaces:**
- Produces: `Config::load(path) -> Config`; fields `int max_contracts_per_market; int max_aggregate_exposure_cents; int max_order_size; int max_daily_loss_cents; int fee_cents_per_contract; int base_edge_cents; double confidence_k; int fair_value_refresh_secs; int fair_value_max_age_secs; int orders_per_sec_budget;`.

- [ ] **Step 1: Write failing test**

```cpp
// trading_engine/tests/test_config.cpp
#include <gtest/gtest.h>
#include "core/config.hpp"
TEST(Config, LoadsEngineJson) {
  auto c = te::Config::load("config/engine.json");
  EXPECT_EQ(c.max_contracts_per_market, 100);
  EXPECT_EQ(c.fee_cents_per_contract, 1);
  EXPECT_GT(c.confidence_k, 0.0);
}
```

- [ ] **Step 2: Run — expect FAIL** (`config.hpp` not found). `ctest --test-dir build -R Config`.

- [ ] **Step 3: Write `engine.json`**

```json
{
  "max_contracts_per_market": 100,
  "max_aggregate_exposure_cents": 500000,
  "max_order_size": 25,
  "max_daily_loss_cents": 20000,
  "fee_cents_per_contract": 1,
  "base_edge_cents": 2,
  "confidence_k": 8.0,
  "fair_value_refresh_secs": 60,
  "fair_value_max_age_secs": 1800,
  "orders_per_sec_budget": 5
}
```

- [ ] **Step 4: Write `config.hpp`**

```cpp
#pragma once
#include <string>
namespace te {
struct Config {
  int max_contracts_per_market{};
  int max_aggregate_exposure_cents{};
  int max_order_size{};
  int max_daily_loss_cents{};
  int fee_cents_per_contract{};
  int base_edge_cents{};
  double confidence_k{};
  int fair_value_refresh_secs{};
  int fair_value_max_age_secs{};
  int orders_per_sec_budget{};
  static Config load(const std::string& path);
};
}
```

- [ ] **Step 5: Write `config.cpp`**

```cpp
#include "core/config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
namespace te {
Config Config::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("config not found: " + path);
  nlohmann::json j; f >> j;
  Config c;
  c.max_contracts_per_market   = j.at("max_contracts_per_market");
  c.max_aggregate_exposure_cents = j.at("max_aggregate_exposure_cents");
  c.max_order_size             = j.at("max_order_size");
  c.max_daily_loss_cents       = j.at("max_daily_loss_cents");
  c.fee_cents_per_contract     = j.at("fee_cents_per_contract");
  c.base_edge_cents            = j.at("base_edge_cents");
  c.confidence_k               = j.at("confidence_k");
  c.fair_value_refresh_secs    = j.at("fair_value_refresh_secs");
  c.fair_value_max_age_secs    = j.at("fair_value_max_age_secs");
  c.orders_per_sec_budget      = j.at("orders_per_sec_budget");
  return c;
}
}
```

- [ ] **Step 6:** Add `config.cpp` to a `te_lib` static library in CMake and link `te_tests` to it. Update `CMakeLists.txt`:

```cmake
file(GLOB LIB_SRCS src/*/*.cpp)
add_library(te_lib STATIC ${LIB_SRCS})
target_include_directories(te_lib PUBLIC src)
target_link_libraries(te_lib PUBLIC nlohmann_json::nlohmann_json simdjson OpenSSL::Crypto OpenSSL::SSL)
target_link_libraries(te_tests PRIVATE te_lib)
```
Set the test working dir so `config/engine.json` resolves: add `set_tests_properties` via `gtest_discover_tests(te_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})`.

- [ ] **Step 7:** Run `ctest --test-dir build -R Config --output-on-failure` → PASS.

- [ ] **Step 8: Commit**

```bash
git add trading_engine/src/core/config.* trading_engine/config/engine.json trading_engine/tests/test_config.cpp trading_engine/CMakeLists.txt
git commit -m "feat(engine): JSON config loader with risk/fee/threshold params"
```

---

### Task 3: Telemetry (JSONL event log)

**Files:**
- Create: `trading_engine/src/telemetry/telemetry.hpp`, `.cpp`
- Test: `trading_engine/tests/test_telemetry.cpp`

**Interfaces:**
- Produces: `Telemetry(std::ostream&)`; `void event(std::string type, nlohmann::json fields)` writes one JSON object per line with a monotonically increasing `seq`.

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include <sstream>
#include "telemetry/telemetry.hpp"
TEST(Telemetry, WritesOneJsonLinePerEvent) {
  std::ostringstream os;
  te::Telemetry t(os);
  t.event("quote", {{"ticker","KXNBA-XYZ"},{"bid",54},{"ask",56}});
  t.event("fill",  {{"ticker","KXNBA-XYZ"},{"price",55}});
  auto s = os.str();
  EXPECT_EQ(std::count(s.begin(), s.end(), '\n'), 2);
  EXPECT_NE(s.find("\"type\":\"quote\""), std::string::npos);
  EXPECT_NE(s.find("\"seq\":0"), std::string::npos);
}
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Implement**

```cpp
// telemetry.hpp
#pragma once
#include <ostream>
#include <string>
#include <nlohmann/json.hpp>
namespace te {
class Telemetry {
 public:
  explicit Telemetry(std::ostream& out) : out_(out) {}
  void event(const std::string& type, nlohmann::json fields) {
    fields["type"] = type;
    fields["seq"]  = seq_++;
    out_ << fields.dump() << '\n';
    out_.flush();
  }
 private:
  std::ostream& out_;
  long seq_ = 0;
};
}
```
(Header-only; no `.cpp` needed — delete the `.cpp` glob entry if empty.)

- [ ] **Step 4:** Run → PASS.

- [ ] **Step 5: Commit**

```bash
git add trading_engine/src/telemetry/telemetry.hpp trading_engine/tests/test_telemetry.cpp
git commit -m "feat(engine): JSONL telemetry event logger"
```

---

## PHASE M1 — Market Data

### Task 4: OrderBook (snapshot + delta application)

**Files:**
- Create: `trading_engine/src/market_data/order_book.hpp`, `.cpp`
- Test: `trading_engine/tests/test_order_book.cpp` (extend)

**Interfaces:**
- Consumes: `core/types.hpp`.
- Produces:
  `struct BookSnapshot { std::vector<PriceLevel> yes; std::vector<PriceLevel> no; };`
  `struct BookDelta { Side side; Cents price; int delta_qty; };`
  `class OrderBook` with `void apply_snapshot(const BookSnapshot&)`, `void apply_delta(const BookDelta&)`, `std::optional<Cents> best_yes_bid() const`, `best_yes_ask() const`, `best_no_bid() const`, `best_no_ask() const`, `int qty_at(Side, Cents) const`, `bool crossed() const`.

> Kalshi convention used here: the book is quoted in YES bids and NO bids. A YES **ask** at price `p` is equivalent to a NO **bid** at `100 - p`. `best_yes_ask()` is derived as `100 - best_no_bid()`.

- [ ] **Step 1: Failing tests**

```cpp
#include <gtest/gtest.h>
#include "market_data/order_book.hpp"
using namespace te;
TEST(OrderBook, SnapshotThenBestPrices) {
  OrderBook b;
  b.apply_snapshot({/*yes*/{{54,10},{53,5}}, /*no*/{{44,8},{43,3}}});
  EXPECT_EQ(b.best_yes_bid().value(), 54);
  EXPECT_EQ(b.best_no_bid().value(), 44);
  EXPECT_EQ(b.best_yes_ask().value(), 56); // 100 - best_no_bid(44)
  EXPECT_EQ(b.best_no_ask().value(), 46);  // 100 - best_yes_bid(54)
}
TEST(OrderBook, DeltaAddsAndRemovesLevels) {
  OrderBook b;
  b.apply_snapshot({{{54,10}},{{44,8}}});
  b.apply_delta({Side::Yes, 54, -10}); // remove all qty at 54
  b.apply_delta({Side::Yes, 55, 7});   // new best yes bid
  EXPECT_EQ(b.best_yes_bid().value(), 55);
  EXPECT_EQ(b.qty_at(Side::Yes, 54), 0);
}
TEST(OrderBook, DetectsCrossed) {
  OrderBook b;
  b.apply_snapshot({{{60,5}},{{45,5}}}); // yes_bid 60, yes_ask 55 -> crossed
  EXPECT_TRUE(b.crossed());
}
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Implement `order_book.hpp`**

```cpp
#pragma once
#include <map>
#include <optional>
#include <vector>
#include "core/types.hpp"
namespace te {
struct BookSnapshot { std::vector<PriceLevel> yes, no; };
struct BookDelta { Side side; Cents price; int delta_qty; };
class OrderBook {
 public:
  void apply_snapshot(const BookSnapshot& s);
  void apply_delta(const BookDelta& d);
  std::optional<Cents> best_yes_bid() const;
  std::optional<Cents> best_no_bid() const;
  std::optional<Cents> best_yes_ask() const; // 100 - best_no_bid
  std::optional<Cents> best_no_ask() const;  // 100 - best_yes_bid
  int qty_at(Side s, Cents p) const;
  bool crossed() const;
 private:
  std::map<Cents,int> yes_;  // price -> qty (YES bids)
  std::map<Cents,int> no_;   // price -> qty (NO bids)
  static std::optional<Cents> best(const std::map<Cents,int>& m);
};
}
```

- [ ] **Step 4: Implement `order_book.cpp`**

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
  int q = m[d.price] + d.delta_qty;
  if (q <= 0) m.erase(d.price); else m[d.price] = q;
}
std::optional<Cents> OrderBook::best(const std::map<Cents,int>& m) {
  if (m.empty()) return std::nullopt;
  return m.rbegin()->first; // highest bid
}
std::optional<Cents> OrderBook::best_yes_bid() const { return best(yes_); }
std::optional<Cents> OrderBook::best_no_bid()  const { return best(no_); }
std::optional<Cents> OrderBook::best_yes_ask() const {
  auto nb = best_no_bid(); if (!nb) return std::nullopt; return 100 - *nb;
}
std::optional<Cents> OrderBook::best_no_ask()  const {
  auto yb = best_yes_bid(); if (!yb) return std::nullopt; return 100 - *yb;
}
int OrderBook::qty_at(Side s, Cents p) const {
  auto& m = (s == Side::Yes) ? yes_ : no_;
  auto it = m.find(p); return it == m.end() ? 0 : it->second;
}
bool OrderBook::crossed() const {
  auto yb = best_yes_bid(); auto ya = best_yes_ask();
  return yb && ya && *yb >= *ya;
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): order book with snapshot/delta application and crossed detection`.

---

### Task 5: Kalshi message parser (simdjson)

**Files:**
- Create: `trading_engine/src/market_data/kalshi_messages.hpp`, `.cpp`
- Test: `trading_engine/tests/test_kalshi_messages.cpp`

**Interfaces:**
- Consumes: `order_book.hpp` (`BookSnapshot`, `BookDelta`).
- Produces:
  `enum class MsgKind { Snapshot, Delta, Other };`
  `struct ParsedMsg { MsgKind kind; Ticker ticker; BookSnapshot snapshot; BookDelta delta; };`
  `ParsedMsg parse_ws_message(std::string_view json);`

> Kalshi `orderbook_snapshot` payload: `{"type":"orderbook_snapshot","msg":{"market_ticker":"...","yes":[[price,qty],...],"no":[[price,qty],...]}}`. `orderbook_delta`: `{"type":"orderbook_delta","msg":{"market_ticker":"...","price":P,"delta":D,"side":"yes"|"no"}}`. Confirm exact field names against a live capture in Task 8-integration; adjust the parser if Kalshi differs.

- [ ] **Step 1: Failing tests**

```cpp
#include <gtest/gtest.h>
#include "market_data/kalshi_messages.hpp"
using namespace te;
TEST(Parse, Snapshot) {
  auto m = parse_ws_message(R"({"type":"orderbook_snapshot","msg":{
    "market_ticker":"KXNBA-25-ABC","yes":[[54,10],[53,5]],"no":[[44,8]]}})");
  EXPECT_EQ(m.kind, MsgKind::Snapshot);
  EXPECT_EQ(m.ticker, "KXNBA-25-ABC");
  ASSERT_EQ(m.snapshot.yes.size(), 2u);
  EXPECT_EQ(m.snapshot.yes[0].price, 54);
  EXPECT_EQ(m.snapshot.no[0].qty, 8);
}
TEST(Parse, Delta) {
  auto m = parse_ws_message(R"({"type":"orderbook_delta","msg":{
    "market_ticker":"KXNBA-25-ABC","price":55,"delta":-3,"side":"yes"}})");
  EXPECT_EQ(m.kind, MsgKind::Delta);
  EXPECT_EQ(m.delta.side, Side::Yes);
  EXPECT_EQ(m.delta.price, 55);
  EXPECT_EQ(m.delta.delta_qty, -3);
}
TEST(Parse, Other) {
  auto m = parse_ws_message(R"({"type":"subscribed","msg":{}})");
  EXPECT_EQ(m.kind, MsgKind::Other);
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement** `kalshi_messages.hpp`:

```cpp
#pragma once
#include <string_view>
#include "market_data/order_book.hpp"
namespace te {
enum class MsgKind { Snapshot, Delta, Other };
struct ParsedMsg { MsgKind kind{MsgKind::Other}; Ticker ticker; BookSnapshot snapshot; BookDelta delta{}; };
ParsedMsg parse_ws_message(std::string_view json);
}
```

- [ ] **Step 4: Implement `kalshi_messages.cpp`**

```cpp
#include "market_data/kalshi_messages.hpp"
#include <simdjson.h>
namespace te {
ParsedMsg parse_ws_message(std::string_view json) {
  static thread_local simdjson::ondemand::parser parser;
  simdjson::padded_string buf(json);
  ParsedMsg out;
  auto doc = parser.iterate(buf);
  std::string_view type;
  if (doc["type"].get(type)) return out;
  auto msg = doc["msg"];
  if (type == "orderbook_snapshot") {
    out.kind = MsgKind::Snapshot;
    out.ticker = std::string(std::string_view(msg["market_ticker"]));
    for (auto lvl : msg["yes"].get_array())
      { auto a = lvl.get_array().begin(); Cents p = int64_t(*a); ++a; int q = int64_t(*a);
        out.snapshot.yes.push_back({p,q}); }
    for (auto lvl : msg["no"].get_array())
      { auto a = lvl.get_array().begin(); Cents p = int64_t(*a); ++a; int q = int64_t(*a);
        out.snapshot.no.push_back({p,q}); }
    return out;
  }
  if (type == "orderbook_delta") {
    out.kind = MsgKind::Delta;
    out.ticker = std::string(std::string_view(msg["market_ticker"]));
    out.delta.price = int64_t(msg["price"]);
    out.delta.delta_qty = int64_t(msg["delta"]);
    std::string_view side = msg["side"];
    out.delta.side = (side == "yes") ? Side::Yes : Side::No;
    return out;
  }
  return out; // Other
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): simdjson parser for Kalshi orderbook snapshot/delta`.

---

### Task 6: Kalshi RSA-PSS auth signer

**Files:**
- Create: `trading_engine/src/market_data/kalshi_auth.hpp`, `.cpp`
- Test: `trading_engine/tests/test_kalshi_auth.cpp`

**Interfaces:**
- Produces:
  `class KalshiSigner { KalshiSigner(std::string key_id, std::string pem_private_key); std::string key_id() const; std::string sign(std::string_view message) const; /* base64 RSA-PSS */ };`
  Free helper: `std::string ws_sign_message(long ts_ms) => std::to_string(ts_ms) + "GET" + "/trade-api/ws/v2";`

- [ ] **Step 1: Failing test** (round-trips a locally-generated key; verifies signature with the public key)

```cpp
#include <gtest/gtest.h>
#include "market_data/kalshi_auth.hpp"
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
// A 2048-bit test private key generated with:
//   openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out test_key.pem
static const char* kTestKeyPem = R"(-----BEGIN PRIVATE KEY-----
...PASTE A REAL LOCALLY-GENERATED TEST KEY HERE (test-only, never a real Kalshi key)...
-----END PRIVATE KEY-----)";
TEST(Auth, ProducesNonEmptyBase64Signature) {
  te::KalshiSigner s("test-key-id", kTestKeyPem);
  auto sig = s.sign(te::ws_sign_message(1700000000000L));
  EXPECT_FALSE(sig.empty());
  // base64 chars only
  for (char c : sig) EXPECT_TRUE(isalnum(c) || c=='+' || c=='/' || c=='=');
}
TEST(Auth, WsSignMessageFormat) {
  EXPECT_EQ(te::ws_sign_message(123), "123GET/trade-api/ws/v2");
}
```
> The implementer generates the test key with the `openssl` command in the comment and pastes it into `kTestKeyPem`. It is a throwaway key for unit testing only — never a Kalshi credential, never committed as a real secret.

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement `kalshi_auth.hpp`**

```cpp
#pragma once
#include <string>
#include <string_view>
namespace te {
std::string ws_sign_message(long ts_ms);
class KalshiSigner {
 public:
  KalshiSigner(std::string key_id, std::string pem_private_key);
  ~KalshiSigner();
  std::string key_id() const { return key_id_; }
  std::string sign(std::string_view message) const; // base64(RSA-PSS SHA256)
 private:
  std::string key_id_;
  void* pkey_ = nullptr; // EVP_PKEY*
};
}
```

- [ ] **Step 4: Implement `kalshi_auth.cpp`**

```cpp
#include "market_data/kalshi_auth.hpp"
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <stdexcept>
#include <vector>
namespace te {
std::string ws_sign_message(long ts_ms) {
  return std::to_string(ts_ms) + "GET" + "/trade-api/ws/v2";
}
static std::string b64(const unsigned char* data, size_t len) {
  BIO* b = BIO_new(BIO_s_mem()); BIO* f = BIO_new(BIO_f_base64());
  BIO_set_flags(f, BIO_FLAGS_BASE64_NO_NL); b = BIO_push(f, b);
  BIO_write(b, data, (int)len); BIO_flush(b);
  BUF_MEM* bptr; BIO_get_mem_ptr(b, &bptr);
  std::string out(bptr->data, bptr->length); BIO_free_all(b); return out;
}
KalshiSigner::KalshiSigner(std::string key_id, std::string pem) : key_id_(std::move(key_id)) {
  BIO* bio = BIO_new_mem_buf(pem.data(), (int)pem.size());
  EVP_PKEY* k = PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr);
  BIO_free(bio);
  if (!k) throw std::runtime_error("failed to parse RSA private key");
  pkey_ = k;
}
KalshiSigner::~KalshiSigner() { if (pkey_) EVP_PKEY_free((EVP_PKEY*)pkey_); }
std::string KalshiSigner::sign(std::string_view msg) const {
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  EVP_PKEY_CTX* pctx = nullptr;
  if (EVP_DigestSignInit(ctx, &pctx, EVP_sha256(), nullptr, (EVP_PKEY*)pkey_) <= 0)
    throw std::runtime_error("DigestSignInit");
  EVP_PKEY_CTX_set_rsa_padding(pctx, RSA_PKCS1_PSS_PADDING);
  EVP_PKEY_CTX_set_rsa_pss_saltlen(pctx, RSA_PSS_SALTLEN_DIGEST); // == digest len (32)
  EVP_PKEY_CTX_set_rsa_mgf1_md(pctx, EVP_sha256());
  if (EVP_DigestSignUpdate(ctx, msg.data(), msg.size()) <= 0)
    throw std::runtime_error("DigestSignUpdate");
  size_t siglen = 0;
  EVP_DigestSignFinal(ctx, nullptr, &siglen);
  std::vector<unsigned char> sig(siglen);
  if (EVP_DigestSignFinal(ctx, sig.data(), &siglen) <= 0)
    throw std::runtime_error("DigestSignFinal");
  EVP_MD_CTX_free(ctx);
  return b64(sig.data(), siglen);
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): OpenSSL RSA-PSS signer for Kalshi auth`.

---

### Task 7: MarketDataGateway (Boost.Beast WS client)

**Files:**
- Create: `trading_engine/src/market_data/gateway.hpp`, `.cpp`
- Test: `trading_engine/tests/test_gateway.cpp` (unit-tests the message→book routing with an injected feed; the live socket is exercised in the M1 integration step, not a unit test)

**Interfaces:**
- Consumes: `KalshiSigner`, `parse_ws_message`, `OrderBook`.
- Produces:
  `class MarketDataGateway { using UpdateCb = std::function<void(const Ticker&, const OrderBook&)>;`
  `void on_update(UpdateCb);`
  `void handle_raw(std::string_view json); // testable seam: parse + apply + fire callback`
  `const OrderBook& book(const Ticker&) const;`
  `void run(const std::vector<Ticker>& watchlist); // opens WS, auth, subscribe (integration) }`

> Rationale for the `handle_raw` seam: it makes the parse→apply→callback path deterministically unit-testable with no network. `run()` wires a standard Boost.Beast TLS WebSocket read loop that calls `handle_raw` on each frame; that loop is well-trodden library boilerplate (resolve → TLS handshake → WS handshake with the three `KALSHI-ACCESS-*` headers from `KalshiSigner` → send subscribe frame → async read loop). Verify it live in the M1 integration step below rather than in a unit test.

- [ ] **Step 1: Failing test (routing seam only)**

```cpp
#include <gtest/gtest.h>
#include "market_data/gateway.hpp"
using namespace te;
TEST(Gateway, RoutesSnapshotThenDeltaToBookAndFiresCallback) {
  MarketDataGateway g;
  int calls = 0; Ticker last;
  g.on_update([&](const Ticker& t, const OrderBook&){ ++calls; last = t; });
  g.handle_raw(R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[54,10]],"no":[[44,8]]}})");
  g.handle_raw(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":55,"delta":7,"side":"yes"}})");
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(last, "T");
  EXPECT_EQ(g.book("T").best_yes_bid().value(), 55);
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement `gateway.hpp`**

```cpp
#pragma once
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "market_data/order_book.hpp"
namespace te {
class MarketDataGateway {
 public:
  using UpdateCb = std::function<void(const Ticker&, const OrderBook&)>;
  void on_update(UpdateCb cb) { cb_ = std::move(cb); }
  void handle_raw(std::string_view json);
  const OrderBook& book(const Ticker& t) const { return books_.at(t); }
  // void run(...) added in the integration step; declared out-of-band to keep unit build light.
 private:
  std::unordered_map<Ticker, OrderBook> books_;
  UpdateCb cb_;
};
}
```

- [ ] **Step 4: Implement `gateway.cpp`**

```cpp
#include "market_data/gateway.hpp"
#include "market_data/kalshi_messages.hpp"
namespace te {
void MarketDataGateway::handle_raw(std::string_view json) {
  ParsedMsg m = parse_ws_message(json);
  if (m.kind == MsgKind::Snapshot) {
    books_[m.ticker].apply_snapshot(m.snapshot);
  } else if (m.kind == MsgKind::Delta) {
    books_[m.ticker].apply_delta(m.delta);
  } else return;
  if (cb_) cb_(m.ticker, books_[m.ticker]);
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): market-data gateway routing (parse→book→callback seam)`.

- [ ] **Step 7 (integration, manual — end of M1):** Add `run(watchlist)` using Boost.Beast (TLS WS to `api.elections.kalshi.com/trade-api/ws/v2`, handshake headers from `KalshiSigner`, subscribe `{"cmd":"subscribe","params":{"channels":["orderbook_delta"],"market_tickers":[...]}}`, read loop → `handle_raw`). Run against one real NBA ticker and **eyeball the maintained book against Kalshi's website** to confirm correctness. Commit `feat(engine): live Kalshi WS connect + subscribe (M1 verified)`. This step has no unit test — it is verified by live observation, logged via Telemetry.

---

## PHASE M2 — Fair value + market map

### Task 8: Python fair-value publisher

**Files:**
- Create: `backend_ml/publish_fair_values.py`
- Test: `tests/test_publish_fair_values.py`

**Interfaces:**
- Consumes: existing `backend_ml/predict.py` `predict_games()` (returns list of dicts with `home_win_probability`, `confidence_score`, `home_team_id`, `away_team_id`, `date`, `game_id`).
- Produces: function `build_fair_values(predictions, watchlist) -> list[dict]` and a `main()` that writes `trading_engine/fair_values.json`. Each row: `{"ticker": str, "p_yes": float, "confidence": float, "asof": iso8601, "game_id": str}` where `p_yes` = the home team's win probability for the YES(home wins) contract.

> `build_fair_values` is pure and unit-tested with a fake predictions list + fake watchlist, so the test never calls the real model, nba_api, or Supabase.

- [ ] **Step 1: Failing test**

```python
# tests/test_publish_fair_values.py
from backend_ml.publish_fair_values import build_fair_values

def test_maps_prediction_to_ticker_by_teams_and_date():
    preds = [{"game_id":"0022500123","date":"2026-07-10",
              "home_team_id":1610612744,"away_team_id":1610612747,
              "home_win_probability":0.62,"confidence_score":0.62}]
    watchlist = [{"ticker":"KXNBA-26JUL10-GSWLAL",
                  "home_team_id":1610612744,"away_team_id":1610612747,
                  "game_date":"2026-07-10"}]
    rows = build_fair_values(preds, watchlist)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "KXNBA-26JUL10-GSWLAL"
    assert abs(rows[0]["p_yes"] - 0.62) < 1e-9
    assert rows[0]["confidence"] == 0.62

def test_skips_unmapped_game():
    preds = [{"game_id":"x","date":"2026-07-10","home_team_id":1,"away_team_id":2,
              "home_win_probability":0.5,"confidence_score":0.5}]
    assert build_fair_values(preds, watchlist=[]) == []
```

- [ ] **Step 2:** Run `pytest tests/test_publish_fair_values.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `backend_ml/publish_fair_values.py`**

```python
"""Publish per-Kalshi-ticker fair values from the existing ensemble model.

Reuses backend_ml/predict.py:predict_games() output. The only new Python file
in this project; the model math is untouched.
"""
import json, datetime, os
from pathlib import Path

def build_fair_values(predictions, watchlist):
    """Pure mapping: (model predictions, ticker watchlist) -> fair-value rows.

    Joins on (home_team_id, away_team_id, game_date). Unmapped games are
    skipped (fail-closed): a wrong ticker means trading the wrong game.
    """
    index = {(w["home_team_id"], w["away_team_id"], w["game_date"]): w["ticker"]
             for w in watchlist}
    rows = []
    for p in predictions:
        key = (p["home_team_id"], p["away_team_id"], p["date"])
        ticker = index.get(key)
        if ticker is None:
            continue
        rows.append({
            "ticker": ticker,
            "p_yes": float(p["home_win_probability"]),   # YES = home wins
            "confidence": float(p["confidence_score"]),
            "asof": datetime.datetime.utcnow().isoformat() + "Z",
            "game_id": p["game_id"],
        })
    return rows

def main():
    from backend_ml.predict import predict_games  # imported lazily; heavy deps
    wl_path = os.getenv("WATCHLIST_PATH", "trading_engine/config/watchlist.json")
    out_path = os.getenv("FAIR_VALUES_PATH", "trading_engine/fair_values.json")
    watchlist = json.loads(Path(wl_path).read_text())
    predictions = predict_games(day_offset=0)   # existing model entrypoint
    if not isinstance(predictions, list):
        raise SystemExit("predict_games did not return a prediction list")
    rows = build_fair_values(predictions, watchlist)
    Path(out_path).write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} fair values -> {out_path}")

if __name__ == "__main__":
    main()
```

> Note: `predict_games()` currently prints and upserts to Supabase and returns after building `preds`. Confirm it `return preds` (it appends rows to `preds` through `predict.py`); if it returns `False`/`None` on the no-games path, `main()` already guards with the `isinstance` check. If `predict_games` needs a small change to return the list, make that the minimal edit and note it in the commit — do not alter the model math.

- [ ] **Step 4:** Run `pytest tests/test_publish_fair_values.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add backend_ml/publish_fair_values.py tests/test_publish_fair_values.py
git commit -m "feat(ml): publish per-Kalshi-ticker fair values from predict_games()"
```

---

### Task 9: FairValueProvider (C++ reader)

**Files:**
- Create: `trading_engine/src/fair_value/fair_value.hpp`, `.cpp`
- Test: `trading_engine/tests/test_fair_value.cpp`

**Interfaces:**
- Produces:
  `struct FairValue { double p_yes; double confidence; long asof_epoch_ms; };`
  `class FairValueProvider { void load_from_file(const std::string& path); std::optional<FairValue> fair_value(const Ticker&) const; bool is_stale(const Ticker&, long now_ms, int max_age_secs) const; };`

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include <fstream>
#include "fair_value/fair_value.hpp"
using namespace te;
TEST(FairValue, LoadsAndLooksUp) {
  { std::ofstream f("fv_test.json");
    f << R"([{"ticker":"T","p_yes":0.62,"confidence":0.62,"asof":"2026-07-10T00:00:00Z"}])"; }
  FairValueProvider p; p.load_from_file("fv_test.json");
  auto fv = p.fair_value("T");
  ASSERT_TRUE(fv.has_value());
  EXPECT_NEAR(fv->p_yes, 0.62, 1e-9);
  EXPECT_FALSE(p.fair_value("MISSING").has_value());
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement** `fair_value.hpp`:

```cpp
#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include "core/types.hpp"
namespace te {
struct FairValue { double p_yes; double confidence; long asof_epoch_ms; };
class FairValueProvider {
 public:
  void load_from_file(const std::string& path);
  std::optional<FairValue> fair_value(const Ticker& t) const;
  bool is_stale(const Ticker& t, long now_ms, int max_age_secs) const;
 private:
  std::unordered_map<Ticker, FairValue> map_;
};
}
```

- [ ] **Step 4: Implement `fair_value.cpp`** (parse ISO8601 `asof` to epoch ms; nlohmann):

```cpp
#include "fair_value/fair_value.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <ctime>
namespace te {
static long iso_to_ms(const std::string& s) {
  std::tm tm{}; // parse "YYYY-MM-DDTHH:MM:SS"
  strptime(s.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
  return (long)timegm(&tm) * 1000L;
}
void FairValueProvider::load_from_file(const std::string& path) {
  std::ifstream f(path); if (!f) return;
  nlohmann::json j; f >> j;
  decltype(map_) fresh;
  for (auto& r : j) {
    FairValue fv{r.at("p_yes").get<double>(), r.at("confidence").get<double>(),
                 iso_to_ms(r.at("asof").get<std::string>())};
    fresh[r.at("ticker").get<std::string>()] = fv;
  }
  map_.swap(fresh);
}
std::optional<FairValue> FairValueProvider::fair_value(const Ticker& t) const {
  auto it = map_.find(t); if (it == map_.end()) return std::nullopt; return it->second;
}
bool FairValueProvider::is_stale(const Ticker& t, long now_ms, int max_age_secs) const {
  auto it = map_.find(t); if (it == map_.end()) return true;
  return (now_ms - it->second.asof_epoch_ms) > (long)max_age_secs * 1000L;
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): FairValueProvider reads fair_values.json with staleness check`.

---

### Task 10: MarketMap + watchlist config

**Files:**
- Create: `trading_engine/src/market_map/market_map.hpp`, `.cpp`
- Create: `trading_engine/config/watchlist.json`
- Test: `trading_engine/tests/test_market_map.cpp`

**Interfaces:**
- Produces:
  `struct GameRef { int home_team_id; int away_team_id; std::string game_date; };`
  `class MarketMap { void load(const std::string& path); std::vector<Ticker> watchlist() const; std::optional<GameRef> game_for(const Ticker&) const; };`

> v1 uses a **static** watchlist (`config/watchlist.json`) authored per slate — the same file `publish_fair_values.py` consumes, keeping both sides on one source of truth. Automatic ticker discovery via Kalshi's markets REST endpoint is deferred (noted in the spec's open questions).

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include <fstream>
#include "market_map/market_map.hpp"
using namespace te;
TEST(MarketMap, LoadsWatchlistAndResolves) {
  { std::ofstream f("wl_test.json");
    f << R"([{"ticker":"KXNBA-T","home_team_id":1610612744,"away_team_id":1610612747,"game_date":"2026-07-10"}])"; }
  MarketMap m; m.load("wl_test.json");
  ASSERT_EQ(m.watchlist().size(), 1u);
  EXPECT_EQ(m.watchlist()[0], "KXNBA-T");
  auto g = m.game_for("KXNBA-T");
  ASSERT_TRUE(g.has_value());
  EXPECT_EQ(g->home_team_id, 1610612744);
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement `market_map.hpp`**

```cpp
#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/types.hpp"
namespace te {
struct GameRef { int home_team_id; int away_team_id; std::string game_date; };
class MarketMap {
 public:
  void load(const std::string& path);
  std::vector<Ticker> watchlist() const;
  std::optional<GameRef> game_for(const Ticker& t) const;
 private:
  std::unordered_map<Ticker, GameRef> map_;
};
}
```

- [ ] **Step 4: Implement `market_map.cpp`**

```cpp
#include "market_map/market_map.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
namespace te {
void MarketMap::load(const std::string& path) {
  std::ifstream f(path); if (!f) throw std::runtime_error("watchlist not found: " + path);
  nlohmann::json j; f >> j;
  for (auto& r : j)
    map_[r.at("ticker")] = GameRef{r.at("home_team_id"), r.at("away_team_id"), r.at("game_date")};
}
std::vector<Ticker> MarketMap::watchlist() const {
  std::vector<Ticker> v; v.reserve(map_.size());
  for (auto& [k,_] : map_) v.push_back(k); return v;
}
std::optional<GameRef> MarketMap::game_for(const Ticker& t) const {
  auto it = map_.find(t); if (it == map_.end()) return std::nullopt; return it->second;
}
}
```

- [ ] **Step 5:** Create `config/watchlist.json` with `[]` (populated per slate). Run → PASS. **Step 6: Commit** `feat(engine): static MarketMap watchlist (ticker↔game)`.

---

## PHASE M3 — Strategy, risk, paper execution

### Task 11: Pricing — prob→cents, fee-aware edge threshold

**Files:**
- Create: `trading_engine/src/strategy/pricing.hpp`, `.cpp`
- Test: `trading_engine/tests/test_pricing.cpp`

**Interfaces:**
- Produces:
  `Cents fair_price_cents(double p_yes);  // round(100*p_yes), clamped 1..99`
  `int edge_threshold_cents(double confidence, const Config&);  // base_edge + fee + k*(1-confidence)`

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include "strategy/pricing.hpp"
#include "core/config.hpp"
using namespace te;
TEST(Pricing, FairPriceRoundsAndClamps) {
  EXPECT_EQ(fair_price_cents(0.624), 62);
  EXPECT_EQ(fair_price_cents(0.001), 1);   // clamp low
  EXPECT_EQ(fair_price_cents(0.999), 99);  // clamp high
}
TEST(Pricing, EdgeThresholdWidensWhenLessConfident) {
  Config c; c.base_edge_cents=2; c.fee_cents_per_contract=1; c.confidence_k=8.0;
  int hi = edge_threshold_cents(0.95, c); // 2+1+8*0.05 = 3.4 -> 3
  int lo = edge_threshold_cents(0.55, c); // 2+1+8*0.45 = 6.6 -> 6
  EXPECT_LT(hi, lo);
  EXPECT_EQ(hi, 3);
  EXPECT_EQ(lo, 6);
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement `pricing.hpp`**

```cpp
#pragma once
#include "core/types.hpp"
#include "core/config.hpp"
namespace te {
Cents fair_price_cents(double p_yes);
int edge_threshold_cents(double confidence, const Config& c);
}
```

- [ ] **Step 4: Implement `pricing.cpp`**

```cpp
#include "strategy/pricing.hpp"
#include <algorithm>
#include <cmath>
namespace te {
Cents fair_price_cents(double p_yes) {
  int c = (int)std::lround(100.0 * p_yes);
  return std::clamp(c, 1, 99);
}
int edge_threshold_cents(double confidence, const Config& c) {
  double t = c.base_edge_cents + c.fee_cents_per_contract + c.confidence_k * (1.0 - confidence);
  return (int)std::floor(t);
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): fee-aware pricing and confidence-scaled edge threshold`.

---

### Task 12: Arb detector

**Files:**
- Create: `trading_engine/src/strategy/arb.hpp`, `.cpp`
- Test: `trading_engine/tests/test_arb.cpp`

**Interfaces:**
- Consumes: `OrderBook`, `Config`.
- Produces:
  `struct ArbSignal { bool present; Action action; int qty; Cents yes_price; Cents no_price; };`
  `ArbSignal detect_arb(const OrderBook&, const Config&);`
  Logic: if `best_yes_ask + best_no_ask + 2*fee < 100` → buy both (locked profit). If `best_yes_bid + best_no_bid - 2*fee > 100` → sell both.

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include "strategy/arb.hpp"
using namespace te;
TEST(Arb, DetectsBuyBothWhenAsksUnderPar) {
  OrderBook b; // yes_ask = 100-no_bid, no_ask = 100-yes_bid
  b.apply_snapshot({/*yes bids*/{{40,5}}, /*no bids*/{{45,5}}});
  // yes_ask = 100-45 = 55, no_ask = 100-40 = 60 -> 55+60=115 (no arb)
  Config c; c.fee_cents_per_contract=1;
  EXPECT_FALSE(detect_arb(b,c).present);
  OrderBook b2;
  b2.apply_snapshot({{{52,5}}, {{53,5}}}); // yes_ask=47, no_ask=48 -> 95 < 100-2 -> arb
  auto s = detect_arb(b2,c);
  EXPECT_TRUE(s.present);
  EXPECT_EQ(s.action, Action::Buy);
  EXPECT_EQ(s.yes_price, 47);
  EXPECT_EQ(s.no_price, 48);
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement `arb.hpp`**

```cpp
#pragma once
#include "market_data/order_book.hpp"
#include "core/config.hpp"
namespace te {
struct ArbSignal { bool present{false}; Action action{Action::Buy}; int qty{0}; Cents yes_price{0}; Cents no_price{0}; };
ArbSignal detect_arb(const OrderBook& b, const Config& c);
}
```

- [ ] **Step 4: Implement `arb.cpp`**

```cpp
#include "strategy/arb.hpp"
#include <algorithm>
namespace te {
ArbSignal detect_arb(const OrderBook& b, const Config& c) {
  ArbSignal s;
  int fee2 = 2 * c.fee_cents_per_contract;
  auto ya = b.best_yes_ask(); auto na = b.best_no_ask();
  if (ya && na && (*ya + *na + fee2) < 100) {
    int q = std::min(b.qty_at(Side::No, 100 - *ya), b.qty_at(Side::Yes, 100 - *na));
    s = {true, Action::Buy, std::max(q,1), *ya, *na}; return s;
  }
  auto yb = b.best_yes_bid(); auto nb = b.best_no_bid();
  if (yb && nb && (*yb + *nb - fee2) > 100) {
    int q = std::min(b.qty_at(Side::Yes, *yb), b.qty_at(Side::No, *nb));
    s = {true, Action::Sell, std::max(q,1), *yb, *nb}; return s;
  }
  return s;
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): fee-aware intra-market arb detector`.

---

### Task 13: Market-maker quote construction

**Files:**
- Create: `trading_engine/src/strategy/market_maker.hpp`, `.cpp`
- Test: `trading_engine/tests/test_market_maker.cpp`

**Interfaces:**
- Produces:
  `struct Quote { Cents bid; Cents ask; int size; };`
  `Quote make_quote(Cents fair_price, double confidence, int inventory, const Config&);`
  Logic: `half_spread = max(1, edge_threshold_cents(confidence,c)/2)`; `skew = clamp(inventory/skew_div, -maxskew, maxskew)`; `bid=clamp(fair-half-skew,1,99)`, `ask=clamp(fair+half-skew,1,99)`, `size=max_order_size`.

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include "strategy/market_maker.hpp"
using namespace te;
TEST(MarketMaker, SymmetricWhenFlat) {
  Config c; c.base_edge_cents=2; c.fee_cents_per_contract=1; c.confidence_k=8.0; c.max_order_size=25;
  auto q = make_quote(62, 0.95, /*inventory*/0, c); // threshold 3 -> half 1
  EXPECT_EQ(q.bid, 61);
  EXPECT_EQ(q.ask, 63);
  EXPECT_EQ(q.size, 25);
}
TEST(MarketMaker, SkewsDownWhenLong) {
  Config c; c.base_edge_cents=2; c.fee_cents_per_contract=1; c.confidence_k=8.0; c.max_order_size=25;
  auto flat = make_quote(62, 0.95, 0, c);
  auto lng  = make_quote(62, 0.95, 50, c); // long inventory -> quotes shift down
  EXPECT_LT(lng.bid, flat.bid);
  EXPECT_LT(lng.ask, flat.ask);
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement `market_maker.hpp`**

```cpp
#pragma once
#include "core/types.hpp"
#include "core/config.hpp"
namespace te {
struct Quote { Cents bid; Cents ask; int size; };
Quote make_quote(Cents fair_price, double confidence, int inventory, const Config& c);
}
```

- [ ] **Step 4: Implement `market_maker.cpp`**

```cpp
#include "strategy/market_maker.hpp"
#include "strategy/pricing.hpp"
#include <algorithm>
namespace te {
Quote make_quote(Cents fair, double confidence, int inventory, const Config& c) {
  int half = std::max(1, edge_threshold_cents(confidence, c) / 2);
  int skew = std::clamp(inventory / 20, -5, 5);   // 1 cent per 20 contracts, capped ±5
  Cents bid = std::clamp(fair - half - skew, 1, 99);
  Cents ask = std::clamp(fair + half - skew, 1, 99);
  return {bid, ask, c.max_order_size};
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): inventory-skewed market-maker quoting`.

---

### Task 14: Edge-taker

**Files:**
- Create: `trading_engine/src/strategy/edge_taker.hpp`, `.cpp`
- Test: `trading_engine/tests/test_edge_taker.cpp`

**Interfaces:**
- Produces:
  `struct TakeSignal { bool present; Action action; Side side; Cents price; int size; };`
  `TakeSignal detect_take(const OrderBook&, Cents fair_price, int threshold, const Config&);`
  Logic: if `best_yes_ask <= fair - threshold` → Buy YES at ask. If `best_yes_bid >= fair + threshold` → Sell YES at bid.

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include "strategy/edge_taker.hpp"
using namespace te;
TEST(EdgeTaker, BuysWhenAskBelowFairMinusThreshold) {
  OrderBook b; b.apply_snapshot({{{40,5}},{{52,7}}}); // yes_ask=100-52=48
  Config c; c.max_order_size=25;
  auto s = detect_take(b, /*fair*/62, /*threshold*/5, c); // 48 <= 62-5 -> take
  EXPECT_TRUE(s.present);
  EXPECT_EQ(s.action, Action::Buy);
  EXPECT_EQ(s.side, Side::Yes);
  EXPECT_EQ(s.price, 48);
}
TEST(EdgeTaker, NoTakeInsideThreshold) {
  OrderBook b; b.apply_snapshot({{{40,5}},{{40,5}}}); // yes_ask=60
  Config c; c.max_order_size=25;
  EXPECT_FALSE(detect_take(b, 62, 5, c).present); // 60 > 57
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3–4: Implement** `edge_taker.hpp/.cpp`:

```cpp
// edge_taker.hpp
#pragma once
#include "market_data/order_book.hpp"
#include "core/config.hpp"
namespace te {
struct TakeSignal { bool present{false}; Action action{Action::Buy}; Side side{Side::Yes}; Cents price{0}; int size{0}; };
TakeSignal detect_take(const OrderBook& b, Cents fair, int threshold, const Config& c);
}
```
```cpp
// edge_taker.cpp
#include "strategy/edge_taker.hpp"
namespace te {
TakeSignal detect_take(const OrderBook& b, Cents fair, int threshold, const Config& c) {
  TakeSignal s;
  auto ya = b.best_yes_ask();
  if (ya && *ya <= fair - threshold) { s = {true, Action::Buy, Side::Yes, *ya, c.max_order_size}; return s; }
  auto yb = b.best_yes_bid();
  if (yb && *yb >= fair + threshold) { s = {true, Action::Sell, Side::Yes, *yb, c.max_order_size}; return s; }
  return s;
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): edge-taker over fair value with threshold`.

---

### Task 15: RiskManager + kill switch

**Files:**
- Create: `trading_engine/src/risk/risk_manager.hpp`, `.cpp`
- Test: `trading_engine/tests/test_risk_manager.cpp`

**Interfaces:**
- Consumes: `Config`, `Order` (defined here or in execution — define `Order` in `execution/order_venue.hpp`, Task 16, and have Task 15 include it). To avoid a cycle, define the shared `Order`/`Side` in `execution/order_venue.hpp` and include it here.
- Produces:
  `class RiskManager { RiskManager(const Config&); void set_position(const Ticker&, int); void record_realized_pnl(int cents); void trip_kill_switch(); bool killed() const; RiskDecision check(const Order&, bool fair_value_stale, bool book_crossed); };`
  `struct RiskDecision { bool allow; std::string reason; int approved_qty; };`

> Order type is introduced in Task 16 but referenced here — to keep tasks independently buildable, **Task 15 Step 0 creates `execution/order_venue.hpp` with the `Order`/`Fill` structs only** (the venue class body comes in Task 16). This is the one deliberate cross-task shared header.

- [ ] **Step 0: Create `execution/order_venue.hpp` shared types**

```cpp
#pragma once
#include <string>
#include "core/types.hpp"
namespace te {
struct Order { Ticker ticker; Side side; Action action; Cents price; int qty; };
struct Fill  { Ticker ticker; Side side; Action action; Cents price; int qty; };
}
```

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include "risk/risk_manager.hpp"
using namespace te;
static Order buy(const char* t, int qty){ return {t, Side::Yes, Action::Buy, 50, qty}; }
TEST(Risk, BlocksWhenStaleOrCrossedOrKilled) {
  Config c; c.max_contracts_per_market=100; c.max_order_size=25; c.max_daily_loss_cents=20000;
  c.max_aggregate_exposure_cents=500000;
  RiskManager r(c);
  EXPECT_FALSE(r.check(buy("T",10), /*stale*/true,  false).allow);
  EXPECT_FALSE(r.check(buy("T",10), false, /*crossed*/true ).allow);
  r.trip_kill_switch();
  EXPECT_TRUE(r.killed());
  EXPECT_FALSE(r.check(buy("T",10), false, false).allow);
}
TEST(Risk, CapsQtyToPositionLimit) {
  Config c; c.max_contracts_per_market=100; c.max_order_size=25;
  c.max_aggregate_exposure_cents=500000; c.max_daily_loss_cents=20000;
  RiskManager r(c);
  r.set_position("T", 95);                 // only 5 more allowed
  auto d = r.check(buy("T",25), false, false);
  EXPECT_TRUE(d.allow);
  EXPECT_EQ(d.approved_qty, 5);
}
TEST(Risk, KillsOnMaxDailyLoss) {
  Config c; c.max_contracts_per_market=100; c.max_order_size=25;
  c.max_aggregate_exposure_cents=500000; c.max_daily_loss_cents=20000;
  RiskManager r(c);
  r.record_realized_pnl(-20001);
  EXPECT_TRUE(r.killed());
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement `risk_manager.hpp`**

```cpp
#pragma once
#include <string>
#include <unordered_map>
#include "core/config.hpp"
#include "execution/order_venue.hpp"
namespace te {
struct RiskDecision { bool allow; std::string reason; int approved_qty; };
class RiskManager {
 public:
  explicit RiskManager(const Config& c) : c_(c) {}
  void set_position(const Ticker& t, int pos) { pos_[t] = pos; }
  void record_realized_pnl(int cents);
  void trip_kill_switch() { killed_ = true; }
  bool killed() const { return killed_; }
  RiskDecision check(const Order& o, bool fair_stale, bool book_crossed);
 private:
  const Config& c_;
  std::unordered_map<Ticker,int> pos_;
  int realized_pnl_cents_ = 0;
  bool killed_ = false;
};
}
```

- [ ] **Step 4: Implement `risk_manager.cpp`**

```cpp
#include "risk/risk_manager.hpp"
#include <algorithm>
namespace te {
void RiskManager::record_realized_pnl(int cents) {
  realized_pnl_cents_ += cents;
  if (realized_pnl_cents_ <= -c_.max_daily_loss_cents) killed_ = true;
}
RiskDecision RiskManager::check(const Order& o, bool fair_stale, bool book_crossed) {
  if (killed_)       return {false, "kill_switch", 0};
  if (fair_stale)    return {false, "stale_fair_value", 0};
  if (book_crossed)  return {false, "book_crossed", 0};
  int qty = std::min(o.qty, c_.max_order_size);
  int cur = pos_.count(o.ticker) ? pos_[o.ticker] : 0;
  int signed_dir = (o.action == Action::Buy) ? 1 : -1;
  int room = c_.max_contracts_per_market - std::abs(cur + signed_dir * 0); // headroom to cap
  int allowed = c_.max_contracts_per_market - std::abs(cur);
  if (o.action == Action::Buy && cur >= 0) qty = std::min(qty, std::max(0, allowed));
  if (o.action == Action::Sell && cur <= 0) qty = std::min(qty, std::max(0, allowed));
  if (qty <= 0) return {false, "position_limit", 0};
  return {true, "ok", qty};
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): risk manager with position caps, loss kill-switch, fail-closed gates`.

---

### Task 16: PaperVenue

**Files:**
- Modify: `trading_engine/src/execution/order_venue.hpp` (add `OrderVenue` interface)
- Create: `trading_engine/src/execution/paper_venue.hpp`, `.cpp`
- Test: `trading_engine/tests/test_paper_venue.cpp`

**Interfaces:**
- Produces:
  `class OrderVenue { virtual std::string place(const Order&) = 0; virtual void cancel(const std::string&) = 0; virtual int position(const Ticker&) const = 0; virtual int realized_pnl_cents() const = 0; };`
  `class PaperVenue : public OrderVenue` — fills marketable orders immediately against a supplied `OrderBook` up to available qty; tracks positions and realized P&L. YES position math: buying YES at price `p` costs `p` cents/contract; a later sell at `q` realizes `q-p`.

- [ ] **Step 1: Failing test**

```cpp
#include <gtest/gtest.h>
#include "execution/paper_venue.hpp"
using namespace te;
TEST(Paper, BuyThenSellRealizesPnl) {
  OrderBook book; book.apply_snapshot({{{60,100}},{{35,100}}}); // yes_ask=65, yes_bid=60
  PaperVenue v;
  // Buy 10 YES marketable: fills at yes_ask=65
  v.place_against(book, {"T", Side::Yes, Action::Buy, 65, 10});
  EXPECT_EQ(v.position("T"), 10);
  // Sell 10 YES marketable: fills at yes_bid=60 -> realized (60-65)*10 = -50
  v.place_against(book, {"T", Side::Yes, Action::Sell, 60, 10});
  EXPECT_EQ(v.position("T"), 0);
  EXPECT_EQ(v.realized_pnl_cents(), -50);
}
TEST(Paper, PartialFillToBookQty) {
  OrderBook book; book.apply_snapshot({{{60,3}},{{35,3}}}); // only 3 at touch
  PaperVenue v;
  v.place_against(book, {"T", Side::Yes, Action::Buy, 65, 10}); // only 3 available
  EXPECT_EQ(v.position("T"), 3);
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement interface in `order_venue.hpp`** (append):

```cpp
namespace te {
class OrderVenue {
 public:
  virtual ~OrderVenue() = default;
  virtual std::string place(const Order&) = 0;
  virtual void cancel(const std::string&) = 0;
  virtual int position(const Ticker&) const = 0;
  virtual int realized_pnl_cents() const = 0;
};
}
```

- [ ] **Step 4: Implement `paper_venue.hpp/.cpp`**

```cpp
// paper_venue.hpp
#pragma once
#include <unordered_map>
#include "execution/order_venue.hpp"
#include "market_data/order_book.hpp"
namespace te {
class PaperVenue : public OrderVenue {
 public:
  // marketable fill against a live book snapshot; returns fill id
  std::string place_against(const OrderBook& book, const Order& o);
  std::string place(const Order&) override { return "noop"; } // wired via place_against in v1
  void cancel(const std::string&) override {}
  int position(const Ticker& t) const override { auto it=pos_.find(t); return it==pos_.end()?0:it->second; }
  int realized_pnl_cents() const override { return realized_; }
 private:
  std::unordered_map<Ticker,int> pos_;
  std::unordered_map<Ticker,long> cost_basis_; // sum of signed (price*qty) for open position
  int realized_ = 0;
  long fill_id_ = 0;
};
}
```
```cpp
// paper_venue.cpp
#include "execution/paper_venue.hpp"
#include <algorithm>
namespace te {
std::string PaperVenue::place_against(const OrderBook& book, const Order& o) {
  // Determine touch price + available qty for a marketable YES order.
  Cents px; int avail;
  if (o.action == Action::Buy)  { auto a=book.best_yes_ask(); if(!a) return "noliq"; px=*a; avail=book.qty_at(Side::No, 100-*a); }
  else                          { auto b=book.best_yes_bid(); if(!b) return "noliq"; px=*b; avail=book.qty_at(Side::Yes, *b); }
  int qty = std::min(o.qty, std::max(0, avail));
  if (qty <= 0) return "noliq";
  int& pos = pos_[o.ticker];
  int dir = (o.action == Action::Buy) ? 1 : -1;
  // Realize against opposing open position using average cost basis.
  if ((dir > 0 && pos < 0) || (dir < 0 && pos > 0)) {
    int closing = std::min(qty, std::abs(pos));
    double avg = (double)cost_basis_[o.ticker] / pos; // avg signed price
    realized_ += (int)std::lround((dir < 0 ? (px - avg) : (avg - px)) * closing);
    cost_basis_[o.ticker] -= (long)std::lround(avg) * closing * (pos>0?1:-1);
    pos += dir * closing;
    qty -= closing;
  }
  if (qty > 0) { pos += dir * qty; cost_basis_[o.ticker] += (long)dir * px * qty; }
  return "fill-" + std::to_string(fill_id_++);
}
}
```
> If the average-cost-basis arithmetic feels heavy, the two tests pin the exact expected P&L; keep iterating the implementation until both pass. The simplest correct model: track `(total_qty, avg_price)` per ticker and realize `(sell-avg)` on reductions.

- [ ] **Step 5:** Run → PASS (iterate until `BuyThenSellRealizesPnl` gives -50 and partial fill gives 3). **Step 6: Commit** `feat(engine): paper venue with marketable fills, positions, realized P&L`.

---

### Task 17: StrategyEngine (orchestration)

**Files:**
- Create: `trading_engine/src/strategy/strategy_engine.hpp`, `.cpp`
- Test: `trading_engine/tests/test_strategy_engine.cpp`

**Interfaces:**
- Consumes: `FairValueProvider`, `MarketMap`, `RiskManager`, `PaperVenue`, `Config`, `Telemetry`, pricing/arb/mm/edge functions.
- Produces:
  `class StrategyEngine { StrategyEngine(deps...); void on_book_update(const Ticker&, const OrderBook&, long now_ms); };`
  On each update: skip if no fair value / stale / crossed (fail-closed via RiskManager); else run arb → edge-take → (fallback) market-make, gate each order through RiskManager, and execute approved orders via `PaperVenue::place_against`. Log every decision.

- [ ] **Step 1: Failing test** (inject a book that should trigger an edge-take buy and assert a paper position results)

```cpp
#include <gtest/gtest.h>
#include <sstream>
#include "strategy/strategy_engine.hpp"
using namespace te;
TEST(StrategyEngine, TakesEdgeAndOpensPaperPosition) {
  Config c; c.base_edge_cents=2; c.fee_cents_per_contract=1; c.confidence_k=8.0;
  c.max_order_size=25; c.max_contracts_per_market=100; c.max_aggregate_exposure_cents=500000;
  c.max_daily_loss_cents=20000; c.fair_value_max_age_secs=1800;
  FairValueProvider fv; // inject via test file
  { std::ofstream f("se_fv.json");
    f << R"([{"ticker":"T","p_yes":0.62,"confidence":0.95,"asof":"2026-07-10T00:00:00Z"}])"; }
  fv.load_from_file("se_fv.json");
  RiskManager risk(c); PaperVenue venue;
  std::ostringstream log; Telemetry tel(log);
  StrategyEngine eng(c, fv, risk, venue, tel);
  OrderBook b; b.apply_snapshot({{{40,50}},{{52,50}}}); // yes_ask = 48; fair=62, thr=3 -> take buy
  eng.on_book_update("T", b, /*now_ms*/1752105600000L); // ~2026-07-10, fresh
  EXPECT_GT(venue.position("T"), 0);
  EXPECT_NE(log.str().find("\"type\":\"take\""), std::string::npos);
}
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement `strategy_engine.hpp`**

```cpp
#pragma once
#include "core/config.hpp"
#include "fair_value/fair_value.hpp"
#include "risk/risk_manager.hpp"
#include "execution/paper_venue.hpp"
#include "telemetry/telemetry.hpp"
#include "market_data/order_book.hpp"
namespace te {
class StrategyEngine {
 public:
  StrategyEngine(const Config& c, FairValueProvider& fv, RiskManager& risk,
                 PaperVenue& venue, Telemetry& tel)
    : c_(c), fv_(fv), risk_(risk), venue_(venue), tel_(tel) {}
  void on_book_update(const Ticker& t, const OrderBook& b, long now_ms);
 private:
  const Config& c_; FairValueProvider& fv_; RiskManager& risk_;
  PaperVenue& venue_; Telemetry& tel_;
};
}
```

- [ ] **Step 4: Implement `strategy_engine.cpp`**

```cpp
#include "strategy/strategy_engine.hpp"
#include "strategy/pricing.hpp"
#include "strategy/arb.hpp"
#include "strategy/edge_taker.hpp"
#include "strategy/market_maker.hpp"
namespace te {
void StrategyEngine::on_book_update(const Ticker& t, const OrderBook& b, long now_ms) {
  auto fv = fv_.fair_value(t);
  bool stale = fv_.is_stale(t, now_ms, c_.fair_value_max_age_secs);
  bool crossed = b.crossed();
  if (!fv || stale || crossed) {
    tel_.event("skip", {{"ticker",t},{"has_fv",fv.has_value()},{"stale",stale},{"crossed",crossed}});
    return;
  }
  Cents fair = fair_price_cents(fv->p_yes);
  int thr = edge_threshold_cents(fv->confidence, c_);

  // 1) Arb
  if (auto s = detect_arb(b, c_); s.present) {
    Order o{t, Side::Yes, s.action, s.yes_price, s.qty};
    auto d = risk_.check(o, stale, crossed);
    if (d.allow) { o.qty=d.approved_qty; venue_.place_against(b, o);
      risk_.set_position(t, venue_.position(t));
      tel_.event("arb", {{"ticker",t},{"qty",d.approved_qty},{"yes",s.yes_price},{"no",s.no_price}}); }
    return;
  }
  // 2) Edge-take
  if (auto s = detect_take(b, fair, thr, c_); s.present) {
    Order o{t, s.side, s.action, s.price, s.size};
    auto d = risk_.check(o, stale, crossed);
    if (d.allow) { o.qty=d.approved_qty; venue_.place_against(b, o);
      risk_.set_position(t, venue_.position(t));
      tel_.event("take", {{"ticker",t},{"price",s.price},{"qty",d.approved_qty}}); }
    return;
  }
  // 3) Market-make (log the intended quote; paper resting-order sim is M4)
  auto q = make_quote(fair, fv->confidence, venue_.position(t), c_);
  tel_.event("quote", {{"ticker",t},{"bid",q.bid},{"ask",q.ask},{"size",q.size},{"fair",fair}});
}
}
```

- [ ] **Step 5:** Run → PASS. **Step 6: Commit** `feat(engine): strategy engine wiring arb→take→quote through risk into paper venue`.

---

### Task 18: main.cpp assembly + live paper run

**Files:**
- Create: `trading_engine/src/main.cpp`
- Modify: `trading_engine/CMakeLists.txt` (add `te_engine` executable)

**Interfaces:**
- Consumes: all components. Wires `MarketMap → MarketDataGateway.run(watchlist)`, a background thread refreshing `FairValueProvider.load_from_file` every `fair_value_refresh_secs`, and `gateway.on_update(...)` → `StrategyEngine.on_book_update(t, book, now_ms)`. Reads API creds from env `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`.

- [ ] **Step 1: Implement `main.cpp`** (no unit test — this is the composition root, verified by the live paper run)

```cpp
#include "core/config.hpp"
#include "market_map/market_map.hpp"
#include "fair_value/fair_value.hpp"
#include "market_data/gateway.hpp"
#include "risk/risk_manager.hpp"
#include "execution/paper_venue.hpp"
#include "strategy/strategy_engine.hpp"
#include "telemetry/telemetry.hpp"
#include <chrono>
#include <thread>
#include <atomic>
#include <iostream>
using namespace te;
static long now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
int main() {
  Config c = Config::load("config/engine.json");
  MarketMap map; map.load("config/watchlist.json");
  FairValueProvider fv; fv.load_from_file("fair_values.json");
  RiskManager risk(c); PaperVenue venue; Telemetry tel(std::cout);
  StrategyEngine eng(c, fv, risk, venue, tel);

  std::atomic<bool> running{true};
  std::thread refresher([&]{
    while (running) {
      std::this_thread::sleep_for(std::chrono::seconds(c.fair_value_refresh_secs));
      fv.load_from_file("fair_values.json");
    }
  });

  MarketDataGateway gw;
  gw.on_update([&](const Ticker& t, const OrderBook& b){ eng.on_book_update(t, b, now_ms()); });
  gw.run(map.watchlist());  // blocks; Ctrl-C to stop

  running = false; refresher.join();
  return 0;
}
```

- [ ] **Step 2:** Add executable to `CMakeLists.txt`:

```cmake
add_executable(te_engine src/main.cpp)
target_link_libraries(te_engine PRIVATE te_lib)
target_include_directories(te_engine PRIVATE src)
```

- [ ] **Step 3: Live paper session (manual verification).** Generate `fair_values.json` via `python -m backend_ml.publish_fair_values`, populate `config/watchlist.json` with tonight's tickers, export `KALSHI_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH`, run `./build/te_engine`, and confirm via the Telemetry JSONL that books update, skips fire on missing/stale fair values, and paper positions/P&L evolve sanely. **No real orders are placed** (PaperVenue only).

- [ ] **Step 4: Commit** `feat(engine): composition root + live paper run (M3 verified)`.

---

## PHASE M4 — Hardening

### Task 19: Deterministic record/replay harness

**Files:**
- Create: `trading_engine/src/market_data/recorder.hpp` (tee raw frames to a file in `gateway.run`)
- Create: `trading_engine/tests/test_replay.cpp`
- Create: fixture `trading_engine/tests/fixtures/replay_sample.jsonl` (a dozen captured frames)

**Interfaces:**
- Produces: `replay_file(path, StrategyEngine&, MarketDataGateway&)` that feeds each recorded frame through `gateway.handle_raw` and drives `on_book_update`. Deterministic: same fixture → identical Telemetry output.

- [ ] **Step 1: Failing test** — replay the fixture twice, assert byte-identical Telemetry output.

```cpp
#include <gtest/gtest.h>
#include <sstream><fstream>
#include "strategy/strategy_engine.hpp"
#include "market_data/gateway.hpp"
// helper reads fixtures/replay_sample.jsonl, calls gw.handle_raw per line,
// gw.on_update -> eng.on_book_update(t,b, FIXED_now_ms)
TEST(Replay, Deterministic) {
  auto run = []{
    std::ostringstream log; /* build engine with fixed config + fv file */
    /* ... feed fixture ... */ return log.str();
  };
  EXPECT_EQ(run(), run());
}
```

- [ ] **Step 2–5:** Implement the replay helper (fixed `now_ms` so staleness is deterministic), capture a real fixture during the Task 18 live run, run → PASS, commit `test(engine): deterministic record/replay harness`.

---

### Task 20: Kill-switch file flag + reconnection

**Files:**
- Modify: `trading_engine/src/risk/risk_manager.cpp` (poll a kill-file), `trading_engine/src/market_data/gateway.cpp` (reconnect + resubscribe on disconnect)
- Test: `trading_engine/tests/test_risk_manager.cpp` (extend: kill-file trips `killed()`)

- [ ] **Step 1: Failing test** — writing a sentinel file `KILL` makes `RiskManager::poll_kill_file("KILL")` set `killed()`.

```cpp
TEST(Risk, KillFileTripsSwitch) {
  Config c; RiskManager r(c);
  { std::ofstream f("KILL_TEST"); f << "1"; }
  r.poll_kill_file("KILL_TEST");
  EXPECT_TRUE(r.killed());
}
```

- [ ] **Step 2–4:** Implement `poll_kill_file` (if file exists → `trip_kill_switch()`), call it at the top of `StrategyEngine::on_book_update`; add Beast reconnect-with-backoff that re-sends the subscribe frame and relies on the fresh `orderbook_snapshot` to rebuild books. Run → PASS.

- [ ] **Step 5: Commit** `feat(engine): kill-switch file flag + WS reconnection with resubscribe`.

- [ ] **Step 6: Drill (manual).** During a live paper run, `touch KILL` and confirm quoting halts within one update cycle; kill the network and confirm books rebuild on reconnect. Log both via Telemetry.

---

## Milestone gates

- **M1 done:** Task 7 integration — live book matches Kalshi UI.
- **M2 done:** Task 10 — fair values load and map to tickers.
- **M3 done:** Task 18 — a live paper session produces sane positions/P&L with zero real orders.
- **M4 done:** Tasks 19–20 — replay is deterministic; kill switch + reconnect drilled.
- **M5 (NOT in this plan):** `LiveKalshiVenue` + go-live checklist — a separate spec/plan and explicit review.

---

## Self-review notes (author)

- **Spec coverage:** every spec component (MarketDataGateway→T7, FairValueProvider→T9, MarketMap→T10, StrategyEngine→T17, RiskManager→T15/T20, PaperVenue→T16, Telemetry→T3) and the Approach-A bridge (T8) has a task. Fair-value→cents conversion with fee buffer + confidence haircut → T11. Arb/MM/edge → T12/T13/T14.
- **Deferred, on purpose (matches spec non-goals):** LiveKalshiVenue, in-play, cross-market arb, Kalshi REST ticker auto-discovery. Called out at their tasks.
- **Cross-task type sharing:** `Order`/`Fill`/`OrderVenue` live in `execution/order_venue.hpp`, created early in T15 Step 0 and extended in T16 — the one intentional shared header, flagged in both tasks.
- **Open items to confirm during build (not placeholders — external facts):** exact Kalshi WS field names for snapshot/delta (T5 note), exact fee schedule (encoded in `engine.json`, validated at T11), Kalshi subscribe-frame shape (T7). Each has a concrete default and a verification step.
