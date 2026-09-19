# Design: Trading-engine ingest hardening (spec 1 of 4)

**Date:** 2026-09-19
**Status:** Draft for review
**Scope:** `trading_engine/` (C++20). No Go in this spec.

---

## 1. Goal

Make the engine's ingest path correct enough to be the reference implementation
for a later Go port. Concretely:

1. **Book-state output.** A deterministic, integer-only, one-line-per-frame
   serialization of book state. This becomes the cross-language contract that
   the Go port (spec 4) must reproduce byte for byte.
2. **Frame validation.** Every frame is either applied completely or rejected
   with the book untouched. The inputs the September audit showed being
   accepted are now rejected.
3. **Error separation.** A bad frame is counted and reported. It never throws
   and never tears down the connection. Only a failed socket operation
   reconnects.
4. **Fixed-point units.** Prices, quantities and money move from integer cents
   to integer fixed-point, so the engine can represent Kalshi's documented
   sub-penny price grid and fractional contract counts.
5. **Both wire shapes.** The decoder accepts both the legacy integer-cent frame
   shape the engine was written against and the `*_dollars` / `*_fp` string
   shape in Kalshi's current documentation.

### Where this sits

| Spec | Subject | Depends on |
|---|---|---|
| **1 (this)** | Book-state format, validation, error separation, units, dual decoder | — |
| 3 | Recorder and segmented journal | 1 (what a decoded frame is) |
| — | Capture real Kalshi sessions (user step: needs credentials) | 3 |
| 2 | Sequence/gap policy | a real capture |
| 4 | Go port and cross-implementation diff | 1, 3 |

Specs 2–4 get their own design documents. Nothing here depends on facts we do
not yet have.

---

## 2. Context: what exists today

Grounded in a source read and a probe compiled against the built `te_lib`.

**Decoder** (`src/market_data/kalshi_messages.cpp:17-46`). Uses simdjson's
exception-throwing conversions. Reads `type`, and from `msg`: `market_ticker`,
`yes`, `no` (snapshot) or `price`, `delta`, `side` (delta), all as integers.
Never reads `seq` or `sid`. Any `side` other than `"yes"` becomes NO.

**Book** (`src/market_data/order_book.{hpp,cpp}`). Two `std::map<Cents,int>`
(YES bids, NO bids). `apply_snapshot` clears and reloads, skipping `qty <= 0`.
`apply_delta` adds and erases the level if the result is `<= 0`. Both return
`void`; neither validates.

**Gateway** (`src/market_data/gateway.cpp:4-12`). `handle_raw` parses, then
applies via `books_[ticker]`, which default-constructs a book for any unseen
ticker. No try/catch. In the live loop (`gateway_run.cpp:148-157`) an
exception from `handle_raw` is caught by the same handler as socket failures,
so a malformed frame causes a reconnect with backoff.

**Probe results** (all against the built library):

| Input | Today |
|---|---|
| Delta before any snapshot | Applied; creates a book from nothing |
| `seq` 1 → 7 | Applied; `seq` never read |
| Delta −999 on a level holding 30 | Applied; level erased |
| `side: "banana"` | Applied as NO |
| `price: 500` | Applied; best YES bid becomes 500 |
| Snapshot truncated mid-array | Silently ignored, no error |
| Delta missing `delta` | Throws `NO_SUCH_FIELD` |
| Snapshot missing `no` | Throws `NO_SUCH_FIELD` |

**Kalshi's documented wire format** (docs.kalshi.com, fetched 2026-09-14):

```json
{"type":"orderbook_snapshot","sid":2,"seq":2,"msg":{"market_ticker":"FED-23DEC-T3.00",
 "market_id":"…","yes_dollars_fp":[["0.0800","300.00"]],"no_dollars_fp":[["0.5400","20.00"]]}}
{"type":"orderbook_delta","sid":2,"seq":3,"msg":{"market_ticker":"FED-23DEC-T3.00",
 "market_id":"…","price_dollars":"0.9600","delta_fp":"-54.00","side":"yes","ts_ms":1669149841000}}
```

- `seq` is described as a number to check "if you want to guarantee you
  received all the messages". It is **per subscription (`sid`)**, not per
  ticker: one subscription covers many markets.
- An empty side's key is **omitted**, not sent as `[]`.
- Prices are dollar strings with up to four decimals; counts are fixed-point
  strings with two.

**Unresolved until a real capture:** whether the `/trade-api/ws/v2` endpoint the
engine uses (`gateway_run.cpp:58-60`; it answers 401, so it exists) emits the
legacy shape, the documented shape, or both. This spec handles both so the
capture confirms rather than blocks.

**Strategy, risk and venue** assume integer cents throughout: the `100 - p`
complement (`order_book.cpp:20,23`, `arb.cpp:8-15`, `paper_venue.cpp:17`),
`clamp(…,1,99)` (`pricing.cpp:7`, `market_maker.cpp:8-9`), the `inventory/20`
skew (`market_maker.cpp:7`). `PaperVenue` holds average entry price as a
`double` and rounds realized P&L with `lround` (`paper_venue.cpp:30,39,42,61`).

**Baseline behaviour to preserve.** `paper_session` over
`tests/fixtures/replay_sample.jsonl` with `p_yes=0.62, confidence=0.95` emits,
in order: `take@48`, `quote 60/62`, `take@48`, `skip(crossed)`, `take@48`,
`quote 58/60`, `quote 58/60`, `take@66`, `quote 59/61`, then
`session_end position=50 realized_pnl_cents=450`. All quantities 25.

---

## 3. Units

| Quantity | C++ type | Unit | Examples |
|---|---|---|---|
| Price | `Price` (`int64_t`) | 1/10,000 dollar | `"0.0800"` → 800; legacy `42` → 4,200 |
| Quantity | `Qty` (`int64_t`) | 1/100 contract | `"300.00"` → 30,000; legacy `50` → 5,000 |
| Money | `Micros` (`int64_t`) | 1/1,000,000 dollar | `Price × Qty`, exactly |

`kOneDollar = 10'000`, so the YES/NO complement is `kOneDollar - price`.
Valid on-book prices are `1 … 9'999`. `kSettleYes = 10'000`, `kSettleNo = 0`.

`Price × Qty` is dimensionally 10⁻⁴ $ × 10⁻² contract = 10⁻⁶ $ per contract,
so money needs no scaling factor and no rounding.

The aliases are plain `int64_t`, not strong types. Strong types would catch
unit mix-ups at compile time but add a large mechanical surface to a change
that already touches every layer; the unit tests in §9 cover the conversion
points instead. This is a deliberate trade-off, revisitable later.

### Config

`config/engine.json` keeps its existing key names and cent / whole-contract
values. `Config::load` converts on load:

| Key | Converted to |
|---|---|
| `max_order_size`, `max_contracts_per_market` | `Qty` (× 100) |
| `fee_cents_per_contract`, `base_edge_cents` | `Price` (× 100) |
| `max_daily_loss_cents`, `max_aggregate_exposure_cents` | `Micros` (× 10,000) |

Existing config files stay valid.

### Strategy boundary

Fair value arrives as a `double` probability. `fair_price(p_yes)` becomes
`100 × clamp(lround(100 × p_yes), 1, 99)`: the strategy keeps quoting on the
whole-cent grid it uses today. This is the one place a float becomes a price,
and the strategy is not part of the cross-language comparison.

**Strategy arithmetic is done in whole cents and scaled at the end**, not
redone in `Price` units. Today's integer divisions depend on it: with a 3-cent
threshold, `market_maker.cpp:6` computes a half-spread of `3/2 = 1` cent, but
the same division in `Price` units gives `300/2 = 150`, i.e. 1.5 cents, which
changes every quote. So `edge_threshold`, the market-maker half-spread and the
inventory skew (one cent per 20 contracts, capped at ±5) are computed in cents
and whole contracts exactly as now, then multiplied by 100. Arb's minimum
quantity `std::max(q, 1)` becomes one whole contract, `100`.

**Rule: on a book whose prices are all whole cents, strategy, risk and venue
decisions must be identical to today, with every price × 100, every quantity
× 100, and money × 10,000.** §9 enforces this against the baseline in §2.

### Paper venue P&L

Replace `avg_price_` (`double`) with an integer **cost basis** per ticker: the
signed total `Micros` paid for the open position. Opening or extending adds
`price × qty`. Closing `closing` of `|pos|` removes a proportional share:

```
removed = cost_basis × closing / |pos|      // integer division, truncates toward zero
cost_basis -= removed
realized  += proceeds_of_close − removed    // sign per long/short as today
```

Truncation can shift sub-micro-dollar amounts between successive partial
closes, but no money is created or destroyed: whatever `removed` omits stays in
`cost_basis` and is realized on a later close. **Once a position returns to
flat, cumulative realized P&L is exact.** This replaces today's `lround` on a
floating-point average, which has no such guarantee.

`realized_pnl_cents()` becomes `realized_pnl_micros()`. `RiskManager`'s
daily-loss accumulator moves to `Micros`.

---

## 4. Architecture and data flow

```
raw bytes ──▶ decode ──▶ DecodedFrame ──▶ OrderBook::apply ──▶ ApplyResult
                │                                │                   │
                └─ stateless rejects             └─ state rejects    ▼
                                                           BookStateWriter (optional sink)
```

- **Decode** judges the frame on its own: well-formed JSON, required fields,
  valid ticker, valid side, parseable numbers, prices and quantities in range.
- **Book** judges the frame against current state: has this ticker been
  snapshotted, would this delta drive a level negative.
- Nothing on this path throws. Every outcome is a value.

### Types

```cpp
enum class RejectReason : uint8_t {
  None, MalformedJson, MissingField, BadTicker, BadSide,
  PriceOutOfRange, QtyOutOfRange, BadFixedPoint, DuplicateLevel,
  DeltaBeforeSnapshot, LevelUnderflow, SeqGap /* defined; unused until spec 2 */
};

enum class FrameKind : uint8_t { Snapshot, Delta, Ignored };

struct DecodedFrame {
  FrameKind kind;
  Ticker ticker;                 // validated; empty if unknown
  std::optional<int64_t> seq;    // present only if the frame carried one
  BookSnapshot snapshot;         // levels in Price/Qty
  BookDelta delta;               // side, Price, signed Qty
};

struct DecodeResult { DecodedFrame frame; RejectReason reject; };
DecodeResult decode_frame(std::string_view json);

enum class IngestStatus : uint8_t { Applied, Rejected, Ignored };
struct IngestResult { IngestStatus status; RejectReason reason; Ticker ticker; };
```

`OrderBook::apply_snapshot` and `apply_delta` return `RejectReason`
(`None` = applied). A rejected call leaves the book exactly as it was.
`MarketDataGateway::handle_raw` returns `IngestResult`.

### Where each rule lives

| Rule | Reason | Layer |
|---|---|---|
| Not entirely well-formed JSON (including truncated), or a duplicate key in the top-level object or `msg` | `MalformedJson` | decode |
| `type` missing or not a string | `MalformedJson` | decode |
| Snapshot/delta missing `msg`, `market_ticker`, `side`, or its price/delta fields | `MissingField` | decode |
| Ticker empty, longer than 64 bytes, or outside `[A-Za-z0-9._-]` | `BadTicker` | decode |
| `side` not exactly `"yes"` or `"no"` | `BadSide` | decode |
| Number unparseable, too many decimals, sign where none allowed, legacy value not a JSON integer | `BadFixedPoint` | decode |
| Price outside `1 … 9'999` | `PriceOutOfRange` | decode |
| Snapshot level qty ≤ 0, or delta qty = 0 | `QtyOutOfRange` | decode |
| Same price twice on one side of a snapshot | `DuplicateLevel` | decode |
| Delta for a ticker never snapshotted | `DeltaBeforeSnapshot` | book |
| Delta would drive a level below 0 (including a negative delta on an absent level) | `LevelUnderflow` | book |
| Any `type` other than `orderbook_snapshot` / `orderbook_delta` | — (`Ignored`, not a reject) | decode |

A snapshot missing a side key entirely is **valid** and means that side is
empty, matching Kalshi's documented omission. A delta that brings a level to
exactly 0 erases it, as today.

`DeltaBeforeSnapshot` requires `OrderBook` to know whether it has ever been
snapshotted. The gateway stops default-constructing books through
`operator[]`: a delta for a ticker absent from `books_` is rejected without
inserting one.

---

## 5. Decoder

### Error handling

The decoder uses simdjson's **error-code** interface only (`.get(x)` returning
`simdjson::error_code`), never the throwing conversions used today. Each error
code maps to one `RejectReason`.

### Whole-document validation: DOM, not on-demand

The decoder switches from simdjson's on-demand API to its **DOM API**
(`simdjson::dom::parser`). On-demand parses lazily and never looks at bytes it
isn't asked for, so a frame whose ticker and price parse cleanly but which is
malformed further on can be accepted. Go's `encoding/json` validates the whole
document before returning anything. If the two engines validated different
amounts of the frame, the same input could be `ok` in one and `rej` in the
other. The DOM parser validates the entire document up front, which matches.
The performance cost is real but belongs to the spec-4 measurements, not to a
correctness decision.

### Duplicate keys

JSON permits duplicate keys, and implementations disagree about them:
`encoding/json` keeps the last value, while a DOM key lookup typically returns
the first. A frame like `{"price":42,"price":43}` would therefore decode
differently in each engine. **Any duplicate key in the top-level object or in
`msg` is `MalformedJson`.** Both engines must detect it explicitly (C++ by
iterating the object's fields; Go by walking tokens), and spec 4 inherits the
rule.

### Shape detection

Chosen per frame, by which fields are present in `msg`:

| Frame | Documented shape (preferred if present) | Legacy shape |
|---|---|---|
| Snapshot | `yes_dollars_fp` and/or `no_dollars_fp`: arrays of `[string, string]` | `yes` and/or `no`: arrays of `[int, int]` |
| Delta | `price_dollars` (string) + `delta_fp` (string) | `price` (int) + `delta` (int) |

If documented-shape keys are present they are used and legacy keys are
ignored. A snapshot with no side keys of either shape is an empty book on both
sides. A snapshot mixing shapes across its two sides (`yes` legacy, `no_dollars_fp`
documented) is `MalformedJson`: no real frame does this, and accepting it would
force both engines to agree on a precedence rule for nothing.

### Fixed-point parsing — integers only

Strings are parsed digit by digit into `int64_t`. **No `strtod`, no `double`.**
Parsing `"0.0800"` as a double and scaling is exactly the float divergence the
integer format exists to avoid.

| Field | Grammar | Scale |
|---|---|---|
| Price string | `DIGITS [ "." 1–4 DIGITS ]` | × 10,000 |
| Snapshot qty string | `DIGITS [ "." 1–2 DIGITS ]` | × 100 |
| Delta qty string | `[ "-" ] DIGITS [ "." 1–2 DIGITS ]` | × 100 |
| Legacy price | JSON integer | × 100 |
| Legacy qty / delta | JSON integer | × 100 |

`DIGITS` is 1–12 ASCII digits. No `+`, whitespace, exponent, bare `"."`, or
leading/trailing `"."`. Anything else is `BadFixedPoint`, never truncated. The
12-digit cap keeps every scaled value well inside `int64_t`.

### Sequence numbers

`seq` is read from the top level if present and carried on the `DecodedFrame`.
A present but non-integer `seq` is `MalformedJson`. Spec 1 **records** the seq
of the last frame applied to each ticker; it does **not** detect gaps. Gap
policy is per `sid`, not per ticker (§2), and waits for spec 2.

---

## 6. Book-state format — the cross-language contract

One line per frame that has an outcome other than `Ignored`. Written by hand,
not through `nlohmann::json` or `encoding/json`: library formatting is where
two correct implementations diverge.

```
{"f":0,"t":"KXNBA-25-ABC","s":"ok","seq":2,"yes":[[800,30000],[900,1000]],"no":[[5400,2000]]}
{"f":1,"t":"KXNBA-25-ABC","s":"rej","seq":2,"yes":[[800,30000],[900,1000]],"no":[[5400,2000]]}
{"f":3,"t":"","s":"rej","seq":-1,"yes":[],"no":[]}
```

Grammar, exactly:

- Keys in the order `f`, `t`, `s`, `seq`, `yes`, `no`. No whitespace. One `\n`
  at the end. ASCII only.
- `f` is the 0-based index of the frame in the input stream, **counting
  ignored frames**, so `f` is always the input line number and a diff points at
  the exact frame.
- `t` is the ticker **if and only if** the frame is well-formed JSON with no
  duplicate keys, has a `msg` object, and `msg.market_ticker` is present and
  passes the ticker rule. Otherwise `t` is `""`. Defining this by the frame's
  properties rather than by how far a parser got means it cannot depend on
  either implementation's parse order. Because tickers are restricted to
  `[A-Za-z0-9._-]`, no escaping is ever needed.
- For a rejected frame whose ticker is valid but has no book yet (e.g. a
  delta before any snapshot), `seq` is `-1` and both sides are `[]`; no book is
  created.
- `s` is `ok` or `rej`. Reason codes are **not** in the line; they are counted
  C++-side (§7) and not compared. `s` is kept because without it, two engines
  that disagree about whether to apply a no-op frame would emit identical lines.
- `seq` is the seq of the last frame *applied* to this ticker, or `-1` if none
  has carried one. Rejected frames do not advance it.
- `yes` and `no` are the book's levels after the frame, as `[price,qty]` pairs
  in `Price` and `Qty` units, **sorted by price ascending**. An empty side is
  `[]`. For `t:""` both are `[]`.
- Integers: plain decimal, no `+`, no leading zeros, `-` only for negatives.

A rejected frame for a known ticker emits that ticker's **unchanged** book.
That line, compared against the line before it, is what demonstrates fail-closed
behaviour rather than asserting it.

Note on naming: `Telemetry` already emits a field called `seq` which is its own
event counter (`telemetry.hpp:12`), unrelated to Kalshi's `seq`. The two
streams are separate files and never mix; a one-line comment in
`telemetry.hpp` records the distinction.

### Writer

`BookStateWriter` (new, `src/market_data/book_state.{hpp,cpp}`) owns an
`std::ostream&` and the frame counter. The gateway calls it once per frame when
a writer is attached. With no writer attached (the default, and the live
engine), nothing is serialized.

### Replay tool

`tools/book_replay.cpp` (new): reads a `.jsonl` file of raw frames, feeds each
line through `MarketDataGateway::handle_raw` with a `BookStateWriter` on
stdout, and prints a reject-count summary to stderr. This is the binary the
spec-4 cross-implementation diff will run; the Go port ships an equivalent.

---

## 7. Gateway and error separation

- `handle_raw` returns `IngestResult` and cannot throw on frame content.
- The gateway keeps a per-`RejectReason` counter, readable via
  `reject_counts()`.
- The update callback (strategy) fires only on `Applied`.
- `gateway_run.cpp`: the read loop calls `handle_raw` and continues on any
  result. The `catch` around the loop now only ever sees Beast/Asio/SSL
  failures, which is what its comment already claims. A rejected frame logs
  one line to stderr with its reason, rate-limited to one line per reason per
  second so a feed that turns bad cannot flood the log.

What a live engine does after a run of rejects is deliberately **not** decided
here: whether N consecutive `LevelUnderflow`s on one ticker should mark that
book stale and request `get_snapshot` belongs to spec 2's recovery policy.
Spec 1 guarantees only that a rejected frame changes nothing and is counted.

---

## 8. Files touched

| File | Change |
|---|---|
| `src/core/types.hpp` | `Price`, `Qty`, `Micros`, `kOneDollar`; settle constants rescaled; `PriceLevel` uses them |
| `src/core/config.{hpp,cpp}` | Typed fields; unit conversion on load |
| `src/core/fixed_point.{hpp,cpp}` (new) | Integer-only fixed-point parsing |
| `src/market_data/kalshi_messages.{hpp,cpp}` | `decode_frame`, dual shape, error codes, validation |
| `src/market_data/order_book.{hpp,cpp}` | `RejectReason` returns, snapshotted flag, underflow check, last-applied seq |
| `src/market_data/gateway.{hpp,cpp}` | `IngestResult`, no implicit book creation, counters, optional writer |
| `src/market_data/gateway_run.cpp` | Loop continues on rejects; rate-limited reject log |
| `src/market_data/book_state.{hpp,cpp}` (new) | `BookStateWriter` |
| `src/strategy/*`, `src/risk/*`, `src/execution/*` | Unit migration; `PaperVenue` cost basis |
| `src/telemetry/telemetry.hpp` | Comment only |
| `tools/book_replay.cpp` (new), `CMakeLists.txt` | New tool target; new sources |
| `tests/*` | See §9 |

Not touched: `kalshi_auth.*`, `fair_value.*`, `market_map.*`, and the TLS /
handshake half of `gateway_run.cpp`.

---

## 9. Testing

Parameterized gtest (`TEST_P` / `INSTANTIATE_TEST_SUITE_P`), one row per case,
wherever a rule has several inputs. This is the closest C++ equivalent to Go's
table-driven tests, so the two suites will read alike.

| Test file | Covers |
|---|---|
| `test_fixed_point.cpp` (new) | Every grammar row in §5, both edges of every range, the 12-digit cap, every rejected form |
| `test_kalshi_messages.cpp` | Both shapes for snapshot and delta; **the two documented frames from §2 verbatim**; omitted-side snapshot; mixed-shape rejection; every decode-layer reject reason |
| `test_order_book.cpp` | Delta before snapshot, underflow, exact-zero erase, rejected apply leaves the book byte-identical, seq recorded only on apply |
| `test_book_state.cpp` (new) | Exact output strings: empty book, one-sided book, rejected line, `t:""` line, negative and zero integers, ascending level order |
| `test_gateway.cpp` | Rejected frames do not fire the callback; counters increment per reason; no book created by a rejected delta |
| `test_ingest_audit.cpp` (new) | **The eight audit-probe inputs from §2 as one table**, each asserting the outcome below. This is the regression test for the audit itself |
| `test_ingest_no_throw.cpp` (new) | Every byte-prefix of every valid fixture frame, plus single-byte corruptions, through `handle_raw`: never throws, never `Applied` unless the prefix is the whole frame |
| `test_replay.cpp` | Unchanged determinism test, plus a **baseline-equivalence** assertion: the event sequence equals §2's baseline with prices and quantities × 100 and P&L × 10,000 |
| `test_paper_venue.cpp`, `test_risk_manager.cpp`, strategy tests | Existing assertions rescaled; new cost-basis tests including a multi-close sequence that returns to flat with exact P&L |

**Audit outcomes under spec 1.** Not every audit row becomes a reject, and the
test says so rather than hiding it:

| Audit input | Spec-1 outcome |
|---|---|
| Delta before any snapshot | Rejected (`DeltaBeforeSnapshot`); no book created |
| `seq` 1 → 7 | **Applied**, seq recorded as 7. Gap detection is spec 2; the test row carries a comment saying so |
| Delta −999 on a level holding 30 | Rejected (`LevelUnderflow`); book unchanged |
| `side: "banana"` | Rejected (`BadSide`); book unchanged |
| `price: 500` (legacy) | Rejected (`PriceOutOfRange`); book unchanged |
| Snapshot truncated mid-array | Rejected (`MalformedJson`); line emitted with `t:""` |
| Delta missing `delta` | Rejected (`MissingField`); book unchanged |
| Snapshot missing `no` | **Applied** with an empty NO side. That is correct per Kalshi's docs, which omit empty sides; today it throws and forces a reconnect |

**Golden book-state output.** `tests/fixtures/replay_sample.book_state` is the
expected `book_replay` output for the 9-frame fixture. It must be **derived by
hand from the fixture, not generated by running the new code** — a golden file
produced by the implementation under test only proves the code agrees with
itself. The fixture is small enough to trace level by level, and the
derivation is committed alongside it as a comment block in `test_replay.cpp`.

The existing 33 tests are substantially rewritten by the unit change. **The
suite's test count is not a claim until it is green again**, and whatever the
new count is gets reported as measured.

---

## 10. Out of scope

- Sequence-gap detection and recovery, including `get_snapshot` (spec 2).
- Recorder, journal, segment rotation (spec 3).
- Any Go code, and the cross-implementation diff (spec 4).
- Verifying the live WebSocket path against a real Kalshi account.
- Enforcing `max_aggregate_exposure` and `orders_per_sec_budget`.
- Two-legged arb execution.
- Strong unit types (§3).
- Moving `trading_engine/`. It stays where it is; the Go port will be a
  sibling directory.

---

## 11. Success criteria

1. Every test in §9 passes; the new suite count is recorded as measured.
2. `test_ingest_audit` passes with the outcomes in §9: six rejected with the
   book unchanged, the omitted-side snapshot applied, and the seq-gap row
   applied pending spec 2.
3. `book_replay` over the fixture matches the hand-derived golden file byte for
   byte.
4. No input in `test_ingest_no_throw`'s corpus makes `handle_raw` throw.
5. Replay telemetry matches the §2 baseline under the unit scaling.
6. Build is warning-clean under the existing flags.
7. Both READMEs are updated to describe validation, units, and the book-state
   format — and still state plainly that the live path is unverified and
   there is no recorder yet.
