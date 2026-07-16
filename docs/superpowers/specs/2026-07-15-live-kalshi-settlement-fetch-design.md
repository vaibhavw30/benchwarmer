# Live Kalshi Settlement Fetch — Design Spec

**Date:** 2026-07-15
**Status:** Approved design, ready for implementation plan
**Repo:** `/Users/vaibhav.wudaru/benchwarmer-nba`
**Predecessors:** [`2026-07-12-model-vs-close-clv-design.md`](2026-07-12-model-vs-close-clv-design.md), [`../plans/2026-07-12-model-vs-close-clv.md`](../plans/2026-07-12-model-vs-close-clv.md)

---

## 1. Purpose

`model-clv-report` computes model-beats-close (the gold-standard validation:
is `p_model` a sharper predictor than the closing line?) only when
`settlements.json` — a flat `{ticker: 0|1}` map of settled outcomes — is
populated. Today that file is populated by hand: `load_settlements` reads it,
but nothing writes it. Until it exists, the `beats_close` block of the report
stays `null` while model CLV and entry-edge still accrue.

This spec wires the **live Kalshi settlement fetch** that populates
`settlements.json`: after a slate settles, query each captured ticker's Kalshi
resolution and write the `{ticker: 0|1}` map. It ports the C++ engine's
RSA-PSS request signer to Python so the harness can make its own authenticated
REST calls, and adds a dedicated `fetch-settlements` CLI step.

**Non-goals:** the live price/odds capture fetchers (`_cmd_capture` stays a
deferred stub); any change to `model-clv-report`'s pure logic; the C++ engine;
real-money order routing; a full Kalshi REST client (only the one GET we need).

---

## 2. Scope decisions (locked during brainstorming)

| Decision | Choice | Consequence |
|---|---|---|
| Invocation | **Dedicated `fetch-settlements` subcommand** | A separate CLI step queries Kalshi and merges into `settlements.json`, then exits. `model-clv-report` stays offline/pure — it only reads the file. Clean separation of the network-touching step from the scoring step. |
| Ticker source | **Captured snapshot store** (`market_snapshots.jsonl`) | Settle exactly the games being scored, not the whole watchlist. Self-contained: whatever was captured is what gets settled. |
| Already-settled tickers | **Skip** | Settled results are immutable; no redundant queries. Enables idempotent accrual — re-running only fetches the still-open tickers. |
| Unsettled tickers | **Skip, retry next run** | Empty `result` (market still active) is not an error; picked up on a later run. |
| Auth | **Port the C++ `KalshiSigner`** to Python (`cryptography`) | Reuse the engine's exact RSA-PSS scheme and env var names (`KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`). The signer is its own module, reusable later by the deferred capture fetchers. |
| Live call | **Deferred-verify** | The real `requests.get` path needs creds + in-season markets; it is added to the before-live checklist. All pure logic is tested synthetically (no creds/network in the suite). |

---

## 3. Architecture

An extension of `backend_ml/signal_research/`. The post-game settlement step
sits beside the pre-game capture in the harness lifecycle:

```
capture (t-60 / tipoff)  →  fetch-settlements (after games settle)  →  model-clv-report
```

Three focused units:

### 3.1 `kalshi_auth.py` (new) — RSA-PSS request signing

Pure crypto, no network. Ports `trading_engine/src/market_data/kalshi_auth.cpp`
bit-for-bit.

```
class KalshiSigner:
    def __init__(self, key_id: str, private_key_pem: str | bytes): ...
    def sign(self, message: str) -> str          # base64 (no newline) RSA-PSS sig
    def headers(self, method: str, path: str, ts_ms: int) -> dict
    @classmethod
    def from_env(cls) -> "KalshiSigner"           # KALSHI_KEY_ID + KALSHI_PRIVATE_KEY_PATH
```

- **Signing scheme** (must match the C++ signer exactly): message
  `f"{ts_ms}{method}{path}"`; `EVP_DigestSign` with SHA-256; padding
  `RSA_PKCS1_PSS_PADDING`; salt length = digest length (32); MGF1 = SHA-256;
  base64-encode the signature with **no trailing newline**.
- **`headers(method, path, ts_ms)`** returns:
  - `KALSHI-ACCESS-KEY`: the key id
  - `KALSHI-ACCESS-TIMESTAMP`: `str(ts_ms)`
  - `KALSHI-ACCESS-SIGNATURE`: `sign(f"{ts_ms}{method}{path}")`
- **Path signed is path-only** — no host, no query string — matching the C++
  signer (`"/trade-api/v2/markets/{ticker}"`).
- **`from_env()`** reads `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`, loads the
  PEM from the file path, and raises a clear error if either is missing.
- Uses the `cryptography` library
  (`cryptography.hazmat.primitives.asymmetric.padding.PSS` with
  `mgf=MGF1(SHA256())`, `salt_length=PSS.DIGEST_LENGTH`).

### 3.2 `settlement.py` (new) — ticker → outcome

Pure functions with an injected fetcher seam, plus one thin live fetcher.

```
def build_settlements(tickers, fetch_result) -> dict[str, int]
def merge_settlements(existing: dict, new: dict) -> dict
def fetch_kalshi_result(ticker, signer, session, base_url=DEFAULT_BASE_URL) -> str | None
```

- **`build_settlements(tickers, fetch_result)`** — for each ticker,
  `r = fetch_result(ticker)`; `"yes" → 1`, `"no" → 0`, anything else (None,
  `""`, active/unknown) → **skip** (omit from the map). Per-ticker `try/except`:
  an exception on one ticker skips that ticker and continues; the rest still
  settle. Returns `{ticker: 0|1}` for the tickers that resolved.
- **`merge_settlements(existing, new)`** — union with existing preserved (a
  settled result never changes, so this is order-independent for real data;
  existing wins on any key collision). Returns a new dict.
- **`fetch_kalshi_result(ticker, signer, session, base_url)`** — the live seam
  (deferred-verify). Signs and issues
  `GET {base_url}/trade-api/v2/markets/{ticker}`, parses `market.result`,
  returns `"yes"`/`"no"`, or `None` if the field is empty/absent (market still
  active). `DEFAULT_BASE_URL = "https://api.elections.kalshi.com"`.

### 3.3 `report.py` — `fetch-settlements` subcommand

New subcommand + `_cmd_fetch_settlements(args)`:

1. Load unique tickers from the snapshot store (`load_snapshots()` groups rows
   by `game_id`; each moment-row carries a `ticker` — collect the distinct set).
2. Load the current `settlements.json` (`load_settlements()`); drop tickers
   already present (immutable, no re-query).
3. Build a signer from env (`KalshiSigner.from_env()`) — fail fast with a clear
   message if creds are missing, before any network call.
4. Open a `requests.Session`; `build_settlements(remaining_tickers, fetch)` where
   `fetch = lambda t: fetch_kalshi_result(t, signer, session)`.
5. `merge_settlements(existing, new)`; write `settlements.json` (pretty JSON).
6. Print a summary line:
   `settled M of N tickers (K already settled, J still open, E errors)`.

This command is **implemented** (not a `SystemExit` stub like `_cmd_capture`),
but it needs creds + network, so it can only be run live. Its pure helpers and
its ticker-extraction / skip-already-settled / merge logic are fully unit-tested
with an injected fetcher and temp files.

`model-clv-report`, `load_settlements`, `load_snapshots`, and `model_clv.py` are
**unchanged**.

---

## 4. Data flow

```
market_snapshots.jsonl ──▶ unique tickers (across all captured games)
                            minus tickers already in settlements.json (settled = immutable)
                              │
                              ▼  for each remaining ticker
             KalshiSigner.headers("GET", "/trade-api/v2/markets/{ticker}", ts_ms)
                              │
                              ▼
             GET https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}
                              │  parse market.result
                              ▼
                "yes" → 1   "no" → 0   ""/active/unknown → skip (retried next run)
                              │
                              ▼
             merge_settlements(existing, new) ──▶ write settlements.json
                              │
                              ▼
             model-clv-report (unchanged) reads settlements.json → beats_close populated
```

`YES = home-win` by the harness/engine convention, so `result == "yes"` ⇒
outcome `1` (home won), matching how `model_clv` and `load_settlements` interpret
the map.

---

## 5. Error handling

- **Per-ticker resilience** — a network error, non-200 status, 404, or malformed
  JSON on one ticker skips *that* ticker (counted as an error in the summary)
  and continues; the rest still settle. Mirrors `_cmd_capture`'s intended
  per-item isolation.
- **Unsettled** (`result` empty / market active) → `None` → skip, not an error.
  Retried on the next run — this is why we skip-if-already-settled rather than
  overwrite.
- **Missing creds** — `KalshiSigner.from_env()` raises a clear error at command
  start, before any network call.
- **Outcome validation** — only `"yes"`/`"no"` produce a 0/1; anything else is
  treated as unsettled. `load_settlements` independently rejects out-of-range
  values (`raise ValueError` on anything not in `{0,1}`), so a corrupt write
  cannot silently poison the report downstream.

---

## 6. Testing (synthetic-first — zero creds, zero network in the suite)

- **`kalshi_auth`**
  - Generate an RSA keypair in-test (`cryptography`), sign a known message, and
    **verify with the public key** (PSS/SHA-256/salt-32 roundtrip). This proves
    the scheme without needing Kalshi.
  - `headers()` returns the three `KALSHI-ACCESS-*` keys with the timestamp and
    message wired correctly (signature verifies against
    `f"{ts_ms}{method}{path}"`).
  - `from_env()` reads the env vars (monkeypatched) and a temp PEM file; raises
    when a var is missing.
- **`settlement.build_settlements`** — injected `fetch_result` returning
  yes/no/None maps to `{1, 0}` and skips None; a fetcher that raises on one
  ticker skips only that ticker, others survive.
- **`settlement.merge_settlements`** — union; existing entries preserved.
- **`report._cmd_fetch_settlements`** — injected fetcher + temp
  snapshot/settlement files: verifies ticker extraction from snapshots,
  skip-already-settled, merge, and the written file contents. No network.
- **Live `fetch_kalshi_result`** (real `requests.get`) — **deferred-verify**,
  added to the before-live checklist (Section F) alongside the existing
  `_cmd_capture` fetchers.

---

## 7. Deliverables

1. `backend_ml/signal_research/kalshi_auth.py` — `KalshiSigner`.
2. `backend_ml/signal_research/settlement.py` — `build_settlements`,
   `merge_settlements`, `fetch_kalshi_result`.
3. `report.py` — `fetch-settlements` subcommand + `_cmd_fetch_settlements`.
4. `backend_ml/requirements.txt` — add pinned `cryptography` (already resolved
   transitively; make it explicit).
5. Tests for all pure/injected paths above.
6. `docs/superpowers/before-live-checklist.md` — update Section F: mark the live
   settlement fetch as wired-but-deferred-verify; add the live-verification item
   (run the real GET against a settled market, confirm `result == "yes"` maps to
   home-win outcome 1 on a real ticker).
7. `docs/PROJECT-OVERVIEW.md` — note `fetch-settlements` in the CLI list /
   commands where the other subcommands are documented.

---

## 8. Security & scope posture (unchanged invariants)

- **Paper-only.** This adds a *read-only* settlement query. It does not place,
  cancel, or route any order. `LiveKalshiVenue` remains not built.
- **Secrets in env, never committed.** Kalshi RSA key + key id come from
  `KALSHI_KEY_ID` / `KALSHI_PRIVATE_KEY_PATH` — the same env vars the C++ engine
  uses. No key material is written to the repo. `settlements.json` and
  `market_snapshots.jsonl` remain gitignored (under `signal_research/artifacts/`).
- **The live call is the user's to run.** The command is built and tested, but
  running it against a real Kalshi account is a deferred, user-executed step
  (needs creds + in-season markets), consistent with the rest of Section F.
