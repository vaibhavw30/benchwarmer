# Live Kalshi Settlement Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `settlements.json` from live Kalshi market resolutions via a dedicated `fetch-settlements` CLI step, so `model-clv-report`'s `beats_close` block can accrue.

**Architecture:** A Python port of the C++ engine's RSA-PSS request signer (`kalshi_auth.py`), a pure settlement-mapping module with an injected fetcher seam plus one thin live REST fetcher (`settlement.py`), and a `fetch-settlements` subcommand in `report.py` that reads captured tickers from the snapshot store, queries Kalshi for the unsettled ones, and merges results into `settlements.json`. All pure/injected logic is unit-tested with no network or creds; the single live `requests.get` path is deferred-verify.

**Tech Stack:** Python 3.14 (venv at `.venv`), `cryptography` (RSA-PSS signing), `requests` (REST), `pytest`.

**Spec:** [`../specs/2026-07-15-live-kalshi-settlement-fetch-design.md`](../specs/2026-07-15-live-kalshi-settlement-fetch-design.md)

## Global Constraints

- **Signing scheme must match `trading_engine/src/market_data/kalshi_auth.cpp` bit-for-bit:** message = `f"{ts_ms}{method}{path}"`; RSA-PSS; hash SHA-256; MGF1 SHA-256; salt length = digest length (32, i.e. `padding.PSS.DIGEST_LENGTH`); signature base64-encoded with **no trailing newline** (plain `base64.b64encode(...).decode()` — standard base64 has no embedded newlines).
- **Path signed is path-only** — no host, no query string (e.g. `"/trade-api/v2/markets/{ticker}"`).
- **Creds from env, never committed:** `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` — the same env vars the C++ engine uses. No key material in the repo.
- **`YES = home-win`:** `result == "yes"` ⇒ outcome `1`; `result == "no"` ⇒ outcome `0`; anything else (empty / active / unknown) ⇒ skip (omit from the map).
- **Immutable settlements:** already-settled tickers are never re-queried; on a key collision during merge, the **existing** value wins.
- **Paper-only, read-only:** this adds a read-only settlement query. No order is placed, cancelled, or routed. `model-clv-report`, `load_settlements`, `load_snapshots`, and `model_clv.py` are unchanged.
- **Run all tests with the venv interpreter:** `.venv/bin/python -m pytest` (Homebrew python is externally-managed; bare `python` lacks the deps).
- **Tests live in** `backend_ml/signal_research/tests/`.

---

### Task 1: RSA-PSS request signer (`kalshi_auth.py`)

**Files:**
- Modify: `backend_ml/requirements.txt`
- Create: `backend_ml/signal_research/kalshi_auth.py`
- Test: `backend_ml/signal_research/tests/test_kalshi_auth.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `class KalshiSigner`
    - `KalshiSigner(key_id: str, private_key_pem: str | bytes)`
    - `.sign(message: str) -> str` — base64 RSA-PSS signature (no newline)
    - `.headers(method: str, path: str, ts_ms: int) -> dict` — keys `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, `KALSHI-ACCESS-SIGNATURE`
    - `KalshiSigner.from_env() -> KalshiSigner` — reads `KALSHI_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH`

- [ ] **Step 1: Add the `cryptography` dependency**

In `backend_ml/requirements.txt`, under the `# Data Fetching` section (right after the `requests>=2.31.0` line), add:

```
cryptography>=42.0.0  # RSA-PSS signing for Kalshi REST auth (signal_research/kalshi_auth.py)
```

- [ ] **Step 2: Write the failing tests**

Create `backend_ml/signal_research/tests/test_kalshi_auth.py`:

```python
import base64
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from backend_ml.signal_research.kalshi_auth import KalshiSigner


def _gen_pem():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return key, pem


def _verify(public_key, message: str, b64_sig: str):
    # Raises InvalidSignature if the scheme does not match.
    public_key.verify(
        base64.b64decode(b64_sig),
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_sign_roundtrips_with_public_key():
    key, pem = _gen_pem()
    signer = KalshiSigner("kid-123", pem)
    msg = "1700000000000GET/trade-api/v2/markets/T1"
    b64_sig = signer.sign(msg)
    assert "\n" not in b64_sig
    _verify(key.public_key(), msg, b64_sig)  # no raise == scheme matches


def test_sign_accepts_str_pem():
    key, pem = _gen_pem()
    signer = KalshiSigner("kid", pem.decode())
    _verify(key.public_key(), "m", signer.sign("m"))


def test_headers_wire_key_timestamp_and_signature():
    key, pem = _gen_pem()
    signer = KalshiSigner("kid-abc", pem)
    h = signer.headers("GET", "/trade-api/v2/markets/T1", 1700000000000)
    assert h["KALSHI-ACCESS-KEY"] == "kid-abc"
    assert h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"
    # Signature must verify against ts_ms + method + path.
    _verify(key.public_key(),
            "1700000000000GET/trade-api/v2/markets/T1",
            h["KALSHI-ACCESS-SIGNATURE"])


def test_from_env_reads_key_id_and_pem_file(tmp_path, monkeypatch):
    _, pem = _gen_pem()
    pem_path = tmp_path / "key.pem"
    pem_path.write_bytes(pem)
    monkeypatch.setenv("KALSHI_KEY_ID", "env-kid")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(pem_path))
    signer = KalshiSigner.from_env()
    h = signer.headers("GET", "/p", 1)
    assert h["KALSHI-ACCESS-KEY"] == "env-kid"


def test_from_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("KALSHI_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(RuntimeError):
        KalshiSigner.from_env()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend_ml/signal_research/tests/test_kalshi_auth.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research.kalshi_auth'`.

- [ ] **Step 4: Write the implementation**

Create `backend_ml/signal_research/kalshi_auth.py`:

```python
"""RSA-PSS request signing for Kalshi REST, ported bit-for-bit from the C++
engine's KalshiSigner (trading_engine/src/market_data/kalshi_auth.cpp).

Pure crypto, no network. The message signed for a REST call is
``f"{ts_ms}{method}{path}"`` (path only, no host/query), signed with RSA-PSS /
SHA-256 / MGF1-SHA-256 / salt length = digest length (32), base64 no-newline.
"""
import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiSigner:
    def __init__(self, key_id: str, private_key_pem):
        self.key_id = key_id
        if isinstance(private_key_pem, str):
            private_key_pem = private_key_pem.encode()
        self._key = serialization.load_pem_private_key(private_key_pem,
                                                       password=None)

    def sign(self, message: str) -> str:
        sig = self._key.sign(
            message.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def headers(self, method: str, path: str, ts_ms: int) -> dict:
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
            "KALSHI-ACCESS-SIGNATURE": self.sign(f"{ts_ms}{method}{path}"),
        }

    @classmethod
    def from_env(cls) -> "KalshiSigner":
        key_id = os.environ.get("KALSHI_KEY_ID")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not key_path:
            raise RuntimeError(
                "KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set to "
                "fetch settlements (see before-live-checklist).")
        pem = Path(key_path).read_bytes()
        return cls(key_id, pem)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend_ml/signal_research/tests/test_kalshi_auth.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add backend_ml/requirements.txt backend_ml/signal_research/kalshi_auth.py backend_ml/signal_research/tests/test_kalshi_auth.py
git commit -m "feat(signal): Python KalshiSigner (RSA-PSS) ported from C++ engine"
```

---

### Task 2: Settlement mapping (`settlement.py`)

**Files:**
- Create: `backend_ml/signal_research/settlement.py`
- Test: `backend_ml/signal_research/tests/test_settlement.py`

**Interfaces:**
- Consumes: `KalshiSigner` (Task 1) — passed into `fetch_kalshi_result`, not imported here.
- Produces:
  - `DEFAULT_BASE_URL: str`
  - `build_settlements(tickers, fetch_result) -> dict[str, int]` — `fetch_result(ticker) -> "yes"|"no"|None`; maps `"yes"→1`, `"no"→0`, else skip; per-ticker `try/except` skips on error.
  - `merge_settlements(existing: dict, new: dict) -> dict` — union, existing wins on collision.
  - `fetch_kalshi_result(ticker, signer, session, base_url=DEFAULT_BASE_URL) -> str | None` — live seam; signed `GET /trade-api/v2/markets/{ticker}`, returns `market.result` (`"yes"`/`"no"`) or `None`.

- [ ] **Step 1: Write the failing tests**

Create `backend_ml/signal_research/tests/test_settlement.py`:

```python
from backend_ml.signal_research import settlement


def test_build_settlements_maps_yes_no_and_skips_unsettled():
    results = {"T1": "yes", "T2": "no", "T3": None, "T4": ""}
    out = settlement.build_settlements(
        ["T1", "T2", "T3", "T4"], lambda t: results[t])
    assert out == {"T1": 1, "T2": 0}   # T3/T4 skipped


def test_build_settlements_skips_ticker_that_raises():
    def fetch(t):
        if t == "BOOM":
            raise RuntimeError("network down")
        return "yes"
    out = settlement.build_settlements(["A", "BOOM", "B"], fetch)
    assert out == {"A": 1, "B": 1}     # BOOM skipped, others survive


def test_merge_settlements_union_existing_wins():
    existing = {"T1": 1, "T2": 0}
    new = {"T2": 1, "T3": 1}           # T2 collides
    merged = settlement.merge_settlements(existing, new)
    assert merged == {"T1": 1, "T2": 0, "T3": 1}   # existing T2 preserved


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []
    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        return _FakeResp(self._payload)


class _StubSigner:
    def headers(self, method, path, ts_ms):
        return {"KALSHI-ACCESS-KEY": "k"}


def test_fetch_kalshi_result_parses_market_result():
    session = _FakeSession({"market": {"result": "yes"}})
    r = settlement.fetch_kalshi_result("T1", _StubSigner(), session,
                                       base_url="https://example.test")
    assert r == "yes"
    url, headers = session.calls[0]
    assert url == "https://example.test/trade-api/v2/markets/T1"
    assert headers == {"KALSHI-ACCESS-KEY": "k"}


def test_fetch_kalshi_result_returns_none_when_active():
    session = _FakeSession({"market": {"result": ""}})
    assert settlement.fetch_kalshi_result("T1", _StubSigner(), session) is None


def test_fetch_kalshi_result_returns_none_when_field_absent():
    session = _FakeSession({"market": {}})
    assert settlement.fetch_kalshi_result("T1", _StubSigner(), session) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend_ml/signal_research/tests/test_settlement.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_ml.signal_research.settlement'`.

- [ ] **Step 3: Write the implementation**

Create `backend_ml/signal_research/settlement.py`:

```python
"""Map captured Kalshi tickers to settled outcomes {ticker: 0|1}.

build_settlements / merge_settlements are pure (fetcher injected). The live
fetch_kalshi_result issues one signed REST GET and is deferred-verified by the
user (needs creds + in-season markets); tests exercise it with a fake session.

Convention: YES = home-win, so result "yes" -> 1, "no" -> 0.
"""
import time

DEFAULT_BASE_URL = "https://api.elections.kalshi.com"


def build_settlements(tickers, fetch_result) -> dict:
    out = {}
    for ticker in tickers:
        try:
            result = fetch_result(ticker)
        except Exception:
            continue                      # skip this ticker, keep the rest
        if result == "yes":
            out[ticker] = 1
        elif result == "no":
            out[ticker] = 0
        # else: unsettled/unknown -> skip (retried on a later run)
    return out


def merge_settlements(existing: dict, new: dict) -> dict:
    merged = dict(new)
    merged.update(existing)               # existing wins: settlements immutable
    return merged


def fetch_kalshi_result(ticker, signer, session, base_url=DEFAULT_BASE_URL):
    """Signed GET /trade-api/v2/markets/{ticker}; return market.result or None.

    Live seam (deferred-verify). Raises on network/HTTP error so the caller's
    per-ticker try/except in build_settlements can skip just this ticker.
    """
    path = f"/trade-api/v2/markets/{ticker}"
    ts_ms = int(time.time() * 1000)
    headers = signer.headers("GET", path, ts_ms)
    resp = session.get(base_url + path, headers=headers, timeout=10)
    resp.raise_for_status()
    market = resp.json().get("market", {})
    result = market.get("result")
    if result in ("yes", "no"):
        return result
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend_ml/signal_research/tests/test_settlement.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add backend_ml/signal_research/settlement.py backend_ml/signal_research/tests/test_settlement.py
git commit -m "feat(signal): settlement mapping (build/merge + live Kalshi result fetch)"
```

---

### Task 3: `fetch-settlements` CLI subcommand (`report.py`)

**Files:**
- Modify: `backend_ml/signal_research/report.py`
- Test: `backend_ml/signal_research/tests/test_report_fetch_settlements.py`

**Interfaces:**
- Consumes: `load_snapshots()`, `load_settlements()`, `SETTLEMENTS_PATH` (existing in `report.py`); `settlement.build_settlements`/`merge_settlements`/`fetch_kalshi_result` (Task 2); `kalshi_auth.KalshiSigner.from_env` (Task 1).
- Produces:
  - `_unique_tickers(snaps: dict) -> list[str]` — distinct, sorted tickers across all snapshot rows.
  - `fetch-settlements` subcommand → `_cmd_fetch_settlements(args)`; writes `SETTLEMENTS_PATH`, prints a summary.

- [ ] **Step 1: Write the failing tests**

Create `backend_ml/signal_research/tests/test_report_fetch_settlements.py`:

```python
import json
import pytest
from backend_ml.signal_research import report
from backend_ml.signal_research.market_capture import ENTRY_MOMENT, CLOSING_MOMENT


def test_unique_tickers_dedups_across_games_and_moments():
    snaps = {
        "g1": {ENTRY_MOMENT: {"ticker": "T1"}, CLOSING_MOMENT: {"ticker": "T1"}},
        "g2": {ENTRY_MOMENT: {"ticker": "T2"}},
        "g3": {ENTRY_MOMENT: {}},          # no ticker -> ignored
    }
    assert report._unique_tickers(snaps) == ["T1", "T2"]


def test_cmd_fetch_settlements_merges_and_writes(tmp_path, monkeypatch):
    # Snapshot store with two captured tickers; T1 already settled.
    snaps = {
        "g1": {ENTRY_MOMENT: {"ticker": "T1"}},
        "g2": {ENTRY_MOMENT: {"ticker": "T2"}},
    }
    out_path = tmp_path / "settlements.json"
    out_path.write_text('{"T1": 1}')      # T1 already settled -> skipped

    monkeypatch.setattr(report, "load_snapshots", lambda: snaps)
    monkeypatch.setattr(report, "SETTLEMENTS_PATH", str(out_path))
    # load_settlements's default path arg is bound at def-time, so point the
    # whole function at our pre-seeded file (re-reads on call).
    monkeypatch.setattr(
        report, "load_settlements",
        lambda: {k: int(v) for k, v in json.loads(out_path.read_text()).items()})

    # Bypass real creds: from_env returns a dummy signer (fetch is stubbed).
    monkeypatch.setattr(
        "backend_ml.signal_research.kalshi_auth.KalshiSigner.from_env",
        classmethod(lambda cls: object()))
    # Fetcher: only T2 should be queried (T1 already settled); it settles "no".
    queried = []
    def fake_fetch(ticker, signer, session, base_url=None):
        queried.append(ticker)
        return "no"
    monkeypatch.setattr(
        "backend_ml.signal_research.settlement.fetch_kalshi_result", fake_fetch)

    report._cmd_fetch_settlements(None)

    assert queried == ["T2"]              # already-settled T1 not re-queried
    written = json.loads(out_path.read_text())
    assert written == {"T1": 1, "T2": 0}  # existing T1 preserved, T2 added
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend_ml/signal_research/tests/test_report_fetch_settlements.py -q`
Expected: FAIL — `AttributeError: module 'backend_ml.signal_research.report' has no attribute '_unique_tickers'`.

- [ ] **Step 3: Add `_unique_tickers` and `_cmd_fetch_settlements` to `report.py`**

Insert these two functions after `_cmd_model_clv_report` (before `_cmd_capture`) in `backend_ml/signal_research/report.py`:

```python
def _unique_tickers(snaps) -> list:
    tickers = set()
    for moments in snaps.values():
        for row in moments.values():
            t = row.get("ticker")
            if t:
                tickers.add(t)
    return sorted(tickers)


def _cmd_fetch_settlements(args):
    # Live, creds+network path. Populates SETTLEMENTS_PATH from Kalshi
    # resolutions for the tickers already captured in the snapshot store.
    import requests
    from backend_ml.signal_research import settlement
    from backend_ml.signal_research.kalshi_auth import KalshiSigner

    snaps = load_snapshots()
    existing = load_settlements()
    candidates = _unique_tickers(snaps)
    remaining = [t for t in candidates if t not in existing]

    signer = KalshiSigner.from_env()      # fail-fast if creds missing
    session = requests.Session()
    new = settlement.build_settlements(
        remaining,
        lambda t: settlement.fetch_kalshi_result(t, signer, session))
    merged = settlement.merge_settlements(existing, new)

    out = Path(SETTLEMENTS_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, sort_keys=True))

    already = len(candidates) - len(remaining)
    unresolved = len(remaining) - len(new)
    print(f"settled {len(new)} of {len(candidates)} captured tickers "
          f"({already} already settled, {unresolved} unresolved) "
          f"-> {SETTLEMENTS_PATH}")
```

- [ ] **Step 4: Register the subcommand in `main()`**

In `backend_ml/signal_research/report.py`, inside `main()`, add this parser registration immediately after the `pcap = sub.add_parser("capture")` / `pcap.set_defaults(...)` block:

```python
    pfs = sub.add_parser("fetch-settlements")
    pfs.set_defaults(func=_cmd_fetch_settlements)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend_ml/signal_research/tests/test_report_fetch_settlements.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the whole harness suite (no regressions)**

Run: `.venv/bin/python -m pytest backend_ml/signal_research/ -q`
Expected: PASS — all prior tests plus the new ones (no failures, no errors).

- [ ] **Step 7: Commit**

```bash
git add backend_ml/signal_research/report.py backend_ml/signal_research/tests/test_report_fetch_settlements.py
git commit -m "feat(signal): fetch-settlements subcommand populates settlements.json"
```

---

### Task 4: Documentation (checklist + overview)

**Files:**
- Modify: `docs/superpowers/before-live-checklist.md`
- Modify: `docs/PROJECT-OVERVIEW.md`

**Interfaces:**
- Consumes: the `fetch-settlements` subcommand (Task 3). No code, no tests.

- [ ] **Step 1: Update the before-live checklist Section F item**

In `docs/superpowers/before-live-checklist.md`, replace the existing bullet that begins **"Live Kalshi settlement for model-CLV."** (the `- [ ]` item around line 68) with:

```markdown
- [ ] **Live Kalshi settlement for model-CLV (wired — needs live verification).**
  `python -m backend_ml.signal_research.report fetch-settlements` reads the
  captured tickers from `market_snapshots.jsonl`, signs a `GET
  /trade-api/v2/markets/{ticker}` per ticker with `KalshiSigner` (RSA-PSS,
  ported from the C++ engine), maps `result` "yes"->1 / "no"->0, skips
  already-settled and still-open tickers, and merges into
  `signal_research/artifacts/settlements.json`. Pure/injected paths are tested;
  the real `settlement.fetch_kalshi_result` REST call is unrun here. **To
  verify live:** set `KALSHI_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH`, run
  `fetch-settlements` against a slate that has already settled, and confirm on
  the Kalshi website that a home-win game wrote `1` and a home-loss game wrote
  `0` (i.e. `result == "yes"` really is the home side). Then
  `model-clv-report` populates `beats_close`. Pass `--signal-version
  raw|recalibrated` to `model-clv-report` matching how `fair_values.json` was
  published at capture time.
```

- [ ] **Step 2: Note `fetch-settlements` in PROJECT-OVERVIEW**

In `docs/PROJECT-OVERVIEW.md`, in the `report.py` bullet under §3 (currently:
`**report.py** — CLI: evaluate, capture, clv-report, model-clv-report.`),
replace it with:

```markdown
- **`report.py`** — CLI: `evaluate`, `capture`, `clv-report`,
  `model-clv-report`, `fetch-settlements` (queries Kalshi for captured tickers'
  resolutions and populates `settlements.json`; needs creds).
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/before-live-checklist.md docs/PROJECT-OVERVIEW.md
git commit -m "docs: fetch-settlements wired (checklist + overview)"
```

---

## Self-Review

**Spec coverage:**
- §3.1 `kalshi_auth.py` (KalshiSigner: sign/headers/from_env, scheme) → Task 1. ✓
- §3.2 `settlement.py` (build/merge/fetch) → Task 2. ✓
- §3.3 `fetch-settlements` subcommand + ticker extraction + skip-settled + merge + write + summary → Task 3. ✓
- §4 data flow (snapshots → sign → GET → map → merge → write) → Tasks 2+3. ✓
- §5 error handling (per-ticker skip, unsettled skip, missing-creds fail-fast, outcome validation via existing `load_settlements`) → Tasks 1 (from_env raise), 2 (try/except + None), 3 (from_env fail-fast). ✓
- §6 testing (auth roundtrip, headers, from_env; build/merge; fetch parse; command with injected fetcher) → Tasks 1–3 tests. ✓
- §7 deliverables 1–7 → Tasks 1 (dep+auth), 2 (settlement), 3 (subcommand), 4 (docs 6+7). ✓
- §8 posture (paper-only, read-only, secrets in env) → Global Constraints + no engine/model changes. ✓

**Note on the summary line (intentional deviation):** the spec §3.3 step 6 lists "J still open, E errors" separately. `build_settlements` swallows both into "not returned" to keep the pure function free of network-error accounting, so the plan collapses them into a single `unresolved` count. Faithful to intent (an honest per-run summary), simpler boundary.

**Placeholder scan:** no TBD/TODO; every code and test step shows complete content. ✓

**Type consistency:** `KalshiSigner(key_id, private_key_pem)`, `.sign(str)->str`, `.headers(method, path, ts_ms)->dict`, `.from_env()` used identically in Tasks 1→2→3. `build_settlements(tickers, fetch_result)`, `merge_settlements(existing, new)`, `fetch_kalshi_result(ticker, signer, session, base_url=...)` defined in Task 2 and called with matching signatures in Task 3. `_unique_tickers(snaps)->list` defined and tested in Task 3. ✓
