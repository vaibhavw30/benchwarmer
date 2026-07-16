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
