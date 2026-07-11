#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include "strategy/strategy_engine.hpp"
#include "market_data/gateway.hpp"
using namespace te;

// Deterministic replay of a hand-authored fixture of recorded Kalshi WS
// frames (tests/fixtures/replay_sample.jsonl, one ticker "T"). Proves the
// strategy pipeline (gateway -> book -> StrategyEngine -> risk/venue ->
// telemetry) is a pure function of (frames, fixed now_ms, config, fair
// value): replaying it twice must produce byte-identical Telemetry output.
namespace {

// asof "2026-07-10T00:00:00Z" == epoch 1783641600000 ms; now_ms is 60s
// later, well within fair_value_max_age_secs=1800, and held fixed across
// every book-update callback so staleness never depends on wall-clock time.
constexpr long kFixedNowMs = 1783641600000L + 60000L;

std::string run_replay() {
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

  // Tests run with WORKING_DIRECTORY = trading_engine/ (gtest_discover_tests
  // in CMakeLists.txt), so this relative path is correct there.
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

  return log.str();
}

}  // namespace

TEST(Replay, Deterministic) {
  std::string first = run_replay();
  std::string second = run_replay();
  EXPECT_EQ(first, second);

  // Prove the pipeline actually ran (not just that empty == empty) and that
  // the fixture exercises more than one branch of the strategy engine.
  EXPECT_NE(first.find("\"type\":\"take\""), std::string::npos);
  EXPECT_NE(first.find("\"type\":\"quote\""), std::string::npos);
  EXPECT_NE(first.find("\"type\":\"skip\""), std::string::npos);
}
