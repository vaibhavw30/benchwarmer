#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include "strategy/strategy_engine.hpp"
using namespace te;

TEST(StrategyEngine, TakesEdgeAndOpensPaperPosition) {
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
  { std::ofstream f("se_fv.json");
    f << R"([{"ticker":"T","p_yes":0.62,"confidence":0.95,"asof":"2026-07-10T00:00:00Z"}])"; }
  fv.load_from_file("se_fv.json");

  RiskManager risk(c);
  PaperVenue venue;
  std::ostringstream log;
  Telemetry tel(log);
  StrategyEngine eng(c, fv, risk, venue, tel);

  OrderBook b;
  b.apply_snapshot({{{40, 50}}, {{52, 50}}});  // yes_ask = 48; fair=62, thr=3 -> take buy

  // asof "2026-07-10T00:00:00Z" == epoch 1783641600000 ms; use a now_ms
  // fresh relative to that (well within fair_value_max_age_secs=1800).
  const long now_ms = 1783641600000L + 60000L;
  eng.on_book_update("T", b, now_ms);

  EXPECT_GT(venue.position("T"), 0);
  EXPECT_NE(log.str().find("\"type\":\"take\""), std::string::npos);
}
