#include <gtest/gtest.h>
#include <cstdio>
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

TEST(StrategyEngine, KillFileHaltsTrading) {
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
  { std::ofstream f("se_fv_kill.json");
    f << R"([{"ticker":"T","p_yes":0.62,"confidence":0.95,"asof":"2026-07-10T00:00:00Z"}])"; }
  fv.load_from_file("se_fv_kill.json");

  RiskManager risk(c);
  PaperVenue venue;
  std::ostringstream log;
  Telemetry tel(log);
  StrategyEngine eng(c, fv, risk, venue, tel);

  const std::string kill_path = "KILL_STRATEGY_TEST";
  { std::ofstream f(kill_path); f << "1"; }
  eng.set_kill_file(kill_path);

  OrderBook b;
  b.apply_snapshot({{{40, 50}}, {{52, 50}}});  // yes_ask = 48; fair=62, thr=3 -> would normally take buy

  const long now_ms = 1783641600000L + 60000L;
  eng.on_book_update("T", b, now_ms);

  EXPECT_EQ(venue.position("T"), 0);
  EXPECT_NE(log.str().find("\"type\":\"killed\""), std::string::npos);

  std::remove(kill_path.c_str());
}

TEST(StrategyEngine, DailyLossKillSwitchTripsFromRealizedPnl) {
  Config c;
  c.base_edge_cents = 2;
  c.fee_cents_per_contract = 1;
  c.confidence_k = 8.0;
  c.max_order_size = 25;
  c.max_contracts_per_market = 100;
  c.max_aggregate_exposure_cents = 500000;
  c.max_daily_loss_cents = 40;  // small, so a single losing round-trip trips it
  c.fair_value_max_age_secs = 1800;

  FairValueProvider fv;
  RiskManager risk(c);
  PaperVenue venue;
  std::ostringstream log;
  Telemetry tel(log);
  StrategyEngine eng(c, fv, risk, venue, tel);

  const long now_ms = 1783641600000L + 60000L;

  // 1) Open a long position via edge-take BUY: fair=62, thr=3 -> yes_ask=48
  // is <= 62-3=59, so the strategy buys 25 YES @ 48c.
  { std::ofstream f("se_fv_kill_pnl.json");
    f << R"([{"ticker":"T","p_yes":0.62,"confidence":0.95,"asof":"2026-07-10T00:00:00Z"}])"; }
  fv.load_from_file("se_fv_kill_pnl.json");

  OrderBook b1;
  b1.apply_snapshot({{{40, 50}}, {{52, 50}}});  // yes_bid=40, yes_ask=100-52=48
  eng.on_book_update("T", b1, now_ms);

  ASSERT_EQ(venue.position("T"), 25);
  ASSERT_FALSE(risk.killed());

  // 2) Fair value drops sharply (p_yes=0.10 -> fair=10, thr=3). Present a book
  // whose yes_ask (100-no_bid=100-50=50) is ABOVE fair-thr=7 (so the BUY leg
  // of detect_take does NOT fire) but whose yes_bid=30 is >= fair+thr=13, so
  // the strategy SELLS 25 YES @ 30c against its 25-lot long position bought
  // @ avg 48c -> realized P&L = (30-48)*25 = -450 cents, well past the
  // -40 cent max_daily_loss_cents kill threshold.
  { std::ofstream f("se_fv_kill_pnl.json");
    f << R"([{"ticker":"T","p_yes":0.10,"confidence":0.95,"asof":"2026-07-10T00:01:00Z"}])"; }
  fv.load_from_file("se_fv_kill_pnl.json");

  OrderBook b2;
  b2.apply_snapshot({{{30, 50}}, {{50, 50}}});  // yes_bid=30, yes_ask=100-50=50
  eng.on_book_update("T", b2, now_ms + 60000L);

  EXPECT_EQ(venue.position("T"), 0);  // 25-lot sell fully closed the long
  EXPECT_EQ(venue.realized_pnl_cents(), -450);
  EXPECT_TRUE(risk.killed());
  EXPECT_NE(log.str().find("\"type\":\"take\""), std::string::npos);

  // 3) Further updates must be blocked by the now-tripped kill switch: no new
  // position change, and a "killed" event is logged instead of a trade.
  OrderBook b3;
  b3.apply_snapshot({{{40, 50}}, {{52, 50}}});  // same as b1: would normally re-open a long
  eng.on_book_update("T", b3, now_ms + 120000L);

  EXPECT_EQ(venue.position("T"), 0);
  EXPECT_NE(log.str().find("\"type\":\"killed\""), std::string::npos);

  std::remove("se_fv_kill_pnl.json");
}
