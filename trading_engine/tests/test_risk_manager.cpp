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
