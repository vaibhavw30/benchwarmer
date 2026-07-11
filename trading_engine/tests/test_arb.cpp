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
