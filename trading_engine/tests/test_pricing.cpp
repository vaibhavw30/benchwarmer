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
