#include <gtest/gtest.h>
#include "strategy/market_maker.hpp"
using namespace te;
TEST(MarketMaker, SymmetricWhenFlat) {
  Config c; c.base_edge_cents=2; c.fee_cents_per_contract=1; c.confidence_k=8.0; c.max_order_size=25;
  auto q = make_quote(62, 0.95, /*inventory*/0, c); // threshold 3 -> half 1
  EXPECT_EQ(q.bid, 61);
  EXPECT_EQ(q.ask, 63);
  EXPECT_EQ(q.size, 25);
}
TEST(MarketMaker, SkewsDownWhenLong) {
  Config c; c.base_edge_cents=2; c.fee_cents_per_contract=1; c.confidence_k=8.0; c.max_order_size=25;
  auto flat = make_quote(62, 0.95, 0, c);
  auto lng  = make_quote(62, 0.95, 50, c); // long inventory -> quotes shift down
  EXPECT_LT(lng.bid, flat.bid);
  EXPECT_LT(lng.ask, flat.ask);
}
