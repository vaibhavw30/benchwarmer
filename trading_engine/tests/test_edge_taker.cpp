#include <gtest/gtest.h>
#include "strategy/edge_taker.hpp"
using namespace te;
TEST(EdgeTaker, BuysWhenAskBelowFairMinusThreshold) {
  OrderBook b; b.apply_snapshot({{{40,5}},{{52,7}}}); // yes_ask=100-52=48
  Config c; c.max_order_size=25;
  auto s = detect_take(b, /*fair*/62, /*threshold*/5, c); // 48 <= 62-5 -> take
  EXPECT_TRUE(s.present);
  EXPECT_EQ(s.action, Action::Buy);
  EXPECT_EQ(s.side, Side::Yes);
  EXPECT_EQ(s.price, 48);
}
TEST(EdgeTaker, NoTakeInsideThreshold) {
  OrderBook b; b.apply_snapshot({{{40,5}},{{40,5}}}); // yes_ask=60
  Config c; c.max_order_size=25;
  EXPECT_FALSE(detect_take(b, 62, 5, c).present); // 60 > 57
}
