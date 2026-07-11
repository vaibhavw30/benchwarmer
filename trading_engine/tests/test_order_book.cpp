// trading_engine/tests/test_order_book.cpp
#include <gtest/gtest.h>
#include "core/types.hpp"
#include "market_data/order_book.hpp"
using namespace te;

TEST(Scaffold, TypesCompile) {
  te::PriceLevel lvl{55, 10};
  EXPECT_EQ(lvl.price, 55);
  EXPECT_EQ(te::kSettleYes, 100);
}

TEST(OrderBook, SnapshotThenBestPrices) {
  OrderBook b;
  b.apply_snapshot({/*yes*/{{54,10},{53,5}}, /*no*/{{44,8},{43,3}}});
  EXPECT_EQ(b.best_yes_bid().value(), 54);
  EXPECT_EQ(b.best_no_bid().value(), 44);
  EXPECT_EQ(b.best_yes_ask().value(), 56); // 100 - best_no_bid(44)
  EXPECT_EQ(b.best_no_ask().value(), 46);  // 100 - best_yes_bid(54)
}

TEST(OrderBook, DeltaAddsAndRemovesLevels) {
  OrderBook b;
  b.apply_snapshot({{{54,10}},{{44,8}}});
  b.apply_delta({Side::Yes, 54, -10}); // remove all qty at 54
  b.apply_delta({Side::Yes, 55, 7});   // new best yes bid
  EXPECT_EQ(b.best_yes_bid().value(), 55);
  EXPECT_EQ(b.qty_at(Side::Yes, 54), 0);
}

TEST(OrderBook, DetectsCrossed) {
  OrderBook b;
  b.apply_snapshot({{{60,5}},{{45,5}}}); // yes_bid 60, yes_ask 55 -> crossed
  EXPECT_TRUE(b.crossed());
}
