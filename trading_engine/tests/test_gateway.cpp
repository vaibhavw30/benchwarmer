#include <gtest/gtest.h>
#include "market_data/gateway.hpp"
using namespace te;
TEST(Gateway, RoutesSnapshotThenDeltaToBookAndFiresCallback) {
  MarketDataGateway g;
  int calls = 0; Ticker last;
  g.on_update([&](const Ticker& t, const OrderBook&){ ++calls; last = t; });
  g.handle_raw(R"({"type":"orderbook_snapshot","msg":{"market_ticker":"T","yes":[[54,10]],"no":[[44,8]]}})");
  g.handle_raw(R"({"type":"orderbook_delta","msg":{"market_ticker":"T","price":55,"delta":7,"side":"yes"}})");
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(last, "T");
  EXPECT_EQ(g.book("T").best_yes_bid().value(), 55);
}
