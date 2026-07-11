#include <gtest/gtest.h>
#include "market_data/kalshi_messages.hpp"
using namespace te;
TEST(Parse, Snapshot) {
  auto m = parse_ws_message(R"({"type":"orderbook_snapshot","msg":{
    "market_ticker":"KXNBA-25-ABC","yes":[[54,10],[53,5]],"no":[[44,8]]}})");
  EXPECT_EQ(m.kind, MsgKind::Snapshot);
  EXPECT_EQ(m.ticker, "KXNBA-25-ABC");
  ASSERT_EQ(m.snapshot.yes.size(), 2u);
  EXPECT_EQ(m.snapshot.yes[0].price, 54);
  EXPECT_EQ(m.snapshot.no[0].qty, 8);
}
TEST(Parse, Delta) {
  auto m = parse_ws_message(R"({"type":"orderbook_delta","msg":{
    "market_ticker":"KXNBA-25-ABC","price":55,"delta":-3,"side":"yes"}})");
  EXPECT_EQ(m.kind, MsgKind::Delta);
  EXPECT_EQ(m.delta.side, Side::Yes);
  EXPECT_EQ(m.delta.price, 55);
  EXPECT_EQ(m.delta.delta_qty, -3);
}
TEST(Parse, Other) {
  auto m = parse_ws_message(R"({"type":"subscribed","msg":{}})");
  EXPECT_EQ(m.kind, MsgKind::Other);
}
