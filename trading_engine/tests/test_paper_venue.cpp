#include <gtest/gtest.h>
#include "execution/paper_venue.hpp"
using namespace te;

TEST(Paper, BuyThenSellRealizesPnl) {
  OrderBook book; book.apply_snapshot({{{60,100}},{{35,100}}}); // yes_ask=65, yes_bid=60
  PaperVenue v;
  // Buy 10 YES marketable: fills at yes_ask=65
  v.place_against(book, {"T", Side::Yes, Action::Buy, 65, 10});
  EXPECT_EQ(v.position("T"), 10);
  // Sell 10 YES marketable: fills at yes_bid=60 -> realized (60-65)*10 = -50
  v.place_against(book, {"T", Side::Yes, Action::Sell, 60, 10});
  EXPECT_EQ(v.position("T"), 0);
  EXPECT_EQ(v.realized_pnl_cents(), -50);
}

TEST(Paper, PartialFillToBookQty) {
  OrderBook book; book.apply_snapshot({{{60,3}},{{35,3}}}); // only 3 at touch
  PaperVenue v;
  v.place_against(book, {"T", Side::Yes, Action::Buy, 65, 10}); // only 3 available
  EXPECT_EQ(v.position("T"), 3);
}

// Extra: pins avg-cost math across a flip (buy 10@65, sell 15@60 -> close 10
// realizing -50, then open 5 short @60).
TEST(Paper, FlipToShortAveragesCorrectly) {
  OrderBook book; book.apply_snapshot({{{60,100}},{{35,100}}}); // yes_ask=65, yes_bid=60
  PaperVenue v;
  v.place_against(book, {"T", Side::Yes, Action::Buy, 65, 10});
  EXPECT_EQ(v.position("T"), 10);
  v.place_against(book, {"T", Side::Yes, Action::Sell, 60, 15});
  EXPECT_EQ(v.position("T"), -5);
  EXPECT_EQ(v.realized_pnl_cents(), -50);
}
