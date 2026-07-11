#include <gtest/gtest.h>
#include <fstream>
#include "market_map/market_map.hpp"
using namespace te;
TEST(MarketMap, LoadsWatchlistAndResolves) {
  { std::ofstream f("wl_test.json");
    f << R"([{"ticker":"KXNBA-T","home_team_id":1610612744,"away_team_id":1610612747,"game_date":"2026-07-10"}])"; }
  MarketMap m; m.load("wl_test.json");
  ASSERT_EQ(m.watchlist().size(), 1u);
  EXPECT_EQ(m.watchlist()[0], "KXNBA-T");
  auto g = m.game_for("KXNBA-T");
  ASSERT_TRUE(g.has_value());
  EXPECT_EQ(g->home_team_id, 1610612744);
}
