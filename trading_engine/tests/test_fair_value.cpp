#include <gtest/gtest.h>
#include <fstream>
#include "fair_value/fair_value.hpp"
using namespace te;
TEST(FairValue, LoadsAndLooksUp) {
  { std::ofstream f("fv_test.json");
    f << R"([{"ticker":"T","p_yes":0.62,"confidence":0.62,"asof":"2026-07-10T00:00:00Z"}])"; }
  FairValueProvider p; p.load_from_file("fv_test.json");
  auto fv = p.fair_value("T");
  ASSERT_TRUE(fv.has_value());
  EXPECT_NEAR(fv->p_yes, 0.62, 1e-9);
  EXPECT_FALSE(p.fair_value("MISSING").has_value());
}
TEST(FairValue, IsStale) {
  // asof "2026-07-10T00:00:00Z" == epoch 1752105600000 ms
  { std::ofstream f("fv_stale_test.json");
    f << R"([{"ticker":"S","p_yes":0.5,"confidence":0.5,"asof":"2026-07-10T00:00:00Z"}])"; }
  FairValueProvider p; p.load_from_file("fv_stale_test.json");
  auto sfv = p.fair_value("S");
  ASSERT_TRUE(sfv.has_value());
  const long asof_ms = sfv->asof_epoch_ms;  // 2026-07-10T00:00:00Z == 1783641600000 ms
  const int max_age = 1800;  // 30 min
  // within threshold (asof + 60s) -> fresh
  EXPECT_FALSE(p.is_stale("S", asof_ms + 60L * 1000L, max_age));
  // past threshold (asof + 3600s) -> stale
  EXPECT_TRUE(p.is_stale("S", asof_ms + 3600L * 1000L, max_age));
  // missing ticker -> stale
  EXPECT_TRUE(p.is_stale("NOPE", asof_ms + 60L * 1000L, max_age));
}
