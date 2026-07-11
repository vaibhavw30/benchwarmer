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
