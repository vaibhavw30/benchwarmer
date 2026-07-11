// trading_engine/tests/test_config.cpp
#include <gtest/gtest.h>
#include "core/config.hpp"
TEST(Config, LoadsEngineJson) {
  auto c = te::Config::load("config/engine.json");
  EXPECT_EQ(c.max_contracts_per_market, 100);
  EXPECT_EQ(c.fee_cents_per_contract, 1);
  EXPECT_GT(c.confidence_k, 0.0);
}
