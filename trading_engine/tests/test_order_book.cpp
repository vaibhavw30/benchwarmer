// trading_engine/tests/test_order_book.cpp
#include <gtest/gtest.h>
#include "core/types.hpp"
TEST(Scaffold, TypesCompile) {
  te::PriceLevel lvl{55, 10};
  EXPECT_EQ(lvl.price, 55);
  EXPECT_EQ(te::kSettleYes, 100);
}
