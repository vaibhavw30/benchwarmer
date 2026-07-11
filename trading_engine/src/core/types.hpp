// trading_engine/src/core/types.hpp
#pragma once
#include <string>
#include <cstdint>
namespace te {
using Ticker = std::string;
using Cents  = int;            // on-book price, 1..99; settle 0/100
enum class Side   { Yes, No };
enum class Action { Buy, Sell };
struct PriceLevel { Cents price; int qty; };
constexpr Cents kSettleYes = 100;
constexpr Cents kSettleNo  = 0;
}
