#pragma once
#include "market_data/order_book.hpp"
#include "core/config.hpp"
namespace te {
struct ArbSignal { bool present{false}; Action action{Action::Buy}; int qty{0}; Cents yes_price{0}; Cents no_price{0}; };
ArbSignal detect_arb(const OrderBook& b, const Config& c);
}
