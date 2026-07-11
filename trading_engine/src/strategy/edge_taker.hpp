#pragma once
#include "market_data/order_book.hpp"
#include "core/config.hpp"
namespace te {
struct TakeSignal { bool present{false}; Action action{Action::Buy}; Side side{Side::Yes}; Cents price{0}; int size{0}; };
TakeSignal detect_take(const OrderBook& b, Cents fair, int threshold, const Config& c);
}
