#include "strategy/edge_taker.hpp"
namespace te {
TakeSignal detect_take(const OrderBook& b, Cents fair, int threshold, const Config& c) {
  TakeSignal s;
  auto ya = b.best_yes_ask();
  if (ya && *ya <= fair - threshold) { s = {true, Action::Buy, Side::Yes, *ya, c.max_order_size}; return s; }
  auto yb = b.best_yes_bid();
  if (yb && *yb >= fair + threshold) { s = {true, Action::Sell, Side::Yes, *yb, c.max_order_size}; return s; }
  return s;
}
}
