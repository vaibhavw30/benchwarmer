#include "strategy/arb.hpp"
#include <algorithm>
namespace te {
ArbSignal detect_arb(const OrderBook& b, const Config& c) {
  ArbSignal s;
  int fee2 = 2 * c.fee_cents_per_contract;
  auto ya = b.best_yes_ask(); auto na = b.best_no_ask();
  if (ya && na && (*ya + *na + fee2) < 100) {
    int q = std::min(b.qty_at(Side::No, 100 - *ya), b.qty_at(Side::Yes, 100 - *na));
    s = {true, Action::Buy, std::max(q,1), *ya, *na}; return s;
  }
  auto yb = b.best_yes_bid(); auto nb = b.best_no_bid();
  if (yb && nb && (*yb + *nb - fee2) > 100) {
    int q = std::min(b.qty_at(Side::Yes, *yb), b.qty_at(Side::No, *nb));
    s = {true, Action::Sell, std::max(q,1), *yb, *nb}; return s;
  }
  return s;
}
}
