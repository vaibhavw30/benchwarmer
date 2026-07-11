#include "strategy/market_maker.hpp"
#include "strategy/pricing.hpp"
#include <algorithm>
namespace te {
Quote make_quote(Cents fair, double confidence, int inventory, const Config& c) {
  int half = std::max(1, edge_threshold_cents(confidence, c) / 2);
  int skew = std::clamp(inventory / 20, -5, 5);   // 1 cent per 20 contracts, capped ±5
  Cents bid = std::clamp(fair - half - skew, 1, 99);
  Cents ask = std::clamp(fair + half - skew, 1, 99);
  return {bid, ask, c.max_order_size};
}
}
