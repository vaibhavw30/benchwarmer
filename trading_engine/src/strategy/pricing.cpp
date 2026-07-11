#include "strategy/pricing.hpp"
#include <algorithm>
#include <cmath>
namespace te {
Cents fair_price_cents(double p_yes) {
  int c = (int)std::lround(100.0 * p_yes);
  return std::clamp(c, 1, 99);
}
int edge_threshold_cents(double confidence, const Config& c) {
  double t = c.base_edge_cents + c.fee_cents_per_contract + c.confidence_k * (1.0 - confidence);
  return (int)std::floor(t);
}
}
