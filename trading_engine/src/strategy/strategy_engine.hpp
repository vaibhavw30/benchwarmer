#pragma once
#include "core/config.hpp"
#include "fair_value/fair_value.hpp"
#include "risk/risk_manager.hpp"
#include "execution/paper_venue.hpp"
#include "telemetry/telemetry.hpp"
#include "market_data/order_book.hpp"
namespace te {

// StrategyEngine: per-book-update orchestration. Wires fair value -> arb /
// edge-take / market-make signal detection -> RiskManager gating ->
// PaperVenue execution, logging every decision via Telemetry. Fail-closed:
// any missing/stale fair value or crossed book skips the update entirely.
class StrategyEngine {
 public:
  StrategyEngine(const Config& c, FairValueProvider& fv, RiskManager& risk,
                 PaperVenue& venue, Telemetry& tel)
      : c_(c), fv_(fv), risk_(risk), venue_(venue), tel_(tel) {}

  void on_book_update(const Ticker& t, const OrderBook& b, long now_ms);

 private:
  const Config& c_;
  FairValueProvider& fv_;
  RiskManager& risk_;
  PaperVenue& venue_;
  Telemetry& tel_;
};
}  // namespace te
