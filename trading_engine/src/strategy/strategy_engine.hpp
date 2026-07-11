#pragma once
#include <string>
#include <utility>
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

  // Non-constructor setter (additive): wires a kill-switch file path without
  // touching the constructor signature main.cpp (Task 18) depends on. Empty
  // by default, meaning "no kill file configured" -> polling is skipped.
  void set_kill_file(std::string path) { kill_file_ = std::move(path); }

  void on_book_update(const Ticker& t, const OrderBook& b, long now_ms);

 private:
  const Config& c_;
  FairValueProvider& fv_;
  RiskManager& risk_;
  PaperVenue& venue_;
  Telemetry& tel_;
  std::string kill_file_;
  // Running total of PaperVenue's cumulative realized P&L last reported to
  // RiskManager. Feeding RiskManager the delta each fill (rather than the
  // running total) keeps its own accumulator equal to the venue's cumulative
  // realized P&L, which is what actually trips the daily-loss kill switch.
  int last_realized_cents_ = 0;
};
}  // namespace te
