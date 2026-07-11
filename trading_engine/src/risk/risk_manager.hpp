#pragma once
#include <fstream>
#include <string>
#include <unordered_map>
#include "core/config.hpp"
#include "execution/order_venue.hpp"
namespace te {
struct RiskDecision { bool allow; std::string reason; int approved_qty; };
// NOTE(v1 scope): Config::max_aggregate_exposure_cents and
// Config::orders_per_sec_budget are loaded but NOT enforced by this class in
// v1 -- only max_order_size, max_contracts_per_market, and max_daily_loss_cents
// gate orders below. Don't read the "fail-closed" behavior as covering
// aggregate exposure or order-rate limits.
class RiskManager {
 public:
  explicit RiskManager(const Config& c) : c_(c) {}
  void set_position(const Ticker& t, int pos) { pos_[t] = pos; }
  void record_realized_pnl(int cents);
  void trip_kill_switch() { killed_ = true; }
  bool killed() const { return killed_; }
  // Kill-switch file flag: if a file exists (openable) at `path`, trips the
  // kill switch. Cheap to call every book-update tick; a missing path is a
  // normal no-op (not an error) so operators can drop/remove the file live.
  void poll_kill_file(const std::string& path) {
    std::ifstream f(path);
    if (f.good()) trip_kill_switch();
  }
  RiskDecision check(const Order& o, bool fair_stale, bool book_crossed);
 private:
  const Config& c_;
  std::unordered_map<Ticker,int> pos_;
  int realized_pnl_cents_ = 0;
  bool killed_ = false;
};
}
