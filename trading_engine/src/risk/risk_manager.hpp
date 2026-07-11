#pragma once
#include <string>
#include <unordered_map>
#include "core/config.hpp"
#include "execution/order_venue.hpp"
namespace te {
struct RiskDecision { bool allow; std::string reason; int approved_qty; };
class RiskManager {
 public:
  explicit RiskManager(const Config& c) : c_(c) {}
  void set_position(const Ticker& t, int pos) { pos_[t] = pos; }
  void record_realized_pnl(int cents);
  void trip_kill_switch() { killed_ = true; }
  bool killed() const { return killed_; }
  RiskDecision check(const Order& o, bool fair_stale, bool book_crossed);
 private:
  const Config& c_;
  std::unordered_map<Ticker,int> pos_;
  int realized_pnl_cents_ = 0;
  bool killed_ = false;
};
}
