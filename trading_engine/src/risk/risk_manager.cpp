#include "risk/risk_manager.hpp"
#include <algorithm>
namespace te {
void RiskManager::record_realized_pnl(int cents) {
  realized_pnl_cents_ += cents;
  if (realized_pnl_cents_ <= -c_.max_daily_loss_cents) killed_ = true;
}
RiskDecision RiskManager::check(const Order& o, bool fair_stale, bool book_crossed) {
  if (killed_)       return {false, "kill_switch", 0};
  if (fair_stale)    return {false, "stale_fair_value", 0};
  if (book_crossed)  return {false, "book_crossed", 0};
  int qty = std::min(o.qty, c_.max_order_size);
  int cur = pos_.count(o.ticker) ? pos_[o.ticker] : 0;
  int allowed = c_.max_contracts_per_market - std::abs(cur);
  if (o.action == Action::Buy && cur >= 0) qty = std::min(qty, std::max(0, allowed));
  if (o.action == Action::Sell && cur <= 0) qty = std::min(qty, std::max(0, allowed));
  if (qty <= 0) return {false, "position_limit", 0};
  return {true, "ok", qty};
}
}
