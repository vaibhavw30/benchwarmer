#include "market_data/order_book.hpp"
namespace te {
void OrderBook::apply_snapshot(const BookSnapshot& s) {
  yes_.clear(); no_.clear();
  for (auto& l : s.yes) if (l.qty > 0) yes_[l.price] = l.qty;
  for (auto& l : s.no)  if (l.qty > 0) no_[l.price]  = l.qty;
}
void OrderBook::apply_delta(const BookDelta& d) {
  auto& m = (d.side == Side::Yes) ? yes_ : no_;
  int q = m[d.price] + d.delta_qty;
  if (q <= 0) m.erase(d.price); else m[d.price] = q;
}
std::optional<Cents> OrderBook::best(const std::map<Cents,int>& m) {
  if (m.empty()) return std::nullopt;
  return m.rbegin()->first; // highest bid
}
std::optional<Cents> OrderBook::best_yes_bid() const { return best(yes_); }
std::optional<Cents> OrderBook::best_no_bid()  const { return best(no_); }
std::optional<Cents> OrderBook::best_yes_ask() const {
  auto nb = best_no_bid(); if (!nb) return std::nullopt; return 100 - *nb;
}
std::optional<Cents> OrderBook::best_no_ask()  const {
  auto yb = best_yes_bid(); if (!yb) return std::nullopt; return 100 - *yb;
}
int OrderBook::qty_at(Side s, Cents p) const {
  auto& m = (s == Side::Yes) ? yes_ : no_;
  auto it = m.find(p); return it == m.end() ? 0 : it->second;
}
bool OrderBook::crossed() const {
  auto yb = best_yes_bid(); auto ya = best_yes_ask();
  return yb && ya && *yb >= *ya;
}
}
