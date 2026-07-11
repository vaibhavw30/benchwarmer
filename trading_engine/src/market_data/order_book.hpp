#pragma once
#include <map>
#include <optional>
#include <vector>
#include "core/types.hpp"
namespace te {
struct BookSnapshot { std::vector<PriceLevel> yes, no; };
struct BookDelta { Side side; Cents price; int delta_qty; };
class OrderBook {
 public:
  void apply_snapshot(const BookSnapshot& s);
  void apply_delta(const BookDelta& d);
  std::optional<Cents> best_yes_bid() const;
  std::optional<Cents> best_no_bid() const;
  std::optional<Cents> best_yes_ask() const; // 100 - best_no_bid
  std::optional<Cents> best_no_ask() const;  // 100 - best_yes_bid
  int qty_at(Side s, Cents p) const;
  bool crossed() const;
 private:
  std::map<Cents,int> yes_;  // price -> qty (YES bids)
  std::map<Cents,int> no_;   // price -> qty (NO bids)
  static std::optional<Cents> best(const std::map<Cents,int>& m);
};
}
