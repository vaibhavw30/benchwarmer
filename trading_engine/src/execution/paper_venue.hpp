#pragma once
#include <unordered_map>
#include "execution/order_venue.hpp"
#include "market_data/order_book.hpp"
namespace te {

// PaperVenue: simulated marketable-fill venue. Fills orders immediately
// against a supplied live OrderBook up to available touch quantity, and
// tracks per-ticker position + average entry price for realized P&L.
class PaperVenue : public OrderVenue {
 public:
  // Marketable fill against a live book snapshot; returns a fill id, or
  // "noliq" if there is no touch / no available quantity. This is the v1
  // entry point used by callers that already hold the book.
  std::string place_against(const OrderBook& book, const Order& o);

  // OrderVenue interface. v1 routes fills via place_against(); place() is a
  // noop placeholder until an internal order-routing loop exists.
  std::string place(const Order&) override { return "noop"; }
  void cancel(const std::string&) override {}
  int position(const Ticker& t) const override {
    auto it = pos_.find(t);
    return it == pos_.end() ? 0 : it->second;
  }
  int realized_pnl_cents() const override { return realized_; }

 private:
  std::unordered_map<Ticker, int> pos_;             // ticker -> signed qty
  std::unordered_map<Ticker, double> avg_price_;    // ticker -> avg entry price of open position
  int realized_ = 0;
  long fill_id_ = 0;
};
}
