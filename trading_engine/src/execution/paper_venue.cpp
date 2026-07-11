#include "execution/paper_venue.hpp"
#include <algorithm>
#include <cmath>
namespace te {

std::string PaperVenue::place_against(const OrderBook& book, const Order& o) {
  // Determine the marketable touch price and available quantity for a YES
  // order (Kalshi convention: yes_ask = 100 - best_no_bid, and buying YES at
  // p consumes NO liquidity at 100-p; selling YES at p consumes YES liquidity
  // at p).
  Cents price;
  int avail;
  if (o.action == Action::Buy) {
    auto ask = book.best_yes_ask();
    if (!ask) return "noliq";
    price = *ask;
    avail = book.qty_at(Side::No, 100 - price);
  } else {
    auto bid = book.best_yes_bid();
    if (!bid) return "noliq";
    price = *bid;
    avail = book.qty_at(Side::Yes, price);
  }

  int qty = std::min(o.qty, std::max(0, avail));
  if (qty <= 0) return "noliq";

  int dir = (o.action == Action::Buy) ? 1 : -1;
  int& pos = pos_[o.ticker];
  double& avg = avg_price_[o.ticker];

  // Does this order reduce/close an existing opposing position (buy while
  // short, or sell while long)?
  bool reduces = (pos > 0 && dir < 0) || (pos < 0 && dir > 0);
  if (reduces) {
    int closing = std::min(qty, std::abs(pos));
    if (pos > 0) {
      // Long position being sold: realized = (sell_price - avg) * closed_qty.
      realized_ += static_cast<int>(std::lround((price - avg) * closing));
    } else {
      // Short position being covered: realized = (avg - buy_price) * closed_qty.
      realized_ += static_cast<int>(std::lround((avg - price) * closing));
    }
    pos += dir * closing;
    qty -= closing;

    if (pos == 0 && qty > 0) {
      // Order size exceeded the closed amount: the leftover opens a new
      // position on the other side at the fill price.
      pos = dir * qty;
      avg = price;
      qty = 0;
    }
    // Otherwise, any remaining same-side position keeps its existing avg_price.
  }

  if (qty > 0) {
    // Extends the same-side position (or opens from flat): weighted average
    // of the old position and the new quantity at `price`.
    int old_abs = std::abs(pos);
    avg = (old_abs == 0) ? static_cast<double>(price)
                          : (avg * old_abs + static_cast<double>(price) * qty) / (old_abs + qty);
    pos += dir * qty;
  }

  return "fill-" + std::to_string(fill_id_++);
}
}
