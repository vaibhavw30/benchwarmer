#pragma once
#include <string>
#include "core/types.hpp"
namespace te {
struct Order { Ticker ticker; Side side; Action action; Cents price; int qty; };
struct Fill  { Ticker ticker; Side side; Action action; Cents price; int qty; };

class OrderVenue {
 public:
  virtual ~OrderVenue() = default;
  virtual std::string place(const Order&) = 0;
  virtual void cancel(const std::string&) = 0;
  virtual int position(const Ticker&) const = 0;
  virtual int realized_pnl_cents() const = 0;
};
}
