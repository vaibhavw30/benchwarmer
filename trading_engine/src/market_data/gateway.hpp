#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "market_data/order_book.hpp"
namespace te {
class MarketDataGateway {
 public:
  using UpdateCb = std::function<void(const Ticker&, const OrderBook&)>;
  void on_update(UpdateCb cb) { cb_ = std::move(cb); }
  void handle_raw(std::string_view json);
  const OrderBook& book(const Ticker& t) const { return books_.at(t); }
  void run(const std::vector<Ticker>& watchlist); // live WS connect+auth+subscribe; see gateway_run.cpp
  void stop() { stop_ = true; }                   // request run()'s read loop to halt
 private:
  std::unordered_map<Ticker, OrderBook> books_;
  UpdateCb cb_;
  std::atomic<bool> stop_{false};
};
}
