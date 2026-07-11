#include "core/config.hpp"
#include "market_map/market_map.hpp"
#include "fair_value/fair_value.hpp"
#include "market_data/gateway.hpp"
#include "risk/risk_manager.hpp"
#include "execution/paper_venue.hpp"
#include "strategy/strategy_engine.hpp"
#include "telemetry/telemetry.hpp"
#include <chrono>
#include <thread>
#include <atomic>
#include <iostream>
using namespace te;
static long now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
int main() {
  Config c = Config::load("config/engine.json");
  MarketMap map; map.load("config/watchlist.json");
  FairValueProvider fv; fv.load_from_file("fair_values.json");
  RiskManager risk(c); PaperVenue venue; Telemetry tel(std::cout);
  StrategyEngine eng(c, fv, risk, venue, tel);

  std::atomic<bool> running{true};
  std::thread refresher([&]{
    while (running) {
      std::this_thread::sleep_for(std::chrono::seconds(c.fair_value_refresh_secs));
      fv.load_from_file("fair_values.json");
    }
  });

  MarketDataGateway gw;
  gw.on_update([&](const Ticker& t, const OrderBook& b){ eng.on_book_update(t, b, now_ms()); });
  gw.run(map.watchlist());  // blocks; Ctrl-C to stop

  running = false; refresher.join();
  return 0;
}
