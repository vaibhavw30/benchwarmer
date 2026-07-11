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
  try {
    Config c = Config::load("config/engine.json");
    MarketMap map; map.load("config/watchlist.json");
    FairValueProvider fv; fv.load_from_file("fair_values.json");
    RiskManager risk(c); PaperVenue venue; Telemetry tel(std::cout);
    StrategyEngine eng(c, fv, risk, venue, tel);
    eng.set_kill_file("KILL");  // touch ./KILL during a live run to halt trading (Task 20)

    // Started only after config loads, so a startup failure above lands in the
    // catch below without a thread to join. Sleep is chunked into 1s ticks so
    // shutdown (running=false) is observed promptly rather than after a full
    // refresh interval.
    std::atomic<bool> running{true};
    std::thread refresher([&]{
      while (running) {
        for (int i = 0; i < c.fair_value_refresh_secs && running; ++i)
          std::this_thread::sleep_for(std::chrono::seconds(1));
        if (running) fv.load_from_file("fair_values.json");
      }
    });
    // RAII: guarantees the refresher is stopped and joined before `running`/`fv`/`c`
    // (which it captures by reference) are destroyed — including on the exception
    // path out of gw.run() below, where a still-joinable std::thread would
    // otherwise call std::terminate during unwinding.
    struct Joiner {
      std::atomic<bool>& running;
      std::thread& t;
      ~Joiner() { running = false; if (t.joinable()) t.join(); }
    } joiner{running, refresher};

    MarketDataGateway gw;
    gw.on_update([&](const Ticker& t, const OrderBook& b){ eng.on_book_update(t, b, now_ms()); });
    gw.run(map.watchlist());  // blocks, reconnecting with backoff on drops (Task 20); exits only via gw.stop() (no SIGINT handler wired yet)

    return 0;  // Joiner stops+joins the refresher as this scope exits
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
}
