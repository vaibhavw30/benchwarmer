// Offline paper session: replay a recorded WS-frame fixture against PaperVenue
// using a given fair_values.json. Zero real orders — no network, no creds.
//
// Purpose: compare how a fair value (raw vs RECALIBRATE=1) changes the paper
// trader's decisions and realized P&L on identical recorded market data. The
// live engine (te_engine) needs Kalshi credentials and a live slate; this
// driver exercises the exact same StrategyEngine -> RiskManager -> PaperVenue
// -> Telemetry pipeline offline against a fixture.
//
//   usage: paper_session <fair_values.json> <fixture.jsonl> [ticker]
#include <fstream>
#include <iostream>
#include <string>

#include "market_data/gateway.hpp"
#include "strategy/strategy_engine.hpp"

using namespace te;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: paper_session <fair_values.json> <fixture.jsonl> [ticker]\n";
    return 2;
  }
  const std::string fv_path = argv[1];
  const std::string fixture_path = argv[2];
  const Ticker ticker = (argc > 3) ? std::string(argv[3]) : std::string("T");

  // Fixed clock so staleness never depends on wall-clock time (matches
  // test_replay's kFixedNowMs: asof 2026-07-10T00:00:00Z + 60s).
  constexpr long kFixedNowMs = 1783641600000L + 60000L;

  Config c;
  c.base_edge_cents = 2;
  c.fee_cents_per_contract = 1;
  c.confidence_k = 8.0;
  c.max_order_size = 25;
  c.max_contracts_per_market = 100;
  c.max_aggregate_exposure_cents = 500000;
  c.max_daily_loss_cents = 20000;
  c.fair_value_max_age_secs = 1800;

  FairValueProvider fv;
  fv.load_from_file(fv_path);

  RiskManager risk(c);
  PaperVenue venue;
  Telemetry tel(std::cout);
  StrategyEngine eng(c, fv, risk, venue, tel);

  MarketDataGateway gw;
  gw.on_update([&](const Ticker& t, const OrderBook& b) {
    eng.on_book_update(t, b, kFixedNowMs);
  });

  std::ifstream fixture(fixture_path);
  if (!fixture) {
    std::cerr << "cannot open fixture: " << fixture_path << "\n";
    return 1;
  }
  std::string line;
  while (std::getline(fixture, line)) {
    if (!line.empty()) gw.handle_raw(line);
  }

  std::cout << "{\"type\":\"session_end\",\"ticker\":\"" << ticker
            << "\",\"position\":" << venue.position(ticker)
            << ",\"realized_pnl_cents\":" << venue.realized_pnl_cents() << "}\n";
  return 0;
}
