#include "market_data/gateway.hpp"
#include "market_data/kalshi_messages.hpp"
namespace te {
void MarketDataGateway::handle_raw(std::string_view json) {
  ParsedMsg m = parse_ws_message(json);
  if (m.kind == MsgKind::Snapshot) {
    books_[m.ticker].apply_snapshot(m.snapshot);
  } else if (m.kind == MsgKind::Delta) {
    books_[m.ticker].apply_delta(m.delta);
  } else return;
  if (cb_) cb_(m.ticker, books_[m.ticker]);
}
}
