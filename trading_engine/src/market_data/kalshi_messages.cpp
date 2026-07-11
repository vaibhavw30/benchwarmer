#include "market_data/kalshi_messages.hpp"

// simdjson's own headers trip -Wdeprecated-literal-operator (space before a
// literal-operator suffix) under clang; that warning originates in the
// vendored library, not this file, so it's silenced locally to keep the
// build clean without touching global warning flags.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#include <simdjson.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace te {
ParsedMsg parse_ws_message(std::string_view json) {
  static thread_local simdjson::ondemand::parser parser;
  simdjson::padded_string buf(json);
  ParsedMsg out;
  auto doc = parser.iterate(buf);
  std::string_view type;
  if (doc["type"].get(type)) return out;
  auto msg = doc["msg"];
  if (type == "orderbook_snapshot") {
    out.kind = MsgKind::Snapshot;
    out.ticker = std::string(std::string_view(msg["market_ticker"]));
    for (auto lvl : msg["yes"].get_array())
      { auto a = lvl.get_array().begin(); Cents p = int64_t(*a); ++a; int q = int64_t(*a);
        out.snapshot.yes.push_back({p,q}); }
    for (auto lvl : msg["no"].get_array())
      { auto a = lvl.get_array().begin(); Cents p = int64_t(*a); ++a; int q = int64_t(*a);
        out.snapshot.no.push_back({p,q}); }
    return out;
  }
  if (type == "orderbook_delta") {
    out.kind = MsgKind::Delta;
    out.ticker = std::string(std::string_view(msg["market_ticker"]));
    out.delta.price = int64_t(msg["price"]);
    out.delta.delta_qty = int64_t(msg["delta"]);
    std::string_view side = msg["side"];
    out.delta.side = (side == "yes") ? Side::Yes : Side::No;
    return out;
  }
  return out; // Other
}
}
