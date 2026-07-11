#pragma once
#include <string_view>
#include "market_data/order_book.hpp"
namespace te {
enum class MsgKind { Snapshot, Delta, Other };
struct ParsedMsg { MsgKind kind{MsgKind::Other}; Ticker ticker; BookSnapshot snapshot; BookDelta delta{}; };
ParsedMsg parse_ws_message(std::string_view json);
}
