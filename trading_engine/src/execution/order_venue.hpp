#pragma once
#include <string>
#include "core/types.hpp"
namespace te {
struct Order { Ticker ticker; Side side; Action action; Cents price; int qty; };
struct Fill  { Ticker ticker; Side side; Action action; Cents price; int qty; };
}
