#pragma once
#include "core/types.hpp"
#include "core/config.hpp"
namespace te {
struct Quote { Cents bid; Cents ask; int size; };
Quote make_quote(Cents fair_price, double confidence, int inventory, const Config& c);
}
