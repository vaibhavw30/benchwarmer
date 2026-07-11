#pragma once
#include "core/types.hpp"
#include "core/config.hpp"
namespace te {
Cents fair_price_cents(double p_yes);
int edge_threshold_cents(double confidence, const Config& c);
}
