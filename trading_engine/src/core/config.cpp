#include "core/config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
namespace te {
Config Config::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("config not found: " + path);
  nlohmann::json j; f >> j;
  Config c;
  c.max_contracts_per_market   = j.at("max_contracts_per_market");
  c.max_aggregate_exposure_cents = j.at("max_aggregate_exposure_cents");
  c.max_order_size             = j.at("max_order_size");
  c.max_daily_loss_cents       = j.at("max_daily_loss_cents");
  c.fee_cents_per_contract     = j.at("fee_cents_per_contract");
  c.base_edge_cents            = j.at("base_edge_cents");
  c.confidence_k               = j.at("confidence_k");
  c.fair_value_refresh_secs    = j.at("fair_value_refresh_secs");
  c.fair_value_max_age_secs    = j.at("fair_value_max_age_secs");
  c.orders_per_sec_budget      = j.at("orders_per_sec_budget");
  return c;
}
}
