#pragma once
#include <string>
namespace te {
struct Config {
  int max_contracts_per_market{};
  int max_aggregate_exposure_cents{};
  int max_order_size{};
  int max_daily_loss_cents{};
  int fee_cents_per_contract{};
  int base_edge_cents{};
  double confidence_k{};
  int fair_value_refresh_secs{};
  int fair_value_max_age_secs{};
  int orders_per_sec_budget{};
  static Config load(const std::string& path);
};
}
