#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include "core/types.hpp"
namespace te {
struct FairValue { double p_yes; double confidence; long asof_epoch_ms; };
class FairValueProvider {
 public:
  void load_from_file(const std::string& path);
  std::optional<FairValue> fair_value(const Ticker& t) const;
  bool is_stale(const Ticker& t, long now_ms, int max_age_secs) const;
 private:
  std::unordered_map<Ticker, FairValue> map_;
};
}
