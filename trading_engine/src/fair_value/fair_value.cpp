#include "fair_value/fair_value.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <ctime>
namespace te {
static long iso_to_ms(const std::string& s) {
  std::tm tm{}; // parse "YYYY-MM-DDTHH:MM:SS"
  strptime(s.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
  return (long)timegm(&tm) * 1000L;
}
void FairValueProvider::load_from_file(const std::string& path) {
  std::ifstream f(path); if (!f) return;
  nlohmann::json j; f >> j;
  decltype(map_) fresh;
  for (auto& r : j) {
    FairValue fv{r.at("p_yes").get<double>(), r.at("confidence").get<double>(),
                 iso_to_ms(r.at("asof").get<std::string>())};
    fresh[r.at("ticker").get<std::string>()] = fv;
  }
  map_.swap(fresh);
}
std::optional<FairValue> FairValueProvider::fair_value(const Ticker& t) const {
  auto it = map_.find(t); if (it == map_.end()) return std::nullopt; return it->second;
}
bool FairValueProvider::is_stale(const Ticker& t, long now_ms, int max_age_secs) const {
  auto it = map_.find(t); if (it == map_.end()) return true;
  return (now_ms - it->second.asof_epoch_ms) > (long)max_age_secs * 1000L;
}
}
