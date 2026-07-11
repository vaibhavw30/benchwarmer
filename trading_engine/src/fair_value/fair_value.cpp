#include "fair_value/fair_value.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <ctime>
namespace te {
// Parse "YYYY-MM-DDTHH:MM:SS" to epoch ms. Returns false if the string fails
// to parse (so the caller can skip the row rather than insert a bogus 0-epoch).
static bool iso_to_ms(const std::string& s, long& out_ms) {
  std::tm tm{};
  if (strptime(s.c_str(), "%Y-%m-%dT%H:%M:%S", &tm) == nullptr) return false;
  out_ms = (long)timegm(&tm) * 1000L;
  return true;
}
void FairValueProvider::load_from_file(const std::string& path) {
  std::ifstream f(path); if (!f) return;  // missing file: leave map untouched
  decltype(map_) fresh;
  try {
    nlohmann::json j; f >> j;
    for (auto& r : j) {
      long asof_ms;
      if (!iso_to_ms(r.at("asof").get<std::string>(), asof_ms)) continue;  // skip unparseable timestamp
      FairValue fv{r.at("p_yes").get<double>(), r.at("confidence").get<double>(), asof_ms};
      fresh[r.at("ticker").get<std::string>()] = fv;
    }
  } catch (const std::exception&) {
    return;  // malformed/partial JSON: retain existing map, don't clobber
  }
  // Parsing done above WITHOUT the lock; only the swap is guarded so concurrent
  // readers (gw.run() loop) never observe a half-mutated map.
  std::lock_guard<std::mutex> lk(mtx_);
  map_.swap(fresh);
}
std::optional<FairValue> FairValueProvider::fair_value(const Ticker& t) const {
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = map_.find(t); if (it == map_.end()) return std::nullopt; return it->second;
}
bool FairValueProvider::is_stale(const Ticker& t, long now_ms, int max_age_secs) const {
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = map_.find(t); if (it == map_.end()) return true;
  return (now_ms - it->second.asof_epoch_ms) > (long)max_age_secs * 1000L;
}
}
