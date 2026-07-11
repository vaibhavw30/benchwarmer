#include "market_map/market_map.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
namespace te {
void MarketMap::load(const std::string& path) {
  std::ifstream f(path); if (!f) throw std::runtime_error("watchlist not found: " + path);
  nlohmann::json j; f >> j;
  for (auto& r : j)
    map_[r.at("ticker")] = GameRef{r.at("home_team_id"), r.at("away_team_id"), r.at("game_date")};
}
std::vector<Ticker> MarketMap::watchlist() const {
  std::vector<Ticker> v; v.reserve(map_.size());
  for (auto& [k,_] : map_) v.push_back(k); return v;
}
std::optional<GameRef> MarketMap::game_for(const Ticker& t) const {
  auto it = map_.find(t); if (it == map_.end()) return std::nullopt; return it->second;
}
}
