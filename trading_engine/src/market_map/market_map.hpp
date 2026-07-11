#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/types.hpp"
namespace te {
struct GameRef { int home_team_id; int away_team_id; std::string game_date; };
class MarketMap {
 public:
  void load(const std::string& path);
  std::vector<Ticker> watchlist() const;
  std::optional<GameRef> game_for(const Ticker& t) const;
 private:
  std::unordered_map<Ticker, GameRef> map_;
};
}
