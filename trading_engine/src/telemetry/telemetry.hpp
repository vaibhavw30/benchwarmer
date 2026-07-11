#pragma once
#include <ostream>
#include <string>
#include <nlohmann/json.hpp>

namespace te {
class Telemetry {
 public:
  explicit Telemetry(std::ostream& out) : out_(out) {}
  void event(const std::string& type, nlohmann::json fields) {
    fields["type"] = type;
    fields["seq"]  = seq_++;
    out_ << fields.dump() << '\n';
    out_.flush();
  }
 private:
  std::ostream& out_;
  long seq_ = 0;
};
}
