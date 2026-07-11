#include <gtest/gtest.h>
#include <sstream>
#include <algorithm>
#include "telemetry/telemetry.hpp"

TEST(Telemetry, WritesOneJsonLinePerEvent) {
  std::ostringstream os;
  te::Telemetry t(os);
  t.event("quote", {{"ticker","KXNBA-XYZ"},{"bid",54},{"ask",56}});
  t.event("fill",  {{"ticker","KXNBA-XYZ"},{"price",55}});
  auto s = os.str();
  EXPECT_EQ(std::count(s.begin(), s.end(), '\n'), 2);
  EXPECT_NE(s.find("\"type\":\"quote\""), std::string::npos);
  EXPECT_NE(s.find("\"seq\":0"), std::string::npos);
}
