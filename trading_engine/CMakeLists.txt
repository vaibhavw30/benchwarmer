cmake_minimum_required(VERSION 3.20)
project(trading_engine CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.15.2)
FetchContent_MakeAvailable(googletest)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)
FetchContent_Declare(simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git GIT_TAG v3.10.1)
FetchContent_MakeAvailable(simdjson)
find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED)   # header-only Beast; system Boost ok

enable_testing()
add_library(te_core INTERFACE)
target_include_directories(te_core INTERFACE src)

file(GLOB TEST_SRCS tests/*.cpp)
add_executable(te_tests ${TEST_SRCS})
target_link_libraries(te_tests PRIVATE te_core GTest::gtest_main
  nlohmann_json::nlohmann_json simdjson OpenSSL::Crypto OpenSSL::SSL)
target_include_directories(te_tests PRIVATE src)
include(GoogleTest)
gtest_discover_tests(te_tests)
