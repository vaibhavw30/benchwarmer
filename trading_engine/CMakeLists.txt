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

file(GLOB LIB_SRCS src/*/*.cpp)
add_library(te_lib STATIC ${LIB_SRCS})
target_include_directories(te_lib PUBLIC src)
target_link_libraries(te_lib PUBLIC nlohmann_json::nlohmann_json simdjson OpenSSL::Crypto OpenSSL::SSL)

file(GLOB TEST_SRCS tests/*.cpp)
add_executable(te_tests ${TEST_SRCS})
target_link_libraries(te_tests PRIVATE te_core te_lib GTest::gtest_main
  nlohmann_json::nlohmann_json simdjson OpenSSL::Crypto OpenSSL::SSL)
target_include_directories(te_tests PRIVATE src)
include(GoogleTest)
gtest_discover_tests(te_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(te_engine src/main.cpp)
target_link_libraries(te_engine PRIVATE te_lib)
target_include_directories(te_engine PRIVATE src)

# Offline paper-session driver: replay a fixture against PaperVenue with a
# given fair_values.json (raw vs recalibrated). No network, no credentials.
add_executable(paper_session tools/paper_session.cpp)
target_link_libraries(paper_session PRIVATE te_lib)
target_include_directories(paper_session PRIVATE src)
