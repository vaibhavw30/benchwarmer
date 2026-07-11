// Live Kalshi WebSocket client for MarketDataGateway::run().
//
// NOT LIVE-VERIFIED: there are no Kalshi credentials or network access in
// the environment this was implemented in. This file's bar is "compiles and
// links cleanly" — verification against a real Kalshi account (the M1
// integration step, eyeballing the maintained book against Kalshi's site)
// is deferred to the user. Do not run this against real credentials without
// that manual verification step.
#include "market_data/gateway.hpp"
#include "market_data/kalshi_auth.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <openssl/err.h>

// Boost.Beast/Asio pull in a large amount of vendored header code that
// triggers warnings we don't own; narrowly silence those for this include
// block only. Nothing in our own logic below is covered by this pragma.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace te {
namespace {

namespace beast = boost::beast;
namespace websocket = boost::beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

constexpr const char* kHost = "api.elections.kalshi.com";
constexpr const char* kPort = "443";
constexpr const char* kTarget = "/trade-api/ws/v2";

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open private key file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string env_or_throw(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) throw std::runtime_error(std::string("missing required env var: ") + name);
  return std::string(v);
}

long now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

void MarketDataGateway::run(const std::vector<Ticker>& watchlist) {
  // Credentials: KALSHI_KEY_ID + KALSHI_PRIVATE_KEY_PATH (PEM RSA private key).
  const std::string key_id = env_or_throw("KALSHI_KEY_ID");
  const std::string pem = read_file(env_or_throw("KALSHI_PRIVATE_KEY_PATH"));
  KalshiSigner signer(key_id, pem);

  const long ts_ms = now_ms();
  const std::string signature = signer.sign(ws_sign_message(ts_ms));

  net::io_context ioc;
  ssl::context ctx(ssl::context::tlsv12_client);
  ctx.set_default_verify_paths();
  ctx.set_verify_mode(ssl::verify_peer);

  tcp::resolver resolver(ioc);
  websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws(ioc, ctx);

  auto const results = resolver.resolve(kHost, kPort);
  beast::get_lowest_layer(ws).connect(results);

  if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), kHost))
    throw beast::system_error(
        beast::error_code(static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()));

  ws.next_layer().handshake(ssl::stream_base::client);

  ws.set_option(websocket::stream_base::decorator(
      [&](websocket::request_type& req) {
        req.set("KALSHI-ACCESS-KEY", key_id);
        req.set("KALSHI-ACCESS-TIMESTAMP", std::to_string(ts_ms));
        req.set("KALSHI-ACCESS-SIGNATURE", signature);
      }));

  ws.handshake(kHost, kTarget);

  nlohmann::json sub;
  sub["cmd"] = "subscribe";
  nlohmann::json params;
  params["channels"] = {"orderbook_delta"};
  params["market_tickers"] = watchlist;
  sub["params"] = params;
  ws.write(net::buffer(sub.dump()));

  beast::flat_buffer buffer;
  for (;;) {
    buffer.clear();
    ws.read(buffer);
    if (!ws.got_text()) continue;
    handle_raw(std::string_view(static_cast<const char*>(buffer.data().data()), buffer.size()));
  }
}

}  // namespace te
