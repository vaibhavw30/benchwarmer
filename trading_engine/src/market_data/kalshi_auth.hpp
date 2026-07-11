#pragma once
#include <string>
#include <string_view>
namespace te {
std::string ws_sign_message(long ts_ms);
class KalshiSigner {
 public:
  KalshiSigner(std::string key_id, std::string pem_private_key);
  ~KalshiSigner();
  std::string key_id() const { return key_id_; }
  std::string sign(std::string_view message) const; // base64(RSA-PSS SHA256)
 private:
  std::string key_id_;
  void* pkey_ = nullptr; // EVP_PKEY*
};
}
