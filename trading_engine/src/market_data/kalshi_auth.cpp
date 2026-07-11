#include "market_data/kalshi_auth.hpp"
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <stdexcept>
#include <vector>
namespace te {
std::string ws_sign_message(long ts_ms) {
  return std::to_string(ts_ms) + "GET" + "/trade-api/ws/v2";
}
static std::string b64(const unsigned char* data, size_t len) {
  BIO* b = BIO_new(BIO_s_mem()); BIO* f = BIO_new(BIO_f_base64());
  BIO_set_flags(f, BIO_FLAGS_BASE64_NO_NL); b = BIO_push(f, b);
  BIO_write(b, data, (int)len); BIO_flush(b);
  BUF_MEM* bptr; BIO_get_mem_ptr(b, &bptr);
  std::string out(bptr->data, bptr->length); BIO_free_all(b); return out;
}
KalshiSigner::KalshiSigner(std::string key_id, std::string pem) : key_id_(std::move(key_id)) {
  BIO* bio = BIO_new_mem_buf(pem.data(), (int)pem.size());
  EVP_PKEY* k = PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr);
  BIO_free(bio);
  if (!k) throw std::runtime_error("failed to parse RSA private key");
  pkey_ = k;
}
KalshiSigner::~KalshiSigner() { if (pkey_) EVP_PKEY_free((EVP_PKEY*)pkey_); }
std::string KalshiSigner::sign(std::string_view msg) const {
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  EVP_PKEY_CTX* pctx = nullptr;
  if (EVP_DigestSignInit(ctx, &pctx, EVP_sha256(), nullptr, (EVP_PKEY*)pkey_) <= 0)
    throw std::runtime_error("DigestSignInit");
  EVP_PKEY_CTX_set_rsa_padding(pctx, RSA_PKCS1_PSS_PADDING);
  EVP_PKEY_CTX_set_rsa_pss_saltlen(pctx, RSA_PSS_SALTLEN_DIGEST); // == digest len (32)
  EVP_PKEY_CTX_set_rsa_mgf1_md(pctx, EVP_sha256());
  if (EVP_DigestSignUpdate(ctx, msg.data(), msg.size()) <= 0)
    throw std::runtime_error("DigestSignUpdate");
  size_t siglen = 0;
  EVP_DigestSignFinal(ctx, nullptr, &siglen);
  std::vector<unsigned char> sig(siglen);
  if (EVP_DigestSignFinal(ctx, sig.data(), &siglen) <= 0)
    throw std::runtime_error("DigestSignFinal");
  EVP_MD_CTX_free(ctx);
  return b64(sig.data(), siglen);
}
}
