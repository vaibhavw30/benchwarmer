#include <gtest/gtest.h>
#include "market_data/kalshi_auth.hpp"
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
#include <cctype>

// A 2048-bit test private key generated with:
//   openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out test_key.pem
// This is a freshly-generated, throwaway key used only for this unit test.
// It is NOT a Kalshi credential and has no value outside this test file.
static const char* kTestKeyPem = R"(-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC0+j54kR9NlaHx
xHhYfljWeqmkFG/A6NCr3Yqe74mBALYipwPq5IH5T1AHU+mdpzJgyeNC8RkHzAAf
lvlAcFbMa+oPK+qXlY3QjJypmvxftxHwwlHJbCpAZKgS9bL0hG5BS7tCVKPc7/ct
PCuVZ6QKDff055fzzjBOrJZcGJEit9FcVnoQKpc4tp65dZCt2peCixprAk/TWWbw
nlfBCZ7KbPlkoJ9oVkUfuSpV7x5oC3iFIgrTheJyj4p3TYP0NA1FIzq34r8zkNsc
n5iO5tLvPGziz8AGqyNvRvEIwWGEQEMcrBA4uEEQSEmRr/Z3rwDfuVdaneqwosJK
eu+RM/23AgMBAAECggEAMfQO0UWxX79pG2wxEbASQmVKNOWFMTix/HZioFsa17ZE
D82M2cWoEJIxU0x4o9D2KVwgdCZLU+kdsZqBAyXe937j9LsY/i1EHyfoyDHk7mCA
PbXNUG0gPnTqJY2XD0IMks5eCkGFl2LPFbfRieQ5FaNkaT+RpDSqBdVCjXokeXHO
x67chbaQ2WO5DTzd1FOtWUnLLK2ShC5zpos1SS2dlYGEaa+jCmsC+mafZdm+2zHP
H3o8iIvyefokD8HZq/CAvUU9wELwTdVKUtKtj1Xdc4Mw1hfiNCy1GWytkKEg9EP9
XVXx4eRyT1yEAXYDz692JbYf2XnqIYQsbmn5ps1X+QKBgQDzfqtSeywlAk/tO6SN
a6qSZU7nlCjgdi7amE09Q5Wthv9QGYIXqL0gZJqN1KJBrllM/m5fJeXKVN2bbTjv
9dKAAAh2c+n4t7HO3ukRHtFgRi1Ep2k1HVEU3RA2cUPeSjWcyeiBCjtg/TXokWjh
BDRtMA8c4Zfxu0BncyM4XJ6GzwKBgQC+RaIYMqiUMRYK8FdtFVjqlkfx9eGBgpWL
s339DkjNykBtETzKcgKKeEk35mC6fi2R2NzIF1U7Bcp49v2jkh3stph7/8AEtKsd
iHS9HZnKwBbfPzRMoN4DE/YOFybEgr9fx10eNapK9YhCOJvdHBZVp/oF1V/mIWY0
c/oJWI7UmQKBgQC58oHi+y07FgjzohiH5zDbm4ImV37f62DcjnJt1q73VaCkCtbO
Oo4zrqBYr4k5n2uS4Lpo7wgM+8JAb6iLl66pEV1lGCAVUDL5SEG0UVSTsQPg6ffu
F/VTeX5oFRc/KmzGz2o/IRE3gCcq8+Cj9hITUCA6bg0bDWShm8vJvvFRAQKBgHGK
lAPklvx3njPA4CrUBk9WhnA4zey+xAatgY00rPVAr9ll4+Tay/FdfjPBYg9npEHY
K0erxMyH1B8DJLArTXgoLi4wm6EzPrlM6HzB4ThAEGYADXF8vX8QtlAKOLQjYZgC
G+sfExPQGROLPFdhn2JV7rj0b1mgrKC4ZIiXNARhAoGBAPB7XHSLkTrPSbOGooxI
Eq55G+mj+sEjkqH+3IZWrhPK5rts9HoXi1hfXPaULaLJBrQ3NkL9KDDwuVyr90oi
rDmUeFPuVLyTJJ8Iq8vKAp/hPoleMobfBNLQ+8rqqqZLFiTkLnGNMFEkb7+JaLZP
CzfRH4QaJbSns4rtra7VGEwQ
-----END PRIVATE KEY-----)";

TEST(Auth, ProducesNonEmptyBase64Signature) {
  te::KalshiSigner s("test-key-id", kTestKeyPem);
  auto sig = s.sign(te::ws_sign_message(1700000000000L));
  EXPECT_FALSE(sig.empty());
  // base64 chars only
  for (char c : sig) EXPECT_TRUE(isalnum(c) || c=='+' || c=='/' || c=='=');
}

TEST(Auth, WsSignMessageFormat) {
  EXPECT_EQ(te::ws_sign_message(123), "123GET/trade-api/ws/v2");
}
