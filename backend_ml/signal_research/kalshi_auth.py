"""RSA-PSS request signing for Kalshi REST, ported bit-for-bit from the C++
engine's KalshiSigner (trading_engine/src/market_data/kalshi_auth.cpp).

Pure crypto, no network. The message signed for a REST call is
``f"{ts_ms}{method}{path}"`` (path only, no host/query), signed with RSA-PSS /
SHA-256 / MGF1-SHA-256 / salt length = digest length (32), base64 no-newline.
"""
import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiSigner:
    def __init__(self, key_id: str, private_key_pem):
        self.key_id = key_id
        if isinstance(private_key_pem, str):
            private_key_pem = private_key_pem.encode()
        self._key = serialization.load_pem_private_key(private_key_pem,
                                                       password=None)

    def sign(self, message: str) -> str:
        sig = self._key.sign(
            message.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def headers(self, method: str, path: str, ts_ms: int) -> dict:
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
            "KALSHI-ACCESS-SIGNATURE": self.sign(f"{ts_ms}{method}{path}"),
        }

    @classmethod
    def from_env(cls) -> "KalshiSigner":
        key_id = os.environ.get("KALSHI_KEY_ID")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not key_path:
            raise RuntimeError(
                "KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set to "
                "fetch settlements (see before-live-checklist).")
        pem = Path(key_path).read_bytes()
        return cls(key_id, pem)
