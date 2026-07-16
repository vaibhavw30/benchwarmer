import base64
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from backend_ml.signal_research.kalshi_auth import KalshiSigner


def _gen_pem():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return key, pem


def _verify(public_key, message: str, b64_sig: str):
    # Raises InvalidSignature if the scheme does not match.
    public_key.verify(
        base64.b64decode(b64_sig),
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_sign_roundtrips_with_public_key():
    key, pem = _gen_pem()
    signer = KalshiSigner("kid-123", pem)
    msg = "1700000000000GET/trade-api/v2/markets/T1"
    b64_sig = signer.sign(msg)
    assert "\n" not in b64_sig
    _verify(key.public_key(), msg, b64_sig)  # no raise == scheme matches


def test_sign_accepts_str_pem():
    key, pem = _gen_pem()
    signer = KalshiSigner("kid", pem.decode())
    _verify(key.public_key(), "m", signer.sign("m"))


def test_headers_wire_key_timestamp_and_signature():
    key, pem = _gen_pem()
    signer = KalshiSigner("kid-abc", pem)
    h = signer.headers("GET", "/trade-api/v2/markets/T1", 1700000000000)
    assert h["KALSHI-ACCESS-KEY"] == "kid-abc"
    assert h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"
    # Signature must verify against ts_ms + method + path.
    _verify(key.public_key(),
            "1700000000000GET/trade-api/v2/markets/T1",
            h["KALSHI-ACCESS-SIGNATURE"])


def test_from_env_reads_key_id_and_pem_file(tmp_path, monkeypatch):
    _, pem = _gen_pem()
    pem_path = tmp_path / "key.pem"
    pem_path.write_bytes(pem)
    monkeypatch.setenv("KALSHI_KEY_ID", "env-kid")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(pem_path))
    signer = KalshiSigner.from_env()
    h = signer.headers("GET", "/p", 1)
    assert h["KALSHI-ACCESS-KEY"] == "env-kid"


def test_from_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("KALSHI_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(RuntimeError):
        KalshiSigner.from_env()
