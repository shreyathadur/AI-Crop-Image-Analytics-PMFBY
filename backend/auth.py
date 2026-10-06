"""Password hashing and signed bearer-token helpers using the Python standard library."""
import base64
import hashlib
import hmac
import json
import time

from backend.config import SECRET_KEY


def hash_password(password: str) -> str:
    salt = __import__("secrets").token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"pbkdf2_sha256${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def verify_password(password, stored):
    try:
        _, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.urlsafe_b64decode(salt), 310_000)
        return hmac.compare_digest(base64.urlsafe_b64encode(digest).decode(), expected)
    except (ValueError, TypeError):
        return False


def issue_token(user_id):
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY must be configured before issuing tokens")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": user_id, "exp": int(time.time()) + 86400}).encode()).decode().rstrip("=")
    signature = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    return payload + "." + base64.urlsafe_b64encode(signature).decode().rstrip("=")


def verify_token(token):
    if not SECRET_KEY:
        return None
    try:
        payload, signature = token.split(".")
        expected = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))):
            return None
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return claims["sub"] if claims["exp"] > time.time() else None
    except (ValueError, KeyError, TypeError):
        return None
