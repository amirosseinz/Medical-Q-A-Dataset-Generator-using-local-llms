"""Encryption service for secure API key storage using Fernet symmetric encryption."""
from __future__ import annotations

import base64
import hashlib
import logging

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings

logger = logging.getLogger(__name__)

_cipher = None


def _get_cipher() -> Fernet:
    global _cipher
    if _cipher is None:
        settings = get_settings()
        # Derive a 32-byte key from SECRET_KEY using SHA-256
        key_bytes = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        _cipher = Fernet(fernet_key)
    return _cipher


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string value. Returns base64-encoded encrypted string."""
    cipher = _get_cipher()
    return cipher.encrypt(plaintext.encode()).decode()


def decrypt_value(encrypted: str) -> str:
    """Decrypt an encrypted string value."""
    cipher = _get_cipher()
    try:
        return cipher.decrypt(encrypted.encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt value — invalid token or corrupted data")
        raise ValueError("Decryption failed: invalid key or corrupted data")


def mask_api_key(key: str) -> str:
    """Return a masked version of an API key for display purposes."""
    if len(key) <= 8:
        return "••••••••"
    return f"{'•' * (len(key) - 4)}{key[-4:]}"
