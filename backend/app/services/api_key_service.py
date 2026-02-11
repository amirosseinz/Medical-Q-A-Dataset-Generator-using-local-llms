"""Unified API Key Service — single source of truth for all API key operations.

Provides centralized storage, retrieval, caching, and validation of LLM provider
API keys across ALL features: generation, review, fact-checking, etc.

Usage:
    from app.services.api_key_service import APIKeyService
    service = APIKeyService(db)
    key = service.get_key("openai")          # returns decrypted key or ""
    key = service.get_key("gemini")          # handles google↔gemini alias
    service.set_key("openai", "sk-xxx")      # encrypt + store
    providers = service.list_configured()    # ["openai", "anthropic"]
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from sqlalchemy.orm import Session

from app.models.llm_api_key import LLMApiKey
from app.services.encryption_service import decrypt_value, encrypt_value

logger = logging.getLogger(__name__)

# Provider name aliases — normalize to canonical name for DB lookup
_PROVIDER_ALIASES: dict[str, list[str]] = {
    "openai": ["openai"],
    "anthropic": ["anthropic"],
    "gemini": ["gemini", "google"],
    "google": ["gemini", "google"],
    "openrouter": ["openrouter"],
    "ollama": ["ollama"],
}

# In-memory cache: provider_name -> (decrypted_key, timestamp)
_key_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 300.0  # 5 minutes


def _cache_key(provider: str) -> str:
    """Normalize provider name for cache lookup."""
    return provider.lower().strip()


def clear_key_cache(provider: str | None = None) -> None:
    """Clear cached keys. Pass provider to clear one, or None for all."""
    if provider:
        for alias in _PROVIDER_ALIASES.get(provider.lower(), [provider.lower()]):
            _key_cache.pop(_cache_key(alias), None)
    else:
        _key_cache.clear()


class APIKeyService:
    """Centralized API key management for all LLM features."""

    def __init__(self, db: Session):
        self.db = db

    def _lookup_names(self, provider: str) -> list[str]:
        """Get all possible DB provider_name values for a given provider."""
        return _PROVIDER_ALIASES.get(provider.lower().strip(), [provider.lower().strip()])

    def get_key(
        self,
        provider: str,
        api_key_id: str | None = None,
        direct_key: str | None = None,
    ) -> str:
        """Resolve the best API key for a provider.

        Resolution order:
          1. Direct plaintext key (from request body)
          2. Specific stored key by ID
          3. Default key for provider
          4. Any enabled key for provider

        Returns empty string if no key found (never raises).
        """
        # 1. Direct plaintext key
        if direct_key:
            return direct_key

        # 2. Specific key by ID
        if api_key_id:
            record = (
                self.db.query(LLMApiKey)
                .filter(LLMApiKey.id == api_key_id, LLMApiKey.enabled == True)
                .first()
            )
            if record:
                try:
                    key = decrypt_value(record.api_key_encrypted)
                    logger.debug("Resolved API key by ID %s for provider %s", api_key_id[:8], provider)
                    return key
                except Exception as e:
                    logger.error("Failed to decrypt key %s: %s", api_key_id[:8], e)

        # Check in-memory cache
        cache_id = _cache_key(provider)
        cached = _key_cache.get(cache_id)
        if cached and (time.time() - cached[1]) < _CACHE_TTL:
            return cached[0]

        # 3. Default key for provider (check all aliases)
        names = self._lookup_names(provider)
        default = (
            self.db.query(LLMApiKey)
            .filter(
                LLMApiKey.provider_name.in_(names),
                LLMApiKey.is_default == True,
                LLMApiKey.enabled == True,
            )
            .first()
        )
        if default:
            try:
                key = decrypt_value(default.api_key_encrypted)
                _key_cache[cache_id] = (key, time.time())
                logger.debug("Resolved default API key for %s (provider_name=%s)", provider, default.provider_name)
                return key
            except Exception as e:
                logger.error("Failed to decrypt default key for %s: %s", provider, e)

        # 4. Any enabled key for provider
        any_key = (
            self.db.query(LLMApiKey)
            .filter(
                LLMApiKey.provider_name.in_(names),
                LLMApiKey.enabled == True,
            )
            .first()
        )
        if any_key:
            try:
                key = decrypt_value(any_key.api_key_encrypted)
                _key_cache[cache_id] = (key, time.time())
                logger.debug("Resolved fallback API key for %s (provider_name=%s)", provider, any_key.provider_name)
                return key
            except Exception as e:
                logger.error("Failed to decrypt fallback key for %s: %s", provider, e)

        logger.warning("No API key found for provider '%s' (searched: %s)", provider, names)
        return ""

    def has_key(self, provider: str) -> bool:
        """Check if any enabled key exists for provider without decrypting."""
        names = self._lookup_names(provider)
        count = (
            self.db.query(LLMApiKey.id)
            .filter(
                LLMApiKey.provider_name.in_(names),
                LLMApiKey.enabled == True,
            )
            .count()
        )
        return count > 0

    def get_key_record(self, provider: str) -> Optional[LLMApiKey]:
        """Get the best LLMApiKey record for a provider (for stored_key_id lookup)."""
        names = self._lookup_names(provider)
        # Prefer default, then any
        record = (
            self.db.query(LLMApiKey)
            .filter(
                LLMApiKey.provider_name.in_(names),
                LLMApiKey.is_default == True,
                LLMApiKey.enabled == True,
            )
            .first()
        )
        if record:
            return record
        return (
            self.db.query(LLMApiKey)
            .filter(
                LLMApiKey.provider_name.in_(names),
                LLMApiKey.enabled == True,
            )
            .first()
        )

    def list_configured(self) -> list[str]:
        """List provider names that have at least one enabled key."""
        rows = (
            self.db.query(LLMApiKey.provider_name)
            .filter(LLMApiKey.enabled == True)
            .distinct()
            .all()
        )
        return [r[0] for r in rows]

    def validate_key_exists(self, provider: str) -> tuple[bool, str]:
        """Validate that a key exists and can be decrypted.

        Returns (success, message) tuple.
        """
        if provider.lower() == "ollama":
            return True, "Ollama does not require an API key"

        key = self.get_key(provider)
        if key:
            return True, f"API key for {provider} is configured"
        return False, f"No API key configured for {provider}. Please add it in Settings."
