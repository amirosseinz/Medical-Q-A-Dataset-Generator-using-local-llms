"""Singleton HTTP client manager with connection pooling.

Provides a single get_http_client() interface used by both
llm_generation_client.py and llm_review_service.py, eliminating
duplicate client management patterns.

Key features:
  - Event-loop-aware client lifecycle (recreates when Celery spins a new loop)
  - Provider-specific timeout and connection-pool configuration
  - Graceful shutdown via close_all_clients()
"""
from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

# ── Provider-specific timeout configuration ────────────────────────────

_PROVIDER_TIMEOUTS: dict[str, httpx.Timeout] = {
    "ollama": httpx.Timeout(180.0, connect=15.0),
    "openai": httpx.Timeout(120.0, connect=15.0),
    "anthropic": httpx.Timeout(120.0, connect=15.0),
    "gemini": httpx.Timeout(120.0, connect=15.0),
    "google": httpx.Timeout(120.0, connect=15.0),
    "openrouter": httpx.Timeout(120.0, connect=15.0),
}

_DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=15.0)

_CONNECTION_LIMITS = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=10,
    keepalive_expiry=120,
)

# ── Client pool (module-level singletons) ──────────────────────────────

_clients: dict[str, httpx.AsyncClient] = {}
_client_loop_ids: dict[str, int] = {}


def get_http_client(provider: str) -> httpx.AsyncClient:
    """Get or create an httpx AsyncClient for *provider*.

    Automatically recreates the client when the event loop changes
    (happens each time Celery starts a new task).
    """
    loop_id = id(asyncio.get_event_loop())

    if (
        provider not in _clients
        or _clients[provider].is_closed
        or _client_loop_ids.get(provider) != loop_id
    ):
        timeout = _PROVIDER_TIMEOUTS.get(provider, _DEFAULT_TIMEOUT)
        _clients[provider] = httpx.AsyncClient(
            timeout=timeout,
            limits=_CONNECTION_LIMITS,
        )
        _client_loop_ids[provider] = loop_id
        logger.debug("Created new HTTP client for provider '%s'", provider)

    return _clients[provider]


async def close_all_clients() -> None:
    """Close every pooled HTTP client (for graceful shutdown)."""
    for name, client in list(_clients.items()):
        if not client.is_closed:
            try:
                await client.aclose()
            except Exception:
                pass
    _clients.clear()
    _client_loop_ids.clear()
    logger.info("All HTTP clients closed")
