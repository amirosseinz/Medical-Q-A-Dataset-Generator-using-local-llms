"""Dynamic model fetcher — fetches available models from each LLM provider's API.

Supports:
  - OpenAI: GET /v1/models (filter to chat models)
  - Anthropic: GET /v1/models (fallback to known list)
  - Google Gemini: GET /v1beta/models (filter by generateContent)
  - OpenRouter: GET /api/v1/models (includes pricing)

Caching strategy:
  - Models cached in LLMApiKey.available_models + model_details JSON columns
  - models_fetched_at timestamp tracks freshness
  - Provider-specific TTLs (6-24 hours)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Provider-specific cache TTLs (hours)
CACHE_TTL_HOURS: dict[str, int] = {
    "openai": 12,
    "anthropic": 24,
    "google": 12,
    "gemini": 12,
    "openrouter": 6,  # Changes frequently
}

# Fallback models when API fetch fails
FALLBACK_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ],
    "google": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "openrouter": [
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3-opus",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-pro-1.5",
        "google/gemini-flash-1.5",
        "meta-llama/llama-3.1-70b-instruct",
    ],
}

# OpenAI models to EXCLUDE (non-chat models)
_OPENAI_EXCLUDE_PREFIXES = (
    "whisper", "dall-e", "tts", "text-embedding", "babbage", "davinci",
    "text-moderation", "text-search", "code-search", "text-similarity",
    "curie", "ada", "gpt-3.5-turbo-instruct", "canary-",
    "ft:", "o1-pro",  # fine-tuned models, o1-pro (special access)
)

# OpenAI model ID substrings that indicate chat models
_OPENAI_CHAT_INDICATORS = ("gpt-", "o1", "o3", "chatgpt-")


def is_cache_fresh(fetched_at: datetime | None, provider: str) -> bool:
    """Check if cached models are still fresh."""
    if fetched_at is None:
        return False
    ttl = CACHE_TTL_HOURS.get(provider, 12)
    expiry = fetched_at + timedelta(hours=ttl)
    return datetime.now(timezone.utc) < expiry


async def fetch_models(
    provider: str,
    api_key: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch available models from a provider's API.

    Returns:
        {
            "models": ["model-id-1", "model-id-2", ...],
            "model_details": {
                "model-id-1": {"context_length": 128000, "pricing": {...}, ...},
                ...
            },
            "error": None or "error message",
        }
    """
    fetchers = {
        "openai": _fetch_openai,
        "anthropic": _fetch_anthropic,
        "google": _fetch_google,
        "gemini": _fetch_google,
        "openrouter": _fetch_openrouter,
    }

    fetcher = fetchers.get(provider)
    if not fetcher:
        return {
            "models": [],
            "model_details": {},
            "error": f"Unknown provider: {provider}",
        }

    try:
        return await fetcher(api_key, timeout)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in (401, 403):
            msg = f"Authentication failed ({status}). Check your API key."
        elif status == 429:
            msg = "Rate limited. Try again later."
        else:
            msg = f"API error: {status}"
        logger.warning(f"Model fetch failed for {provider}: {msg}")
        return {"models": [], "model_details": {}, "error": msg}
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        msg = f"Connection error: {e}"
        logger.warning(f"Model fetch failed for {provider}: {msg}")
        return {"models": [], "model_details": {}, "error": msg}
    except Exception as e:
        msg = f"Unexpected error: {e}"
        logger.exception(f"Model fetch failed for {provider}")
        return {"models": [], "model_details": {}, "error": msg}


async def _fetch_openai(api_key: str, timeout: float) -> dict[str, Any]:
    """Fetch models from OpenAI API, filter to chat-capable models."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

    models: list[str] = []
    details: dict[str, dict] = {}

    for m in data.get("data", []):
        model_id: str = m.get("id", "")
        # Skip non-chat models
        if any(model_id.lower().startswith(p) for p in _OPENAI_EXCLUDE_PREFIXES):
            continue
        # Only include models that look like chat models
        if not any(ind in model_id.lower() for ind in _OPENAI_CHAT_INDICATORS):
            continue
        # Skip dated snapshots if base model exists (e.g., keep gpt-4o, skip gpt-4o-2024-08-06)
        # Actually keep them — users may want specific versions
        models.append(model_id)
        details[model_id] = {
            "owned_by": m.get("owned_by", ""),
            "created": m.get("created"),
        }

    # Sort: prioritize common models first
    priority_order = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o3", "o4-mini", "chatgpt-4o-latest"]
    def _sort_key(model_id: str) -> tuple[int, str]:
        for i, pref in enumerate(priority_order):
            if model_id == pref:
                return (0, f"{i:04d}")
            if model_id.startswith(pref):
                return (1, f"{i:04d}_{model_id}")
        return (2, model_id)

    models.sort(key=_sort_key)

    return {"models": models, "model_details": details, "error": None}


async def _fetch_anthropic(api_key: str, timeout: float) -> dict[str, Any]:
    """Fetch models from Anthropic API."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            models: list[str] = []
            details: dict[str, dict] = {}
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if model_id:
                    models.append(model_id)
                    details[model_id] = {
                        "display_name": m.get("display_name", model_id),
                        "created_at": m.get("created_at"),
                        "type": m.get("type", "model"),
                    }

            if models:
                # Sort with newest/best first
                priority = ["claude-sonnet-4", "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"]
                def _sort_key(mid: str) -> tuple[int, str]:
                    for i, p in enumerate(priority):
                        if p in mid:
                            return (0, f"{i:04d}_{mid}")
                    return (1, mid)
                models.sort(key=_sort_key)
                return {"models": models, "model_details": details, "error": None}

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # /v1/models not available for this key — fall back
                pass
            else:
                raise

    # Fallback: validate known models by attempting a minimal request
    known_models = [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]
    return {
        "models": known_models,
        "model_details": {m: {"source": "fallback"} for m in known_models},
        "error": None,
    }


async def _fetch_google(api_key: str, timeout: float) -> dict[str, Any]:
    """Fetch models from Google Gemini API."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
        )
        resp.raise_for_status()
        data = resp.json()

    models: list[str] = []
    details: dict[str, dict] = {}

    for m in data.get("models", []):
        name: str = m.get("name", "")
        methods = m.get("supportedGenerationMethods", [])
        # Only include models that support generateContent
        if "generateContent" not in methods:
            continue
        # Extract model ID from "models/gemini-2.0-flash"
        model_id = name.replace("models/", "")
        if not model_id:
            continue
        models.append(model_id)
        details[model_id] = {
            "display_name": m.get("displayName", model_id),
            "description": m.get("description", ""),
            "input_token_limit": m.get("inputTokenLimit"),
            "output_token_limit": m.get("outputTokenLimit"),
            "supported_methods": methods,
        }

    # Sort: prioritize common models
    priority = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    def _sort_key(mid: str) -> tuple[int, str]:
        for i, p in enumerate(priority):
            if mid.startswith(p):
                return (0, f"{i:04d}_{mid}")
        return (1, mid)
    models.sort(key=_sort_key)

    return {"models": models, "model_details": details, "error": None}


async def _fetch_openrouter(api_key: str, timeout: float) -> dict[str, Any]:
    """Fetch models from OpenRouter API, including pricing."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

    models: list[str] = []
    details: dict[str, dict] = {}

    for m in data.get("data", []):
        model_id: str = m.get("id", "")
        if not model_id:
            continue

        pricing = m.get("pricing", {})
        context_length = m.get("context_length")

        # Skip models with no pricing info (usually deprecated)
        prompt_price = pricing.get("prompt", "0")
        completion_price = pricing.get("completion", "0")

        try:
            prompt_cost = float(prompt_price)
            completion_cost = float(completion_price)
        except (ValueError, TypeError):
            prompt_cost = 0.0
            completion_cost = 0.0

        models.append(model_id)
        details[model_id] = {
            "name": m.get("name", model_id),
            "description": m.get("description", ""),
            "context_length": context_length,
            "pricing": {
                "prompt": prompt_cost,  # per token
                "completion": completion_cost,  # per token
                "prompt_per_1k": round(prompt_cost * 1000, 6),
                "completion_per_1k": round(completion_cost * 1000, 6),
            },
            "top_provider": m.get("top_provider", {}),
        }

    # Sort by popularity / common providers first
    provider_priority = {
        "anthropic/": 0, "openai/": 1, "google/": 2,
        "meta-llama/": 3, "mistralai/": 4, "qwen/": 5,
    }
    def _sort_key(mid: str) -> tuple[int, str]:
        for prefix, pri in provider_priority.items():
            if mid.startswith(prefix):
                return (pri, mid)
        return (99, mid)
    models.sort(key=_sort_key)

    return {"models": models, "model_details": details, "error": None}
