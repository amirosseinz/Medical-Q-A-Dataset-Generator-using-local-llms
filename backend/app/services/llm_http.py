"""Unified HTTP callers for all supported LLM providers.

Provides a single ``call_provider()`` interface that routes to the correct
provider API endpoint. Both ``llm_generation_client`` and ``llm_review_service``
delegate their raw HTTP calls here — eliminating duplicate provider functions.

No retry logic lives here; callers handle their own retry strategies.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.services.http_client_manager import get_http_client

logger = logging.getLogger(__name__)


def _get_client(provider: str) -> httpx.AsyncClient:
    """Get an httpx client for the given provider via the shared manager."""
    return get_http_client(provider)


# ── Provider-specific HTTP callers ─────────────────────────────────────

async def _call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Generate text via local Ollama."""
    client = _get_client("ollama")
    resp = await client.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        },
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


async def _call_openai(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Generate text via OpenAI API."""
    client = _get_client("openai")
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _call_anthropic(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Generate text via Anthropic API."""
    client = _get_client("anthropic")
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
        },
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


async def _call_gemini(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Generate text via Google Gemini API."""
    client = _get_client("gemini")
    resp = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
            },
        },
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


async def _call_openrouter(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Generate text via OpenRouter API."""
    client = _get_client("openrouter")
    resp = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://qa-dataset-generator.local",
            "X-Title": "Medical Q&A Dataset Generator",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Provider registry ──────────────────────────────────────────────────

_CLOUD_CALLERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "google": _call_gemini,
    "openrouter": _call_openrouter,
}

# Default models per provider — used when no model is specified
PROVIDER_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "openrouter": [
        "anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-haiku",
        "anthropic/claude-3-opus", "openai/gpt-4o", "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo", "google/gemini-pro-1.5", "google/gemini-flash-1.5",
        "meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-3.1-70b-instruct",
        "mistralai/mistral-large", "mistralai/mixtral-8x22b-instruct",
        "qwen/qwen-2.5-72b-instruct",
    ],
    "ollama": [],
}


async def call_provider(
    prompt: str,
    provider: str,
    model: str,
    api_key: str = "",
    ollama_url: str = "http://host.docker.internal:11434",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1024,
) -> str:
    """Send a prompt to any supported LLM provider and return raw text.

    Raises on HTTP errors — caller is responsible for retry logic.

    Parameters
    ----------
    prompt : the full prompt text
    provider : one of "ollama", "openai", "anthropic", "gemini", "google", "openrouter"
    model : model name/ID
    api_key : required for cloud providers
    ollama_url : Ollama server URL (only used when provider=="ollama")
    temperature : sampling temperature
    top_p : nucleus sampling threshold
    max_tokens : maximum output tokens

    Returns
    -------
    str : the raw LLM response text

    Raises
    ------
    ValueError : unknown provider or missing API key
    httpx.HTTPStatusError : HTTP errors from provider (4xx, 5xx)
    httpx.ConnectError : connection errors
    httpx.TimeoutException : timeout errors
    """
    if provider == "ollama":
        return await _call_ollama(prompt, model, ollama_url, temperature, top_p, max_tokens)

    caller = _CLOUD_CALLERS.get(provider)
    if caller is None:
        raise ValueError(f"Unknown LLM provider: {provider}")
    if not api_key:
        raise ValueError(f"API key required for provider {provider}")

    return await caller(prompt, api_key, model, temperature, top_p, max_tokens)
