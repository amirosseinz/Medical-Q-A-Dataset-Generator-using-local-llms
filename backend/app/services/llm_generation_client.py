"""Unified LLM client for Q&A generation — supports Ollama and cloud providers.

Provides a single ``generate_text()`` interface that routes to the correct
provider (Ollama, OpenAI, Anthropic, Google/Gemini, OpenRouter) via the shared
``llm_http.call_provider()`` caller.  Handles retry strategies:
  - Ollama: fast local retry (0s, 2s, 5s + health check)
  - Cloud: exponential backoff with 429-specific extended schedule
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from app.services.llm_http import call_provider

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
BASE_BACKOFF = 2.0

# Ollama-specific retry: fast retries since it's local (no rate limits)
OLLAMA_MAX_RETRIES = 3
OLLAMA_BACKOFF_SCHEDULE = [0.0, 2.0, 5.0]  # instant retry → 2s → 5s + health check

# 429-specific settings: improved progressive backoff (5s → 10s → 20s → 40s → 60s)
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BASE_BACKOFF = 5.0    # start at 5 seconds (was 60s — way too aggressive)
RATE_LIMIT_MAX_BACKOFF = 60.0    # cap at 60 seconds (was 300s)

# Proactive per-provider throttle delays (seconds between requests)
# Prevents hitting rate limits in the first place
PROVIDER_THROTTLE: dict[str, float] = {
    "gemini": 4.5,      # Gemini free tier: 15 RPM → ~4s between requests
    "google": 4.5,
    "openai": 0.5,      # Paid tier typically allows high throughput
    "anthropic": 1.0,
    "openrouter": 1.5,
    "ollama": 0.0,      # Local, no limits
}


# ── Unified dispatch ───────────────────────────────────────────────────


async def generate_text(
    prompt: str,
    provider: str,
    model: str,
    api_key: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    ollama_url: str = "http://host.docker.internal:11434",
) -> str | None:
    """Send a prompt to any supported LLM and return the raw text response.

    Provider-specific retry strategies:
      - Ollama (local): fast retry (0s, 2s, 5s+health check) — no rate limits
      - Cloud providers: exponential backoff with 429-specific extended schedule

    Returns None on total failure.
    """
    # Validate provider is known
    known = {"ollama", "openai", "anthropic", "gemini", "google", "openrouter"}
    if provider not in known:
        logger.error("Unknown generation provider: %s", provider)
        return None

    last_error: Exception | None = None

    # ── Ollama: fast local retry path ──────────────────────────────────
    if provider == "ollama":
        for attempt in range(OLLAMA_MAX_RETRIES):
            try:
                return await call_provider(
                    prompt, provider, model,
                    ollama_url=ollama_url,
                    temperature=temperature, top_p=top_p,
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                body_preview = e.response.text[:300] if e.response.text else "(empty)"
                logger.warning(
                    "Ollama HTTP %d (attempt %d/%d, model=%s): %s",
                    status, attempt + 1, OLLAMA_MAX_RETRIES, model, body_preview,
                )
                if status in (401, 403, 404):
                    logger.error("Ollama error %d is not retryable (model=%s)", status, model)
                    return None
                wait = OLLAMA_BACKOFF_SCHEDULE[min(attempt, len(OLLAMA_BACKOFF_SCHEDULE) - 1)]
                if attempt == OLLAMA_MAX_RETRIES - 1:
                    # Last attempt: do a health check before giving up
                    from app.services.ollama_service import health_check
                    healthy = await health_check(ollama_url, retries=1, delay=2.0)
                    if not healthy:
                        logger.error("Ollama health check failed — server may be down")
                        return None
                if wait > 0:
                    await asyncio.sleep(wait)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(
                    "Ollama connection issue (attempt %d/%d): %s: %s",
                    attempt + 1, OLLAMA_MAX_RETRIES, type(e).__name__, e,
                )
                wait = OLLAMA_BACKOFF_SCHEDULE[min(attempt, len(OLLAMA_BACKOFF_SCHEDULE) - 1)]
                if wait > 0:
                    await asyncio.sleep(wait)
            except Exception as e:
                logger.error("Ollama unexpected error (%s): %s", type(e).__name__, e)
                return None

        logger.error("Ollama generation failed after %d fast retries: %s", OLLAMA_MAX_RETRIES, last_error)
        return None

    # ── Cloud providers: exponential backoff with rate-limit handling ───
    rate_limited = False
    max_attempts = MAX_RETRIES

    # Proactive throttle: delay before first call for cloud providers
    throttle_delay = PROVIDER_THROTTLE.get(provider, 0.0)

    for attempt in range(max(MAX_RETRIES, RATE_LIMIT_MAX_RETRIES)):
        if attempt >= max_attempts:
            break

        try:
            # Proactive throttle for cloud providers to avoid hitting limits
            if throttle_delay > 0 and attempt == 0:
                await asyncio.sleep(throttle_delay)

            if not api_key:
                logger.error("API key required for provider %s", provider)
                return None
            return await call_provider(
                prompt, provider, model,
                api_key=api_key,
                temperature=temperature, top_p=top_p,
            )

        except httpx.HTTPStatusError as e:
            last_error = e
            status = e.response.status_code
            body_preview = e.response.text[:500] if e.response.text else "(empty)"

            if status == 429:
                # ── Rate-limit: switch to long-backoff schedule ──
                if not rate_limited:
                    rate_limited = True
                    max_attempts = RATE_LIMIT_MAX_RETRIES
                    logger.warning(
                        "Rate limited by %s/%s (HTTP 429) — switching to extended retry "
                        "(%d attempts, %ds base). Body: %s",
                        provider, model, RATE_LIMIT_MAX_RETRIES,
                        RATE_LIMIT_BASE_BACKOFF, body_preview[:200],
                    )

                computed_wait = min(
                    RATE_LIMIT_BASE_BACKOFF * (2 ** attempt),
                    RATE_LIMIT_MAX_BACKOFF,
                )
                # Respect Retry-After header if the provider sends one
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        header_wait = float(retry_after)
                        computed_wait = max(computed_wait, header_wait)
                    except (ValueError, TypeError):
                        pass

                # Add ±20% jitter to prevent thundering herd
                jitter = computed_wait * random.uniform(-0.2, 0.2)
                computed_wait = max(1.0, computed_wait + jitter)

                logger.warning(
                    "429 rate-limit (%s/%s) attempt %d/%d. Waiting %.0fs...",
                    provider, model, attempt + 1, max_attempts, computed_wait,
                )
                await asyncio.sleep(computed_wait)

            elif status in (500, 502, 503, 504):
                # Server errors — standard short backoff
                wait = BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Server error %d (%s/%s) attempt %d/%d. Body: %s. Retrying in %.1fs...",
                    status, provider, model, attempt + 1, max_attempts, body_preview[:200], wait,
                )
                await asyncio.sleep(wait)

            elif status in (401, 403):
                # Auth errors — no point retrying
                logger.error(
                    "Auth error %d from %s — check API key. Body: %s",
                    status, provider, body_preview[:200],
                )
                return None

            else:
                # Other HTTP errors — short backoff, limited retries
                wait = BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "HTTP %d (%s/%s) attempt %d/%d. Body: %s. Retrying in %.1fs...",
                    status, provider, model, attempt + 1, max_attempts, body_preview[:200], wait,
                )
                await asyncio.sleep(wait)

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = e
            wait = BASE_BACKOFF * (2 ** attempt)
            logger.warning(
                "Connection issue (%s/%s) attempt %d/%d: %s: %s. Retrying in %.1fs...",
                provider, model, attempt + 1, max_attempts, type(e).__name__, e, wait,
            )
            await asyncio.sleep(wait)

        except Exception as e:
            logger.error("Unexpected error from %s (%s): %s", provider, type(e).__name__, e)
            return None

    logger.error(
        "Generation failed after %d attempts (%s/%s): %s",
        max_attempts, provider, model, last_error,
    )
    return None


# ── Batch generation (replaces ollama_service.generate_qa_batch) ───────

from app.services.ollama_service import _parse_qa_response, OllamaQAPair


async def generate_qa_batch(
    chunks_with_prompts: list[tuple[str, str]],
    provider: str,
    model: str,
    api_key: str = "",
    ollama_url: str = "http://host.docker.internal:11434",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_concurrent: int = 5,
    max_pairs: int | None = None,
    progress_callback=None,
) -> list[OllamaQAPair]:
    """Process multiple chunks concurrently through any supported LLM provider.

    Includes:
      - Automatic retry with a simplified prompt on parse failure
      - Detailed per-prompt failure logging
      - Dynamic concurrency reduction on rate-limit errors
      - Inter-request spacing after failures

    Returns a list of parsed OllamaQAPair objects (name kept for backward compat).
    """
    # Dynamic concurrency: respect rate limits for cloud providers.
    # Ollama uses the full max_concurrent — GPU can handle queued requests.
    provider_throttle = PROVIDER_THROTTLE.get(provider, 0.0)
    if provider_throttle > 2.0:
        # For heavily rate-limited providers (Gemini free), drop concurrency to 2
        effective_concurrency = min(max_concurrent, 2)
    else:
        effective_concurrency = max_concurrent
    semaphore = asyncio.Semaphore(effective_concurrency)
    results: list[OllamaQAPair] = []
    completed = 0
    rate_limit_hits = 0
    parse_failures = 0
    empty_responses = 0
    total = len(chunks_with_prompts)
    # Base inter-request spacing from provider throttle config
    extra_spacing = provider_throttle

    # Simplified retry prompt used when the first attempt fails to parse
    RETRY_PROMPT_SUFFIX = (
        "\n\nIMPORTANT: Respond with EXACTLY this format and nothing else:\n"
        "Question: <your question here>\n"
        "Answer: <your answer here>\n"
        "Do NOT include any preamble, thinking, or explanation outside these two fields."
    )

    async def _process_one(idx: int, chunk_text: str, prompt: str):
        nonlocal completed, rate_limit_hits, extra_spacing, parse_failures, empty_responses
        async with semaphore:
            if max_pairs and len(results) >= max_pairs:
                completed += 1
                return

            # Add spacing between requests when rate-limited
            if extra_spacing > 0:
                await asyncio.sleep(extra_spacing)

            response_text = await generate_text(
                prompt=prompt,
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p,
                ollama_url=ollama_url,
            )

            pair = None
            if response_text:
                pair = _parse_qa_response(response_text, model)
                if pair is None:
                    # ── Retry once with a simplified prompt ──
                    parse_failures += 1
                    logger.info(
                        "Prompt %d/%d: parse failed (response %d chars). Retrying with simplified prompt...",
                        idx + 1, total, len(response_text),
                    )
                    retry_text = await generate_text(
                        prompt=prompt + RETRY_PROMPT_SUFFIX,
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        temperature=max(0.3, temperature - 0.2),  # lower temp for retry
                        top_p=top_p,
                        ollama_url=ollama_url,
                    )
                    if retry_text:
                        pair = _parse_qa_response(retry_text, model)
                        if pair:
                            logger.info("Prompt %d/%d: retry succeeded", idx + 1, total)
                        else:
                            logger.warning(
                                "Prompt %d/%d: retry also failed to parse. "
                                "First 200 chars of retry response: %s",
                                idx + 1, total, retry_text[:200],
                            )
            else:
                # generate_text returned None — likely rate-limited or connection failure
                empty_responses += 1
                rate_limit_hits += 1
                logger.warning(
                    "Prompt %d/%d: generate_text returned None "
                    "(provider=%s, model=%s — likely connection error, timeout, or rate limit)",
                    idx + 1, total, provider, model,
                )
                if rate_limit_hits >= 2 and extra_spacing < 30.0:
                    extra_spacing = min(extra_spacing + 5.0, 30.0)
                    logger.info(
                        "Batch: increased inter-request spacing to %.0fs after %d failures",
                        extra_spacing, rate_limit_hits,
                    )

            completed += 1
            if pair:
                pair.prompt_index = idx
                results.append(pair)
            if progress_callback:
                await progress_callback(completed, total, pair)

    tasks = [_process_one(i, chunk, prompt) for i, (chunk, prompt) in enumerate(chunks_with_prompts)]
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(
        "Batch complete (%s/%s): %d/%d prompts produced Q&A pairs | "
        "%d parse failures | %d empty responses | %d rate-limit hits",
        provider, model, len(results), total,
        parse_failures, empty_responses, rate_limit_hits,
    )
    return results
