"""Rate limit handler with exponential backoff and smart request spacing."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import httpx

logger = logging.getLogger(__name__)

# Provider-specific concurrent request limits
PROVIDER_CONCURRENCY: dict[str, dict[str, int]] = {
    "openai": {
        "gpt-4o": 5, "gpt-4o-mini": 10, "gpt-4-turbo": 5,
        "gpt-3.5-turbo": 10, "default": 5,
    },
    "anthropic": {"default": 5},
    "gemini": {"default": 10},
    "openrouter": {"default": 5},
    "ollama": {"default": 3},
}

# Provider-specific minimum spacing between requests (seconds)
PROVIDER_SPACING: dict[str, dict[str, float]] = {
    "openai": {
        "gpt-4o": 0.5, "gpt-4o-mini": 0.2, "gpt-4-turbo": 0.5,
        "gpt-3.5-turbo": 0.1, "default": 0.3,
    },
    "anthropic": {"default": 0.4},
    "gemini": {"default": 0.2},
    "openrouter": {"default": 0.3},
    "ollama": {"default": 0.0},
}

# Approximate cost per 1K tokens (input, output) — for estimation
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # OpenRouter models (provider/model format)
    "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic/claude-3.5-haiku": {"input": 0.001, "output": 0.005},
    "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},
    "openai/gpt-4o": {"input": 0.005, "output": 0.015},
    "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "openai/gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "google/gemini-pro-1.5": {"input": 0.00125, "output": 0.005},
    "google/gemini-flash-1.5": {"input": 0.000075, "output": 0.0003},
    "meta-llama/llama-3.1-405b-instruct": {"input": 0.003, "output": 0.003},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.00035, "output": 0.0004},
    "mistralai/mistral-large": {"input": 0.003, "output": 0.009},
    "mistralai/mixtral-8x22b-instruct": {"input": 0.00065, "output": 0.00065},
    "qwen/qwen-2.5-72b-instruct": {"input": 0.0004, "output": 0.0004},
}


def get_concurrency_limit(provider: str, model: str = "") -> int:
    limits = PROVIDER_CONCURRENCY.get(provider, {"default": 3})
    return limits.get(model, limits.get("default", 3))


def get_request_spacing(provider: str, model: str = "") -> float:
    spacing = PROVIDER_SPACING.get(provider, {"default": 0.3})
    return spacing.get(model, spacing.get("default", 0.3))


def estimate_cost(model: str, pair_count: int, avg_tokens_per_pair: int = 500) -> float:
    """Estimate review cost in USD."""
    costs = MODEL_COSTS.get(model)
    if not costs:
        return 0.0
    total_input_tokens = pair_count * avg_tokens_per_pair
    total_output_tokens = pair_count * 150  # ~150 output tokens per review
    cost = (total_input_tokens / 1000 * costs["input"]) + (
        total_output_tokens / 1000 * costs["output"]
    )
    return round(cost, 4)


def recommend_model(
    pair_count: int,
    available_providers: list[dict],
    budget_usd: float | None = None,
) -> dict[str, Any]:
    """Recommend best model based on task size and budget."""
    candidates = []
    for provider in available_providers:
        for model_name in provider.get("models", []):
            costs = MODEL_COSTS.get(model_name)
            if not costs:
                continue
            est_cost = estimate_cost(model_name, pair_count)
            quality = 7  # default
            if "gpt-4" in model_name or "opus" in model_name:
                quality = 9.5
            elif "sonnet" in model_name:
                quality = 9
            elif "haiku" in model_name or "flash" in model_name:
                quality = 8
            elif "3.5" in model_name:
                quality = 7
            candidates.append({
                "provider": provider.get("name", ""),
                "model": model_name,
                "estimated_cost": est_cost,
                "quality_score": quality,
            })

    if budget_usd is not None:
        candidates = [c for c in candidates if c["estimated_cost"] <= budget_usd]

    if not candidates:
        return {"provider": "ollama", "model": "llama3", "estimated_cost": 0, "quality_score": 6}

    if pair_count < 50:
        return max(candidates, key=lambda c: c["quality_score"])
    else:
        return max(
            candidates,
            key=lambda c: c["quality_score"] / max(c["estimated_cost"] + 0.01, 0.01),
        )


@dataclass
class RateLimitState:
    """Tracks rate limit state for a review session."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_count: int = 0
    total_wait_seconds: float = 0.0
    current_retry_attempt: int = 0
    last_request_time: float = 0.0
    is_rate_limited: bool = False
    current_message: str = ""


class RateLimitHandler:
    """Handles rate limiting with exponential backoff for LLM API calls.

    For 429 errors, starts with a 60-second wait and doubles each retry up to
    the max_backoff ceiling. The Retry-After header is respected if present and
    will override the computed wait time (using the larger of the two).
    """

    def __init__(
        self,
        provider: str,
        model: str = "",
        max_retries: int = 8,
        max_backoff: float = 300.0,
        base_rate_limit_wait: float = 60.0,
        speed: str = "normal",
    ):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self.base_rate_limit_wait = base_rate_limit_wait
        self.state = RateLimitState()

        base_spacing = get_request_spacing(provider, model)
        if speed == "slow":
            self._spacing = base_spacing * 2
        elif speed == "fast":
            self._spacing = base_spacing * 0.5
        else:
            self._spacing = base_spacing

    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an async function with rate limit retry logic."""
        # Respect request spacing
        now = time.monotonic()
        elapsed = now - self.state.last_request_time
        if elapsed < self._spacing and self.state.last_request_time > 0:
            await asyncio.sleep(self._spacing - elapsed)

        self.state.last_request_time = time.monotonic()
        self.state.total_requests += 1

        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                self.state.successful_requests += 1
                self.state.current_retry_attempt = 0
                self.state.is_rate_limited = False
                return result

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    self.state.rate_limited_count += 1
                    self.state.is_rate_limited = True
                    self.state.current_retry_attempt = attempt + 1

                    if attempt >= self.max_retries:
                        self.state.failed_requests += 1
                        raise RateLimitExceeded(
                            f"Rate limit persists after {self.max_retries} retries for {self.provider}"
                        ) from e

                    # Compute wait time: start at base_rate_limit_wait (60s) and double
                    computed_wait = min(
                        self.base_rate_limit_wait * (2 ** attempt),
                        self.max_backoff,
                    )

                    # Parse Retry-After header (may be seconds or HTTP date)
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            header_wait = float(retry_after)
                            # Use the larger of computed and header value
                            computed_wait = max(computed_wait, header_wait)
                        except ValueError:
                            pass  # Could be an HTTP date; ignore and use computed

                    self.state.total_wait_seconds += computed_wait
                    self.state.current_message = (
                        f"Rate limited by {self.provider}. "
                        f"Retrying in {computed_wait:.0f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    logger.warning(
                        f"Rate limited by {self.provider} (attempt {attempt + 1}/{self.max_retries}), "
                        f"waiting {computed_wait:.1f}s"
                    )
                    await asyncio.sleep(computed_wait)
                    # Increase spacing after rate limit to reduce future hits
                    self._spacing = min(self._spacing * 1.5, 5.0)
                    continue

                elif e.response.status_code in (401, 403):
                    self.state.failed_requests += 1
                    raise AuthenticationError(
                        f"Authentication failed for {self.provider}: {e.response.status_code}"
                    ) from e

                else:
                    self.state.failed_requests += 1
                    raise

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt >= self.max_retries:
                    self.state.failed_requests += 1
                    raise
                wait_time = min(2 ** attempt, self.max_backoff)
                logger.warning(f"Connection error for {self.provider}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

            except Exception:
                self.state.failed_requests += 1
                raise

        self.state.failed_requests += 1
        raise RateLimitExceeded(f"Max retries exceeded for {self.provider}")


class RateLimitExceeded(Exception):
    pass


class AuthenticationError(Exception):
    pass
