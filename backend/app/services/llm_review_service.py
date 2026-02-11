"""LLM Review Service — uses external LLM APIs to review Q&A pairs.

Architecture
------------
  - Provider-agnostic structured output via prompt discipline + output sanitization
  - Deterministic balanced-brace JSON extraction (no regex for structure)
  - Universal ``<think>`` / markdown stripping via ``json_utils``
  - Single-retry on parse failure with strict correction prompt
  - Strict schema validation with transparent violation logging
  - GPU memory cleanup after batch operations
  - Preserved: provider HTTP isolation, RateLimitHandler, concurrency control

All robustness is implemented via:
  - Prompt discipline
  - Output sanitization
  - Deterministic JSON extraction
  - Schema validation
  - Controlled retry logic
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services.rate_limit_handler import (
    RateLimitHandler,
    get_concurrency_limit,
    RateLimitExceeded,
    AuthenticationError,
)
from app.utils.json_utils import (
    safe_parse_json,
    validate_review_schema,
    validate_fact_check_schema,
    coerce_review,
    coerce_fact_check,
)
from app.utils.gpu_cleanup import release_gpu_memory
from app.services.llm_http import call_provider

logger = logging.getLogger(__name__)


# ── Prompts ────────────────────────────────────────────────────────────
# Designed for maximum parse reliability across all providers and model
# families, including reasoning/thinking models.

_SYSTEM_CONSTRAINT = (
    "You are a data validation engine. "
    "You do not explain. You do not reason step-by-step. "
    "You do not add commentary. You do not use markdown. "
    "You do not wrap output in code fences. "
    "You do not use <think> tags. "
    "You output ONLY a single valid JSON object. "
    "Any text outside the JSON object invalidates your response."
)

REVIEW_PROMPT = """{system_constraint}

TASK: Evaluate the following medical question-answer pair.

Question: {question}
Answer: {answer}

Score each dimension as an integer from 0 to 10.
Set "recommendation" to exactly one of: "approve", "revise", "reject".
Set "feedback" to a brief string (one or two sentences).

OUTPUT — exactly this JSON structure, nothing else:
{{"accuracy": 0, "completeness": 0, "clarity": 0, "relevance": 0, "overall": 0, "recommendation": "approve", "feedback": ""}}"""

FACT_CHECK_PROMPT = """{system_constraint}

TASK: Fact-check the following medical question-answer pair.

Question: {question}
Answer: {answer}

Score "factual_accuracy" as an integer from 0 to 10.
"analysis" must be a JSON array of short strings (key findings).
"suggested_answer" must be a string with an improved answer if factual_accuracy < 8, otherwise null.
"confidence" must be a float from 0.0 to 1.0.

OUTPUT — exactly this JSON structure, nothing else:
{{"factual_accuracy": 0, "analysis": ["point 1"], "suggested_answer": null, "confidence": 0.5}}"""

# Appended to the prompt on a single retry after parse failure
_RETRY_SUFFIX = (
    "\n\nYour previous response was invalid. "
    "Return ONLY valid JSON. "
    "Do not include explanation. "
    "Do not include markdown. "
    "Do not include reasoning. "
    "Do not include <think> tags. "
    "Output only the JSON object."
)


# Review uses low temperature and shorter max_tokens for deterministic structured output
_REVIEW_TEMPERATURE = 0.3
_REVIEW_MAX_TOKENS = 500


# ── Internal LLM call helper ──────────────────────────────────────────

async def _execute_llm_call(
    prompt: str,
    provider: str,
    api_key: str,
    model: str,
    ollama_url: str,
    rate_handler: RateLimitHandler | None,
) -> str:
    """Execute a single LLM call through the unified provider, with optional rate-limit handling."""

    async def _do_call() -> str:
        return await call_provider(
            prompt=prompt,
            provider=provider,
            model=model or "llama3",
            api_key=api_key,
            ollama_url=ollama_url,
            temperature=_REVIEW_TEMPERATURE,
            top_p=0.9,
            max_tokens=_REVIEW_MAX_TOKENS,
        )

    if rate_handler:
        return await rate_handler.execute_with_retry(_do_call)
    return await _do_call()


# ── Parse + validate + retry orchestrators ─────────────────────────────

def _log_parse_diagnostics(
    context: str,
    result,
    retried: bool = False,
    fallback: bool = False,
    violations: list[str] | None = None,
) -> None:
    """Transparent structured logging for every parse attempt."""
    logger.info(
        "JSON parse [%s]: method=%s retried=%s fallback=%s | raw=%s | sanitized=%s%s",
        context,
        result.method,
        retried,
        fallback,
        result.raw_preview[:150] if result.raw_preview else "(empty)",
        result.sanitized_preview[:150] if result.sanitized_preview else "(empty)",
        f" | violations={violations}" if violations else "",
    )


async def _parse_review_with_retry(
    raw: str,
    prompt: str,
    provider: str,
    api_key: str,
    model: str,
    ollama_url: str,
    rate_handler: RateLimitHandler | None,
) -> dict[str, Any]:
    """Parse a review response with schema validation and single retry.

    Flow:
    1. Parse raw → validate schema → return if valid
    2. If parse or validation fails → retry LLM call with correction suffix
    3. Parse retry response → validate → return if valid
    4. If still invalid → coerced fallback
    """
    # ── First attempt ──
    result = safe_parse_json(raw)
    if result.ok:
        valid, violations = validate_review_schema(result.data)
        if valid:
            _log_parse_diagnostics("review", result)
            return coerce_review(result.data)
        logger.warning("Review schema violations (attempt 1): %s", violations)
    else:
        logger.warning("Review JSON parse failed (attempt 1): method=%s", result.method)

    # ── Single retry ──
    try:
        retry_raw = await _execute_llm_call(
            prompt + _RETRY_SUFFIX, provider, api_key, model, ollama_url, rate_handler,
        )
        retry_result = safe_parse_json(retry_raw)
        if retry_result.ok:
            valid, violations = validate_review_schema(retry_result.data)
            if valid:
                _log_parse_diagnostics("review-retry", retry_result, retried=True)
                return coerce_review(retry_result.data)
            logger.warning("Review schema violations (retry): %s", violations)
            # Retry produced parseable but invalid schema — coerce what we can
            _log_parse_diagnostics("review-retry-coerce", retry_result, retried=True, violations=violations)
            return coerce_review(retry_result.data)
        else:
            _log_parse_diagnostics("review-retry-failed", retry_result, retried=True, fallback=True)
    except Exception as e:
        logger.warning("Review retry LLM call failed: %s", e)

    # ── Fallback ──
    # If first attempt parsed but failed validation, coerce it
    if result.ok:
        _log_parse_diagnostics("review-fallback-coerce", result, retried=True, fallback=True)
        return coerce_review(result.data)

    _log_parse_diagnostics("review-fallback", result, retried=True, fallback=True)
    return _fallback_review()


async def _parse_fact_check_with_retry(
    raw: str,
    prompt: str,
    provider: str,
    api_key: str,
    model: str,
    ollama_url: str,
    rate_handler: RateLimitHandler | None,
) -> dict[str, Any]:
    """Parse a fact-check response with schema validation and single retry."""
    # ── First attempt ──
    result = safe_parse_json(raw)
    if result.ok:
        valid, violations = validate_fact_check_schema(result.data)
        if valid:
            _log_parse_diagnostics("fact-check", result)
            return coerce_fact_check(result.data)
        logger.warning("Fact-check schema violations (attempt 1): %s", violations)
    else:
        logger.warning("Fact-check JSON parse failed (attempt 1): method=%s", result.method)

    # ── Single retry ──
    try:
        retry_raw = await _execute_llm_call(
            prompt + _RETRY_SUFFIX, provider, api_key, model, ollama_url, rate_handler,
        )
        retry_result = safe_parse_json(retry_raw)
        if retry_result.ok:
            valid, violations = validate_fact_check_schema(retry_result.data)
            if valid:
                _log_parse_diagnostics("fact-check-retry", retry_result, retried=True)
                return coerce_fact_check(retry_result.data)
            logger.warning("Fact-check schema violations (retry): %s", violations)
            _log_parse_diagnostics("fact-check-retry-coerce", retry_result, retried=True, violations=violations)
            return coerce_fact_check(retry_result.data)
        else:
            _log_parse_diagnostics("fact-check-retry-failed", retry_result, retried=True, fallback=True)
    except Exception as e:
        logger.warning("Fact-check retry LLM call failed: %s", e)

    # ── Fallback ──
    if result.ok:
        _log_parse_diagnostics("fact-check-fallback-coerce", result, retried=True, fallback=True)
        return coerce_fact_check(result.data)

    _log_parse_diagnostics("fact-check-fallback", result, retried=True, fallback=True)
    return _fallback_fact_check()


# ── Public API (interface unchanged) ──────────────────────────────────

async def review_qa_pair(
    question: str,
    answer: str,
    provider: str,
    api_key: str = "",
    model: str = "",
    ollama_url: str = "http://host.docker.internal:11434",
    rate_handler: RateLimitHandler | None = None,
) -> dict[str, Any]:
    """Review a single Q&A pair, optionally with rate limit handling."""
    prompt = REVIEW_PROMPT.format(
        system_constraint=_SYSTEM_CONSTRAINT,
        question=question,
        answer=answer,
    )

    raw = await _execute_llm_call(prompt, provider, api_key, model, ollama_url, rate_handler)
    return await _parse_review_with_retry(
        raw, prompt, provider, api_key, model, ollama_url, rate_handler,
    )


async def fact_check_qa_pair(
    question: str,
    answer: str,
    provider: str,
    api_key: str = "",
    model: str = "",
    ollama_url: str = "http://host.docker.internal:11434",
    rate_handler: RateLimitHandler | None = None,
) -> dict[str, Any]:
    """Fact-check a single Q&A pair."""
    prompt = FACT_CHECK_PROMPT.format(
        system_constraint=_SYSTEM_CONSTRAINT,
        question=question,
        answer=answer,
    )

    raw = await _execute_llm_call(prompt, provider, api_key, model, ollama_url, rate_handler)
    return await _parse_fact_check_with_retry(
        raw, prompt, provider, api_key, model, ollama_url, rate_handler,
    )


async def review_batch(
    pairs: list[dict[str, str]],
    provider: str,
    api_key: str = "",
    model: str = "",
    ollama_url: str = "http://host.docker.internal:11434",
    max_concurrent: int | None = None,
    speed: str = "normal",
    on_pair_complete: Any | None = None,
) -> list[dict[str, Any]]:
    """Review a batch of Q&A pairs with rate limiting and incremental progress.

    Args:
        on_pair_complete: Optional async callback(index, total, result) called after each pair.
    """
    if max_concurrent is None:
        max_concurrent = get_concurrency_limit(provider, model)

    rate_handler = RateLimitHandler(provider, model, speed=speed)
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[dict[str, Any] | None] = [None] * len(pairs)
    completed = 0

    async def review_one(idx: int, pair: dict[str, str]) -> None:
        nonlocal completed
        async with semaphore:
            try:
                result = await review_qa_pair(
                    question=pair["question"],
                    answer=pair["answer"],
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    ollama_url=ollama_url,
                    rate_handler=rate_handler,
                )
                result["qa_pair_id"] = pair["id"]
                results[idx] = result
            except (RateLimitExceeded, AuthenticationError) as e:
                logger.error("Review failed for pair %s: %s", pair.get("id", "?"), e)
                results[idx] = {
                    "qa_pair_id": pair["id"],
                    "error": str(e),
                    "accuracy": 0, "completeness": 0, "clarity": 0,
                    "relevance": 0, "overall": 0,
                    "recommendation": "revise",
                    "feedback": f"Review failed: {e}",
                }
            except Exception as e:
                logger.error("Review failed for pair %s: %s", pair.get("id", "?"), e)
                results[idx] = {
                    "qa_pair_id": pair["id"],
                    "error": str(e),
                    "accuracy": 0, "completeness": 0, "clarity": 0,
                    "relevance": 0, "overall": 0,
                    "recommendation": "revise",
                    "feedback": f"Review failed: {e}",
                }
            finally:
                completed += 1
                if on_pair_complete:
                    try:
                        await on_pair_complete(completed, len(pairs), results[idx])
                    except Exception:
                        pass

    tasks = [review_one(i, p) for i, p in enumerate(pairs)]
    await asyncio.gather(*tasks)

    # GPU cleanup after batch
    release_gpu_memory("post-review-batch")

    return [r for r in results if r is not None]


# ── Fallback factories ─────────────────────────────────────────────────

def _fallback_review() -> dict[str, Any]:
    logger.warning("Review: returning fallback result (all attempts exhausted)")
    return {
        "accuracy": 0,
        "completeness": 0,
        "clarity": 0,
        "relevance": 0,
        "overall": 0,
        "recommendation": "revise",
        "feedback": "Could not parse review response after retry",
    }


def _fallback_fact_check() -> dict[str, Any]:
    logger.warning("Fact-check: returning fallback result (all attempts exhausted)")
    return {
        "factual_accuracy": 0,
        "analysis": [],
        "suggested_answer": None,
        "confidence": 0.0,
    }
