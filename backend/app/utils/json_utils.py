"""Universal JSON extraction and sanitization for LLM output.

All LLM output is treated as hostile text.  This module provides a
deterministic, provider-agnostic pipeline to extract valid JSON from
arbitrary model responses that may contain:

  - ``<think>…</think>`` reasoning blocks
  - Markdown code fences (```json … ```)
  - Leading / trailing commentary
  - Nested JSON objects and arrays
  - Alignment disclaimers

Pipeline
--------
1. **Sanitize** — strip thinking tags, markdown fences, whitespace
2. **Direct parse** — attempt ``json.loads()`` on the cleaned text
3. **Balanced-brace extraction** — character-level scan for first ``{…}``
4. **Parse extracted** — ``json.loads()`` on the substring
5. **Failure** — return ``None``

No regex is used for structural JSON extraction.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. Sanitization ───────────────────────────────────────────────────

def sanitize_llm_output(raw: str) -> str:
    """Strip non-JSON artefacts from raw LLM output.

    Operations (in order):
    1. Remove ``<think>…</think>`` blocks (case-insensitive, greedy)
    2. Remove unclosed ``<think>`` to end-of-string
    3. Same for ``<reasoning>``, ``<thought>``
    4. Remove markdown code fences (``` and ```json)
    5. Trim leading/trailing whitespace
    """
    if not raw:
        return ""

    text = raw

    # Strip closed thinking / reasoning / thought blocks
    for tag in ("think", "reasoning", "thought"):
        text = re.sub(
            rf"<{tag}>.*?</{tag}>", "", text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    # Strip unclosed tags (tag to end-of-string)
    for tag in ("think", "reasoning", "thought"):
        text = re.sub(
            rf"<{tag}>.*$", "", text,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Strip markdown code fences:  ```json … ```  or  ``` … ```
    # Replace opening fence (with optional language tag) and closing fence
    text = re.sub(r"```(?:json|JSON)?\s*\n?", "", text)

    return text.strip()


# ── 2. Balanced-brace extraction ──────────────────────────────────────

def extract_json_object(text: str) -> str | None:
    """Extract the first balanced ``{…}`` substring using character scanning.

    Handles nested objects and arrays, and respects JSON string literals
    (braces inside quoted strings are ignored).

    Returns the substring including the outer braces, or ``None`` if no
    balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            if in_string:
                escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Never balanced
    return None


# ── 3. Safe parse pipeline ────────────────────────────────────────────

class ParseResult:
    """Outcome of a ``safe_parse_json`` call with diagnostic metadata."""

    __slots__ = ("data", "method", "raw_preview", "sanitized_preview")

    def __init__(
        self,
        data: dict[str, Any] | None,
        method: str,
        raw_preview: str = "",
        sanitized_preview: str = "",
    ):
        self.data = data
        self.method = method                # "direct" | "extraction" | "failed"
        self.raw_preview = raw_preview
        self.sanitized_preview = sanitized_preview

    @property
    def ok(self) -> bool:
        return self.data is not None


def safe_parse_json(raw: str) -> ParseResult:
    """Deterministic 3-step JSON extraction from arbitrary LLM output.

    Steps
    -----
    1. Sanitize → attempt ``json.loads``
    2. If that fails → balanced-brace extraction → ``json.loads``
    3. If that fails → return failure result

    Returns a ``ParseResult`` with diagnostic metadata for logging.
    """
    raw_preview = (raw or "")[:300]
    sanitized = sanitize_llm_output(raw or "")
    sanitized_preview = sanitized[:300]

    # Step 1: direct parse on full sanitized text
    try:
        data = json.loads(sanitized)
        if isinstance(data, dict):
            return ParseResult(data, "direct", raw_preview, sanitized_preview)
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 2: balanced-brace extraction
    extracted = extract_json_object(sanitized)
    if extracted:
        try:
            data = json.loads(extracted)
            if isinstance(data, dict):
                return ParseResult(data, "extraction", raw_preview, sanitized_preview)
        except (json.JSONDecodeError, ValueError):
            pass

    # Step 3: failure
    return ParseResult(None, "failed", raw_preview, sanitized_preview)


# ── 4. Schema validation helpers ──────────────────────────────────────

def validate_review_schema(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a parsed review dict against the expected schema.

    Returns (is_valid, list_of_violations).
    Violations are descriptive strings for logging.
    """
    violations: list[str] = []

    _NUMERIC_FIELDS = ("accuracy", "completeness", "clarity", "relevance", "overall")
    for key in _NUMERIC_FIELDS:
        val = data.get(key)
        if val is None:
            violations.append(f"missing required field '{key}'")
        elif not isinstance(val, (int, float)):
            violations.append(f"'{key}' is not numeric: {type(val).__name__}")
        elif not (0 <= val <= 10):
            violations.append(f"'{key}' out of range 0-10: {val}")

    rec = data.get("recommendation")
    if rec not in ("approve", "revise", "reject"):
        violations.append(f"invalid recommendation: {rec!r}")

    fb = data.get("feedback")
    if fb is not None and not isinstance(fb, str):
        violations.append(f"feedback is not a string: {type(fb).__name__}")

    return (len(violations) == 0, violations)


def validate_fact_check_schema(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a parsed fact-check dict against the expected schema."""
    violations: list[str] = []

    fa = data.get("factual_accuracy")
    if fa is None:
        violations.append("missing required field 'factual_accuracy'")
    elif not isinstance(fa, (int, float)):
        violations.append(f"'factual_accuracy' is not numeric: {type(fa).__name__}")
    elif not (0 <= fa <= 10):
        violations.append(f"'factual_accuracy' out of range 0-10: {fa}")

    analysis = data.get("analysis")
    if analysis is not None and not isinstance(analysis, list):
        violations.append(f"'analysis' is not a list: {type(analysis).__name__}")

    conf = data.get("confidence")
    if conf is not None:
        if not isinstance(conf, (int, float)):
            violations.append(f"'confidence' is not numeric: {type(conf).__name__}")
        elif not (0.0 <= conf <= 1.0):
            violations.append(f"'confidence' out of range 0-1: {conf}")

    sa = data.get("suggested_answer")
    if sa is not None and not isinstance(sa, (str, type(None))):
        violations.append(f"'suggested_answer' is not string or null: {type(sa).__name__}")

    return (len(violations) == 0, violations)


# ── 5. Coercion (only used after validation or on fallback) ───────────

def coerce_review(data: dict[str, Any]) -> dict[str, Any]:
    """Clamp numeric fields and fill missing values for a review result.

    Call this only when you accept the data (either validated or fallback).
    """
    out = dict(data)
    for key in ("accuracy", "completeness", "clarity", "relevance", "overall"):
        try:
            out[key] = max(0, min(10, int(round(float(out.get(key, 5))))))
        except (ValueError, TypeError):
            out[key] = 5

    rec = out.get("recommendation")
    if rec not in ("approve", "revise", "reject"):
        overall = out.get("overall", 5)
        if overall >= 7:
            out["recommendation"] = "approve"
        elif overall >= 4:
            out["recommendation"] = "revise"
        else:
            out["recommendation"] = "reject"

    if "feedback" not in out or not isinstance(out.get("feedback"), str):
        out["feedback"] = ""

    return out


def coerce_fact_check(data: dict[str, Any]) -> dict[str, Any]:
    """Clamp and fill missing values for a fact-check result."""
    out: dict[str, Any] = {}
    try:
        out["factual_accuracy"] = max(0, min(10, float(data.get("factual_accuracy", 5))))
    except (ValueError, TypeError):
        out["factual_accuracy"] = 5

    analysis = data.get("analysis", [])
    if isinstance(analysis, list):
        out["analysis"] = [str(a) for a in analysis]
    elif analysis is not None:
        out["analysis"] = [str(analysis)]
    else:
        out["analysis"] = []

    out["suggested_answer"] = data.get("suggested_answer")
    if out["suggested_answer"] is not None:
        out["suggested_answer"] = str(out["suggested_answer"])

    try:
        out["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (ValueError, TypeError):
        out["confidence"] = 0.5

    return out
