"""Ollama API service — connection testing, health checks, and Q&A response parsing.

Provides:
  - test_connection(): verify Ollama is reachable and list models
  - health_check(): pre-generation connectivity verification
  - _parse_qa_response(): robust Q/A parser (handles thinking-model output)
  - OllamaQAPair: dataclass for parsed Q&A results

Actual LLM generation is handled by llm_generation_client.py which supports
both Ollama and cloud providers through a unified interface.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OllamaQAPair:
    """A single Q&A pair returned from LLM generation."""
    question: str
    answer: str
    model: str
    prompt_template: str | None = None
    prompt_index: int = -1  # Index into the original prepared prompts list


async def test_connection(ollama_url: str) -> dict:
    """Test connectivity to Ollama and return available models."""
    base_url = ollama_url.rstrip("/")
    tags_url = f"{base_url}/api/tags"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(tags_url)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "Unknown") for m in data.get("models", [])]
            return {"success": True, "models": models}
        except httpx.ConnectError:
            return {"success": False, "error": "Cannot connect to Ollama. Is it running?"}
        except httpx.TimeoutException:
            return {"success": False, "error": "Connection timed out. Ollama may be slow or unreachable."}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def health_check(ollama_url: str, retries: int = 3, delay: float = 5.0) -> bool:
    """Verify Ollama is responsive before starting expensive generation.

    Retries up to *retries* times with *delay* seconds between attempts.
    Returns True if Ollama responds to a lightweight request.
    """
    for attempt in range(retries):
        result = await test_connection(ollama_url)
        if result["success"]:
            logger.info("Ollama health check passed (%d models available)", len(result.get("models", [])))
            return True
        logger.warning(
            "Ollama health check attempt %d/%d failed: %s",
            attempt + 1, retries, result.get("error"),
        )
        if attempt < retries - 1:
            await asyncio.sleep(delay)
    logger.error("Ollama health check failed after %d attempts", retries)
    return False


def _parse_qa_response(response_text: str, model_name: str) -> OllamaQAPair | None:
    """Parse a Question/Answer pair from an LLM text response.

    Handles many common response formats:
      - ``Question: ... Answer: ...``  (our prompt format)
      - ``Q: ... A: ...``
      - ``**Question:** ... **Answer:** ...``  (Markdown bold)
      - Thinking-model wrappers like ``<think>...</think>`` which must be stripped
      - Multi-line answers that may contain blank lines
    """
    if not response_text:
        return None

    text = response_text

    # ── Strip "thinking" blocks that some models (e.g. deepseek-r1, lfm2.5-thinking) emit
    # First try matching closed tags (greedy to get the largest block)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Handle unclosed thinking tags — strip from <think> to end if no closing tag
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thought>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Strip markdown bold/italic wrapping around labels
    text = re.sub(r"\*\*(Question|Answer|Q|A)\*\*", r"\1", text, flags=re.IGNORECASE)
    text = text.strip()

    if not text:
        logger.debug("Response empty after stripping thinking blocks")
        return None

    # ── Try to find Question/Answer boundaries ──
    # Pattern 1:  Question: ... Answer: ...  (with or without leading ## / ** markers)
    q_match = re.search(
        r"(?:^|\n)\s*(?:#*\s*)?(?:Question|Q)\s*[:：]\s*(.*?)(?=\s*\n\s*(?:#*\s*)?(?:Answer|A)\s*[:：])",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    a_match = re.search(
        r"(?:^|\n)\s*(?:#*\s*)?(?:Answer|A)\s*[:：]\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if not q_match or not a_match:
        # Pattern 2: Lines starting with "Q:" and "A:" (compact format)
        q_match = re.search(r"^Q\s*[:：]\s*(.*?)(?=\nA\s*[:：])", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        a_match = re.search(r"^A\s*[:：]\s*(.*)", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)

    if not q_match or not a_match:
        logger.debug("Could not parse Q/A from response (first 300 chars): %s", text[:300])
        return None

    question = q_match.group(1).strip()
    answer = a_match.group(1).strip()

    # Clean up residual formatting
    question = re.sub(r"\s+", " ", question).strip().strip('"').strip("*")
    # For answers, preserve paragraph breaks but normalise intra-paragraph whitespace
    answer = re.sub(r"[ \t]+", " ", answer).strip().strip('"').strip("*")
    # Remove trailing "---" or "Note:" disclaimers that some models append
    answer = re.sub(r"\n-{3,}.*$", "", answer, flags=re.DOTALL).strip()
    answer = re.sub(r"\n(?:Note|Disclaimer|Sources?)\s*:.*$", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()

    # ── Strip citation artifacts that RAG evidence may have introduced ──
    # Patterns: [Evidence 1], [Evidence 2, 3], *[Evidence 1]*, (Evidence 1),
    #           [1], [2], [Ref 1], [Source 1], etc.
    _CITATION_PATTERNS = [
        r"\[Evidence\s*\d+(?:\s*,\s*\d+)*\]",   # [Evidence 1], [Evidence 1, 2]
        r"\*?\[Evidence\s*\d+\]\*?",               # *[Evidence 1]*
        r"\(Evidence\s*\d+(?:\s*,\s*\d+)*\)",     # (Evidence 1)
        r"\[Ref(?:erence)?\s*\d+\]",              # [Ref 1], [Reference 1]
        r"\[Source\s*\d+\]",                       # [Source 1]
        r"\[\d+\]",                                # [1], [2] — bare numeric citations
    ]
    for pat in _CITATION_PATTERNS:
        answer = re.sub(pat, "", answer, flags=re.IGNORECASE)
        question = re.sub(pat, "", question, flags=re.IGNORECASE)
    # Clean up double spaces left after citation removal
    answer = re.sub(r"  +", " ", answer).strip()
    question = re.sub(r"  +", " ", question).strip()

    # Basic length validation — relaxed thresholds
    if len(question) < 10 or len(answer) < 20:
        logger.debug("Q/A too short after parsing: Q=%d chars, A=%d chars", len(question), len(answer))
        return None

    return OllamaQAPair(question=question, answer=answer, model=model_name)
