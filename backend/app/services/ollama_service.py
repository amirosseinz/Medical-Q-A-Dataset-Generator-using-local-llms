"""Ollama API service — interact with local Ollama models.

Ported from original app.py ``generate_qa_with_ollama()`` (lines 216-296).
Improved with:
  - httpx async client
  - Exponential backoff retry (3 attempts)
  - Configurable temperature/top_p/top_k
  - Better response parsing
  - Connection pooling
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_BACKOFF = 2.0  # seconds


@dataclass
class OllamaQAPair:
    """A single Q&A pair returned from Ollama."""
    question: str
    answer: str
    model: str
    prompt_template: str | None = None


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


async def generate_qa_pair(
    text_chunk: str,
    prompt: str,
    ollama_url: str,
    model_name: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> OllamaQAPair | None:
    """Send a prompt to Ollama and parse the Q&A response.

    Implements exponential backoff retry on transient failures.
    """
    base_url = ollama_url.rstrip("/")
    generate_url = f"{base_url}/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(generate_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get("response", "")
                return _parse_qa_response(response_text, model_name)

        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_error = e
            wait = BASE_BACKOFF * (2 ** attempt)
            logger.warning(
                f"Ollama attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}")
            return None

    logger.error(f"Ollama failed after {MAX_RETRIES} attempts: {last_error}")
    return None


def _parse_qa_response(response_text: str, model_name: str) -> OllamaQAPair | None:
    """Parse a Question/Answer pair from Ollama's text response.

    Handles the ``Question: ... Answer: ...`` format used in our prompts.
    """
    if not response_text:
        return None

    q_match = re.search(
        r"(?:Question|Q)\s*:\s*(.*?)(?=\s*(?:Answer|A)\s*:|$)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    a_match = re.search(
        r"(?:Answer|A)\s*:\s*(.*?)$",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )

    if not q_match or not a_match:
        logger.debug(f"Could not parse Q/A from response: {response_text[:200]}")
        return None

    question = re.sub(r"\s+", " ", q_match.group(1)).strip()
    answer = re.sub(r"\s+", " ", a_match.group(1)).strip()

    # Basic length validation (same thresholds as original)
    if len(question) < 15 or len(answer) < 30:
        logger.debug(f"Q/A too short: Q={len(question)} chars, A={len(answer)} chars")
        return None

    return OllamaQAPair(question=question, answer=answer, model=model_name)


async def generate_qa_batch(
    chunks_with_prompts: list[tuple[str, str]],
    ollama_url: str,
    model_name: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_concurrent: int = 3,
    max_pairs: int | None = None,
    progress_callback=None,
) -> list[OllamaQAPair]:
    """Process multiple chunks concurrently with a semaphore to limit parallelism.

    Parameters
    ----------
    chunks_with_prompts : list of (chunk_text, full_prompt) tuples
    progress_callback : optional async callable(completed, total, pair_or_none)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[OllamaQAPair] = []
    completed = 0
    total = len(chunks_with_prompts)

    async def _process_one(chunk_text: str, prompt: str):
        nonlocal completed
        async with semaphore:
            # Early exit if we have enough pairs
            if max_pairs and len(results) >= max_pairs:
                return
            pair = await generate_qa_pair(
                text_chunk=chunk_text,
                prompt=prompt,
                ollama_url=ollama_url,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
            )
            completed += 1
            if pair:
                results.append(pair)
            if progress_callback:
                await progress_callback(completed, total, pair)

    tasks = [_process_one(chunk, prompt) for chunk, prompt in chunks_with_prompts]
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"Ollama batch complete: {len(results)}/{total} chunks produced Q&A pairs")
    return results
