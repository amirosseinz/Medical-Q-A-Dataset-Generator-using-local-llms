"""Prompt construction — builds LLM prompts with evidence formatted WITHOUT citation artifacts.

This is the ROOT FIX for citation leakage.  Evidence is presented with simple
separators (``--- Medical Source ---``) so the LLM never sees ``[Evidence 1]``
labels it might parrot in its answer.

Also handles:
  - Uniqueness instructions (avoid previously-generated questions)
  - Question-type and difficulty cycling
  - Both RAG and non-RAG prompt paths
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.utils.prompts import build_prompt, build_rag_prompt

logger = logging.getLogger(__name__)


# ── Prepared prompt bundle ─────────────────────────────────────────────

@dataclass
class PreparedPrompt:
    """A prompt bundled with its source-attribution metadata.

    ``chunk_text`` is the primary evidence chunk text — kept for backward
    compatibility with ``generate_qa_batch(chunks_with_prompts)`` which
    expects ``list[tuple[str, str]]``.
    """
    chunk_text: str
    prompt: str
    metadata: dict = field(default_factory=dict)

    def as_tuple(self) -> tuple[str, str]:
        """Return (chunk_text, prompt) pair for generate_qa_batch."""
        return (self.chunk_text, self.prompt)


# ── Evidence formatting (NO citation tags) ─────────────────────────────

def format_evidence_for_prompt(chunks: list[Any]) -> str:
    """Format retrieved evidence chunks into text for the LLM prompt.

    CRITICAL: This function deliberately does NOT use ``[Evidence 1]``,
    ``[Evidence 2]``, or any citation markers.  Evidence is separated by
    simple horizontal rules so the LLM treats them as context rather than
    citable references.

    Parameters
    ----------
    chunks : list of RetrievedChunk (or any object with .source_filename, .content)
    """
    if not chunks:
        return ""

    parts: list[str] = []
    for chunk in chunks:
        source_name = getattr(chunk, "source_filename", "unknown source")
        parts.append(
            f"--- Medical Source ({source_name}) ---\n"
            f"{chunk.content}"
        )
    return "\n\n".join(parts)


# ── Prompt builders ────────────────────────────────────────────────────

def build_generation_prompt(
    question_type: str,
    evidence_text: str,
    domain: str,
    difficulty: str = "intermediate",
    existing_questions: list[str] | None = None,
    is_rag: bool = True,
) -> str:
    """Build a complete prompt for Q&A generation.

    Parameters
    ----------
    question_type : factual | reasoning | comparison | application
    evidence_text : pre-formatted evidence (from ``format_evidence_for_prompt``)
    domain : medical domain / topic
    difficulty : beginner | intermediate | advanced
    existing_questions : if provided, append uniqueness instruction
    is_rag : True for RAG-augmented prompts, False for random-chunk prompts
    """
    if is_rag:
        prompt = build_rag_prompt(question_type, evidence_text, domain, difficulty)
    else:
        prompt = build_prompt(question_type, evidence_text, domain, difficulty)

    if existing_questions:
        recent = existing_questions[-15:]  # keep prompt manageable
        avoid_list = "\n".join(f"  - {q}" for q in recent)
        prompt += (
            "\n\nIMPORTANT: The following questions have already been generated. "
            "You MUST generate a DIFFERENT question that explores a new aspect "
            "or angle of the evidence. Do NOT repeat or rephrase these:\n"
            f"{avoid_list}"
        )

    return prompt


# ── Prompt preparation for RAG and random paths ───────────────────────

def prepare_rag_prompts(
    medical_terms: list[str],
    domain: str,
    question_types: list[str],
    difficulty_levels: list[str],
    target_prompts: int,
    existing_questions: list[str],
    project_id: str,
    db: Any,
    settings: Any,
) -> list[PreparedPrompt]:
    """Prepare prompts via FAISS retrieval for each medical term.

    Handles:
      - Per-term retrieval with varied top_k for evidence diversity
      - Evidence rotation across prompts for the same term
      - Question type and difficulty cycling
      - Full citation metadata tracking (stored in PreparedPrompt.metadata)

    Parameters
    ----------
    target_prompts : total number of prompts to prepare
    """
    from app.services.rag_service import retrieve_for_topic

    pairs_per_term = max(1, target_prompts // max(len(medical_terms), 1))
    prompt_idx = 0

    results: list[PreparedPrompt] = []
    _retrieval_cache: dict[str, Any] = {}

    for term in medical_terms:
        for pair_num in range(pairs_per_term):
            if len(results) >= target_prompts:
                break

            # Cache retrieval results — vary top_k for diversity
            cache_key = f"{term}_{pair_num % 3}"
            if cache_key not in _retrieval_cache:
                varied_top_k = settings.RAG_TOP_K + (pair_num % 3)
                _retrieval_cache[cache_key] = retrieve_for_topic(
                    project_id=project_id,
                    topic=term,
                    top_k=varied_top_k,
                    min_score=settings.RAG_MIN_SCORE,
                    db=db,
                )

            retrieval = _retrieval_cache[cache_key]
            if not retrieval.has_evidence:
                logger.info("No evidence above threshold for '%s'", term)
                continue

            # Rotate evidence order for diversity
            chunks_for_prompt = list(retrieval.chunks)
            if pair_num > 0 and len(chunks_for_prompt) > 1:
                rotation = pair_num % len(chunks_for_prompt)
                chunks_for_prompt = chunks_for_prompt[rotation:] + chunks_for_prompt[:rotation]

            # Format evidence WITHOUT citation tags
            evidence_text = format_evidence_for_prompt(chunks_for_prompt)

            # Cycle question types and difficulty levels
            qt = question_types[prompt_idx % len(question_types)]
            dl = difficulty_levels[prompt_idx % len(difficulty_levels)]
            prompt_idx += 1

            prompt = build_generation_prompt(
                question_type=qt,
                evidence_text=evidence_text,
                domain=domain,
                difficulty=dl,
                existing_questions=existing_questions,
                is_rag=True,
            )

            primary_chunk = chunks_for_prompt[0]
            results.append(PreparedPrompt(
                chunk_text=primary_chunk.content,
                prompt=prompt,
                metadata={
                    "source_type": "rag_ollama",
                    "primary_chunk_id": primary_chunk.chunk_id,
                    "citation_metadata": retrieval.citation_metadata(),
                    "citation_ids": retrieval.citation_ids(),
                    "query_term": term,
                    "retrieval_scores": [c.score for c in retrieval.chunks],
                    "source_filename": primary_chunk.source_filename,
                    "question_type": qt,
                },
            ))

    logger.info(
        "RAG prompt builder: prepared %d prompts from %d terms (target: %d)",
        len(results), len(medical_terms), target_prompts,
    )
    return results


def prepare_random_prompts(
    all_chunks_with_meta: list[tuple[str, str, str | None]],
    domain: str,
    question_types: list[str],
    difficulty_levels: list[str],
    target_prompts: int,
    existing_questions: list[str],
    config: dict,
) -> list[PreparedPrompt]:
    """Prepare prompts by randomly sampling chunks (non-RAG fallback).

    Handles:
      - Equal allocation across source types (PDF, PubMed)
      - Per-source chunk limits from config
      - Question type cycling
    """
    import random
    from collections import defaultdict

    # Group by source type and shuffle
    source_buckets: dict[str, list[tuple[str, str, str | None]]] = defaultdict(list)
    for item in all_chunks_with_meta:
        source_buckets[item[1]].append(item)
    for bucket in source_buckets.values():
        random.shuffle(bucket)

    # Apply per-source limits
    pdf_limit = config.get("pdf_chunk_limit") or 0
    pubmed_limit = config.get("pubmed_chunk_limit") or 0
    if pdf_limit > 0 and "PDF" in source_buckets:
        source_buckets["PDF"] = source_buckets["PDF"][:pdf_limit]
    if pubmed_limit > 0 and "PubMed" in source_buckets:
        source_buckets["PubMed"] = source_buckets["PubMed"][:pubmed_limit]

    # Equal allocation across types
    n_types = max(len(source_buckets), 1)
    equal_share = target_prompts // n_types
    chunks_to_process: list[tuple[str, str, str | None]] = []
    leftover: list[tuple[str, str, str | None]] = []

    for bucket in source_buckets.values():
        take = min(equal_share, len(bucket))
        chunks_to_process.extend(bucket[:take])
        if len(bucket) > take:
            leftover.extend(bucket[take:])

    remaining_slots = target_prompts - len(chunks_to_process)
    if remaining_slots > 0 and leftover:
        random.shuffle(leftover)
        chunks_to_process.extend(leftover[:remaining_slots])

    random.shuffle(chunks_to_process)

    results: list[PreparedPrompt] = []
    for i, (chunk_text, source_type, filename) in enumerate(chunks_to_process):
        qt = question_types[i % len(question_types)]
        dl = difficulty_levels[i % len(difficulty_levels)]

        prompt = build_generation_prompt(
            question_type=qt,
            evidence_text=chunk_text,
            domain=domain,
            difficulty=dl,
            existing_questions=existing_questions,
            is_rag=False,
        )

        results.append(PreparedPrompt(
            chunk_text=chunk_text,
            prompt=prompt,
            metadata={
                "source_type": source_type,
                "source_filename": filename,
            },
        ))

    logger.info(
        "Random prompt builder: prepared %d prompts from %s",
        len(results),
        ", ".join(f"{k}={len(v)}" for k, v in source_buckets.items()),
    )
    return results
