"""Batch LLM generation with adaptive over-generation, mini-batching, and early stopping.

This service manages the higher-level orchestration of LLM calls:
  - Mini-batches of 15–20 prompts to prevent memory buildup
  - Truly adaptive over-generation multiplier (adjusts after each mini-batch)
  - Early stopping when target pair count is reached
  - Memory cleanup between batches (gc + GPU cache)
  - Detailed statistics tracking

Actual LLM calls are delegated to ``llm_generation_client.generate_qa_batch()``.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

from app.services.llm_generation_client import (
    generate_qa_batch as llm_generate_batch,
    OllamaQAPair,
)
from app.utils.gpu_cleanup import release_gpu_memory

logger = logging.getLogger(__name__)


# ── Statistics tracker ─────────────────────────────────────────────────

@dataclass
class BatchStats:
    """Running statistics for a generation batch."""
    prompts_sent: int = 0
    parse_successes: int = 0
    parse_failures: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    semantic_dups: int = 0
    quality_failures: int = 0
    truncated: int = 0
    not_found: int = 0

    @property
    def success_rate(self) -> float:
        """Fraction of prompts that produced parseable Q&A pairs."""
        return self.parse_successes / max(self.prompts_sent, 1)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of parsed pairs that passed quality validation."""
        return self.accepted / max(self.parse_successes, 1)

    @property
    def overall_efficiency(self) -> float:
        """End-to-end: accepted / prompts_sent."""
        return self.accepted / max(self.prompts_sent, 1)

    def summary_line(self) -> str:
        return (
            f"prompts={self.prompts_sent} parsed={self.parse_successes} "
            f"accepted={self.accepted} rejected={self.rejected} "
            f"(dups={self.duplicates} sem_dups={self.semantic_dups} "
            f"quality={self.quality_failures} truncated={self.truncated} "
            f"not_found={self.not_found}) "
            f"efficiency={self.overall_efficiency:.0%}"
        )


# ── Adaptive batch generator ──────────────────────────────────────────

class AdaptiveBatchGenerator:
    """Orchestrates LLM generation with adaptive over-generation and mini-batching.

    Usage::

        gen = AdaptiveBatchGenerator(target_pairs=50)
        prompts = prepare_rag_prompts(...)   # from prompt_builder
        pairs = await gen.generate(prompts, provider="ollama", ...)
        # gen.stats contains full statistics
    """

    def __init__(
        self,
        target_pairs: int,
        initial_multiplier: float = 2.0,
        min_multiplier: float = 1.3,
        max_multiplier: float = 3.5,
        adapt_interval: int = 10,
        mini_batch_size: int = 15,
    ):
        self.target_pairs = target_pairs
        self.multiplier = initial_multiplier
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.adapt_interval = adapt_interval
        self.mini_batch_size = mini_batch_size
        self.stats = BatchStats()

    # ── Public API ─────────────────────────────────────────────────────

    def compute_target_prompts(self) -> int:
        """How many prompts to prepare, given the current multiplier."""
        return int(self.target_pairs * self.multiplier)

    def should_stop(self) -> bool:
        """Early stopping: target already reached."""
        return self.stats.accepted >= self.target_pairs

    def remaining_needed(self) -> int:
        """How many more accepted pairs are still needed."""
        return max(0, self.target_pairs - self.stats.accepted)

    async def generate(
        self,
        chunks_with_prompts: list[tuple[str, str]],
        provider: str,
        model: str,
        api_key: str = "",
        ollama_url: str = "http://host.docker.internal:11434",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_concurrent: int = 5,
        progress_callback: Any | None = None,
    ) -> list[OllamaQAPair]:
        """Process prompts in mini-batches with adaptive multiplier and early stopping.

        Parameters
        ----------
        chunks_with_prompts : list of (chunk_text, prompt) tuples
        progress_callback : optional async(completed, total, pair) callback

        Returns
        -------
        All successfully parsed OllamaQAPair objects across all mini-batches.
        """
        all_pairs: list[OllamaQAPair] = []
        total_prompts = len(chunks_with_prompts)
        total_batches = (total_prompts + self.mini_batch_size - 1) // self.mini_batch_size
        global_completed = 0

        logger.info(
            "AdaptiveBatchGenerator: %d prompts in %d mini-batches "
            "(target=%d pairs, multiplier=%.1fx, concurrency=%d)",
            total_prompts, total_batches, self.target_pairs,
            self.multiplier, max_concurrent,
        )

        for batch_start in range(0, total_prompts, self.mini_batch_size):
            if self.should_stop():
                logger.info(
                    "Early stopping: target %d reached after %d prompts",
                    self.target_pairs, self.stats.prompts_sent,
                )
                break

            batch_end = min(batch_start + self.mini_batch_size, total_prompts)
            batch = chunks_with_prompts[batch_start:batch_end]
            batch_num = batch_start // self.mini_batch_size + 1

            logger.info(
                "Mini-batch %d/%d: %d prompts (accepted so far: %d/%d)",
                batch_num, total_batches, len(batch),
                self.stats.accepted, self.target_pairs,
            )

            # Wrap progress callback to offset by global_completed
            batch_base = global_completed

            async def _batch_progress(completed: int, total: int, pair: Any) -> None:
                if progress_callback:
                    await progress_callback(
                        batch_base + completed, total_prompts, pair,
                    )

            pairs = await llm_generate_batch(
                chunks_with_prompts=batch,
                provider=provider,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
                temperature=temperature,
                top_p=top_p,
                max_concurrent=max_concurrent,
                max_pairs=self.remaining_needed(),
                progress_callback=_batch_progress,
            )

            all_pairs.extend(pairs)
            global_completed += len(batch)
            self.stats.prompts_sent += len(batch)
            self.stats.parse_successes += len(pairs)
            self.stats.parse_failures += len(batch) - len(pairs)

            # Offset prompt_index so it maps to the global prepared list
            for p in pairs:
                if hasattr(p, 'prompt_index') and p.prompt_index >= 0:
                    p.prompt_index += batch_start

            # Adapt multiplier after each mini-batch
            self._adapt_multiplier()

            # Memory cleanup between batches (lightweight — keep model loaded)
            gc.collect()
            release_gpu_memory(f"post-mini-batch-{batch_num}", full_unload=False)

        logger.info(
            "Batch generation complete: %s", self.stats.summary_line(),
        )
        return all_pairs

    # ── Internal ───────────────────────────────────────────────────────

    def update_accepted(self, count: int) -> None:
        """Update accepted count from external validation results.

        Called by the orchestrator after ``QualityValidator`` determines
        how many pairs passed all checks.
        """
        self.stats.accepted += count

    def _adapt_multiplier(self) -> None:
        """Adjust the over-generation multiplier based on observed acceptance rate.

        Uses acceptance_rate (accepted / parsed) when available, falling back
        to success_rate (parsed / sent) when no validation has occurred yet.
        """
        if self.stats.prompts_sent < self.adapt_interval:
            return

        # Prefer acceptance rate (end-to-end quality), fall back to parse rate
        if self.stats.accepted > 0:
            rate = self.stats.acceptance_rate  # accepted / parsed
        else:
            rate = self.stats.success_rate  # parsed / sent

        old_mult = self.multiplier

        if rate < 0.3:
            self.multiplier = min(self.max_multiplier, self.multiplier * 1.4)
        elif rate < 0.5:
            self.multiplier = min(self.max_multiplier, self.multiplier * 1.2)
        elif rate > 0.8:
            self.multiplier = max(self.min_multiplier, self.multiplier * 0.8)

        if abs(self.multiplier - old_mult) > 0.01:
            logger.info(
                "Adaptive multiplier: %.2f → %.2f (rate=%.0f%%, metric=%s)",
                old_mult, self.multiplier, rate * 100,
                "acceptance" if self.stats.accepted > 0 else "parse",
            )
