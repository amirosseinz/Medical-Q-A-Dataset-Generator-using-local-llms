"""Enhanced quality validation with semantic dedup, completeness checks, and stateful tracking.

Wraps the existing ``quality_checker`` functions and adds:
  - **Completeness check**: question must end with ``?``, answer with ``.``/``!``/``?``,
    minimum lengths enforced, truncation detection.
  - **Proactive semantic dedup**: maintains running embedding state so near-duplicate
    questions are caught BEFORE wasting an LLM call (when used in prompt-prep) or
    DURING validation (when used post-generation).
  - **NOT FOUND detection**: LLM responses that contain "NOT FOUND" are flagged.
  - **Detailed rejection logging**: every rejection gets a structured reason string
    suitable for display or export.

Usage::

    validator = QualityValidator(min_quality_score=0.4, semantic_dup_threshold=0.92)

    for pair in ai_pairs:
        result = validator.validate(pair.question, pair.answer)
        if result.passed:
            store(pair)
        else:
            log(result.rejection_reason)

    print(validator.summary())
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from app.services.quality_checker import (
    evaluate_qa_pair,
    CheckResult,
)

logger = logging.getLogger(__name__)


# ── Validation result ──────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of validating a single Q&A pair."""
    passed: bool
    score: float
    checks: list[CheckResult]
    rejection_reason: str | None = None


# ── Quality validator (stateful) ───────────────────────────────────────

class QualityValidator:
    """Stateful quality validator with semantic dedup and completeness enforcement.

    Holds running lists of accepted questions and their embeddings so that
    subsequent pairs can be checked for semantic similarity.

    Parameters
    ----------
    min_quality_score : minimum composite quality score (0–1) for acceptance
    semantic_dup_threshold : cosine similarity above which two questions are
        considered semantic duplicates (0–1).  0.92 catches only true
        paraphrases while allowing topical variation.
    existing_questions : seed list of already-accepted question strings
    """

    def __init__(
        self,
        min_quality_score: float = 0.4,
        semantic_dup_threshold: float = 0.92,
        existing_questions: list[str] | None = None,
    ):
        self.min_quality_score = min_quality_score
        self.semantic_dup_threshold = semantic_dup_threshold
        self.existing_questions: list[str] = list(existing_questions or [])
        self._embeddings: list[np.ndarray] = []

        # Running statistics
        self.total_validated: int = 0
        self.total_accepted: int = 0
        self.total_rejected: int = 0
        self.rejection_counts: dict[str, int] = {
            "not_found": 0,
            "completeness": 0,
            "quality": 0,
            "duplicate": 0,
            "semantic_dup": 0,
        }

    # ── Public API ─────────────────────────────────────────────────────

    def validate(self, question: str, answer: str) -> ValidationResult:
        """Run all quality checks on a Q&A pair.

        Check order (from cheapest to most expensive):
          1. NOT FOUND detection
          2. Completeness (structural checks — no embedding needed)
          3. Standard quality checks (length, format, content, string-dedup)
          4. Semantic dedup (embedding-based — most expensive)

        Returns a ``ValidationResult`` with a clear ``rejection_reason`` if failed.
        """
        self.total_validated += 1

        # ── 1. NOT FOUND check ──
        if "NOT FOUND" in answer.upper()[:50]:
            return self._reject("not_found", "LLM responded NOT FOUND")

        # ── 2. Completeness check ──
        completeness = self._check_completeness(question, answer)
        if not completeness.passed:
            issues = completeness.details.get("issues", [])
            return self._reject(
                "completeness",
                f"Incomplete: {'; '.join(issues)}",
                checks=[completeness],
            )

        # ── 3. Standard quality checks ──
        passed, score, checks = evaluate_qa_pair(
            question, answer, self.existing_questions,
            min_quality_score=self.min_quality_score,
        )
        checks.append(completeness)

        if not passed:
            failed = [c for c in checks if not c.passed]
            # Categorise: is it a duplicate or a quality failure?
            is_dup = any(c.check_type == "duplicate" for c in failed)
            category = "duplicate" if is_dup else "quality"
            detail = ", ".join(
                f"{c.check_type}={c.score:.2f}" for c in failed
            )
            return self._reject(
                category,
                f"{category}: {detail} (score={score:.2f})",
                score=score, checks=checks,
            )

        # ── 4. Semantic dedup ──
        is_dup, sim = self._check_semantic_duplicate(question)
        if is_dup:
            return self._reject(
                "semantic_dup",
                f"Semantic duplicate (similarity={sim:.3f})",
                score=score, checks=checks,
            )

        # ── Accepted ──
        self.total_accepted += 1
        self.existing_questions.append(question)
        self._track_embedding(question)

        # Log transparent scoring breakdown for accepted pair
        breakdown = " | ".join(
            f"{c.check_type}={c.score:.2f}" for c in checks
        )
        logger.debug(
            "Accepted (score=%.3f): %s | Q=%.60s",
            score, breakdown, question,
        )

        return ValidationResult(passed=True, score=score, checks=checks)

    def summary(self) -> dict:
        """Return aggregate validation statistics."""
        return {
            "total_validated": self.total_validated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate": self.total_accepted / max(self.total_validated, 1),
            "rejection_breakdown": dict(self.rejection_counts),
        }

    # ── Completeness check ─────────────────────────────────────────────

    @staticmethod
    def _check_completeness(question: str, answer: str) -> CheckResult:
        """Verify question and answer are complete, well-formed sentences.

        Catches the truncated-question problem (length=0.50 failures) by
        requiring questions to end with ``?`` and answers to end with
        sentence-ending punctuation.
        """
        issues: list[str] = []
        q = question.strip()
        a = answer.strip()

        # Question must end with ?
        if not q.endswith("?"):
            issues.append("Question does not end with '?'")

        # Minimum meaningful length
        if len(q) < 30:
            issues.append(f"Question too short ({len(q)} chars, min 30)")

        # Answer must end with sentence-ending punctuation
        if not re.search(r"[.!?]$", a):
            issues.append("Answer lacks sentence-ending punctuation")

        if len(a) < 50:
            issues.append(f"Answer too short ({len(a)} chars, min 50)")

        # Truncation detection
        if q.rstrip().endswith(("...", "…")) and not q.rstrip().endswith("...."):
            issues.append("Question appears truncated")
        if a.rstrip().endswith(("...", "…")) and not a.rstrip().endswith("...."):
            issues.append("Answer appears truncated")

        passed = len(issues) == 0
        score = max(0.0, 1.0 - len(issues) * 0.25)

        return CheckResult(
            check_type="completeness",
            passed=passed,
            score=score,
            details={"issues": issues},
        )

    # ── Semantic dedup ─────────────────────────────────────────────────

    def _check_semantic_duplicate(self, question: str) -> tuple[bool, float]:
        """Check embedding-level similarity against all accepted questions.

        Returns (is_duplicate, max_similarity).
        """
        if not self._embeddings:
            return False, 0.0

        try:
            from app.services.rag_service import embed_single

            q_vec = embed_single(question)
            stored = np.array(self._embeddings)
            sims = stored @ q_vec  # cosine (already L2-normed)
            max_sim = float(np.max(sims))
            return max_sim >= self.semantic_dup_threshold, max_sim
        except Exception as e:
            logger.debug("Semantic dedup failed (non-critical): %s", e)
            return False, 0.0

    def _track_embedding(self, question: str) -> None:
        """Store the embedding of an accepted question for future dedup."""
        try:
            from app.services.rag_service import embed_single
            self._embeddings.append(embed_single(question))
        except Exception:
            pass  # non-critical — string-based dedup still active

    # ── Internal helpers ───────────────────────────────────────────────

    def _reject(
        self,
        category: str,
        reason: str,
        score: float = 0.0,
        checks: list[CheckResult] | None = None,
    ) -> ValidationResult:
        self.total_rejected += 1
        self.rejection_counts[category] = self.rejection_counts.get(category, 0) + 1
        if checks:
            breakdown = " | ".join(f"{c.check_type}={c.score:.2f}" for c in checks)
            logger.info("Rejected [%s] score=%.3f (%s): %s", category, score, breakdown, reason)
        else:
            logger.info("Rejected [%s]: %s", category, reason)
        return ValidationResult(
            passed=False,
            score=score,
            checks=checks or [],
            rejection_reason=reason,
        )
