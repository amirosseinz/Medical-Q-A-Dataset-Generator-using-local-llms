"""Quality checking service — validate generated Q&A pairs.

Ported from original app.py ``evaluate_qa_pair()`` (lines 359-393).
Expanded with:
  - Composable sub-checks (length, format, relevance, duplicate)
  - Quality score computation (0-1)
  - Duplicate detection via difflib
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Check result ───────────────────────────────────────────────────────

@dataclass
class CheckResult:
    check_type: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: dict = field(default_factory=dict)


# ── Individual checks ──────────────────────────────────────────────────

def check_length(question: str, answer: str) -> CheckResult:
    """Validate question (15-200 chars) and answer (30-500 chars) lengths."""
    q_len = len(question)
    a_len = len(answer)
    q_ok = 15 <= q_len <= 500
    a_ok = 30 <= a_len <= 2000

    # Score: 1.0 if in ideal range, degrades outside
    q_score = 1.0 if 20 <= q_len <= 200 else (0.5 if q_ok else 0.0)
    a_score = 1.0 if 50 <= a_len <= 500 else (0.5 if a_ok else 0.0)
    score = (q_score + a_score) / 2

    return CheckResult(
        check_type="length",
        passed=q_ok and a_ok,
        score=score,
        details={"question_length": q_len, "answer_length": a_len},
    )


def check_format(question: str) -> CheckResult:
    """Verify question looks like a proper question."""
    question_starters = [
        "what", "how", "why", "when", "where", "which", "who",
        "can", "do", "is", "are", "does", "will", "should",
        "explain", "describe", "define", "compare", "discuss",
    ]
    q_lower = question.lower().strip()
    has_question_mark = question.strip().endswith("?")
    starts_with_qword = any(q_lower.startswith(w) for w in question_starters)

    passed = has_question_mark or starts_with_qword
    score = 1.0 if (has_question_mark and starts_with_qword) else (0.7 if passed else 0.0)

    return CheckResult(
        check_type="format",
        passed=passed,
        score=score,
        details={"has_question_mark": has_question_mark, "starts_with_qword": starts_with_qword},
    )


BAD_KEYWORDS = [
    "I cannot generate",
    "I am a language model",
    "I don't have enough information",
    "Based on the text provided",
    "As an AI",
    "I'm sorry",
    "I apologize",
]


def check_content_quality(question: str, answer: str) -> CheckResult:
    """Check for generic AI refusal patterns and low-quality indicators."""
    issues: list[str] = []

    for kw in BAD_KEYWORDS:
        if kw.lower() in answer.lower():
            issues.append(f"Contains bad keyword: {kw}")

    # Check for repetitive content
    words = answer.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append(f"Low word diversity: {unique_ratio:.2f}")

    passed = len(issues) == 0
    score = 1.0 if passed else max(0.0, 1.0 - len(issues) * 0.3)

    return CheckResult(
        check_type="relevance",
        passed=passed,
        score=score,
        details={"issues": issues},
    )


def check_duplicate(
    question: str,
    existing_questions: list[str],
    threshold: float = 0.85,
) -> CheckResult:
    """Check if question is a near-duplicate of any existing question."""
    q_normalized = re.sub(r"\s+", " ", question.lower().strip())

    for existing in existing_questions:
        existing_normalized = re.sub(r"\s+", " ", existing.lower().strip())
        # Exact match
        if q_normalized == existing_normalized:
            return CheckResult(
                check_type="duplicate",
                passed=False,
                score=0.0,
                details={"duplicate_of": existing[:100], "similarity": 1.0},
            )
        # Fuzzy match
        ratio = difflib.SequenceMatcher(None, q_normalized, existing_normalized).ratio()
        if ratio >= threshold:
            return CheckResult(
                check_type="duplicate",
                passed=False,
                score=1.0 - ratio,
                details={"duplicate_of": existing[:100], "similarity": round(ratio, 3)},
            )

    return CheckResult(
        check_type="duplicate",
        passed=True,
        score=1.0,
        details={},
    )


# ── Composite quality score ────────────────────────────────────────────

WEIGHTS = {
    "length": 0.25,
    "format": 0.20,
    "relevance": 0.30,
    "duplicate": 0.25,
}


def compute_quality_score(checks: list[CheckResult]) -> float:
    """Compute a weighted quality score from individual check results."""
    total_weight = 0.0
    weighted_sum = 0.0
    for check in checks:
        w = WEIGHTS.get(check.check_type, 0.1)
        weighted_sum += check.score * w
        total_weight += w
    return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


def evaluate_qa_pair(
    question: str,
    answer: str,
    existing_questions: list[str] | None = None,
) -> tuple[bool, float, list[CheckResult]]:
    """Run all quality checks on a Q&A pair.

    Returns
    -------
    passed : bool — True if all critical checks pass
    quality_score : float 0-1
    checks : list of individual CheckResult objects
    """
    checks = [
        check_length(question, answer),
        check_format(question),
        check_content_quality(question, answer),
    ]

    if existing_questions is not None:
        checks.append(check_duplicate(question, existing_questions))

    all_passed = all(c.passed for c in checks)
    quality_score = compute_quality_score(checks)

    return all_passed, quality_score, checks
