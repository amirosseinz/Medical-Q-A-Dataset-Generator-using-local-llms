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
    """Validate question and answer lengths.

    Relaxed thresholds:
      - Question: 10-500 chars (ideal 20-200)
      - Answer: 20-3000 chars (ideal 50-1000)
    """
    q_len = len(question)
    a_len = len(answer)
    q_ok = 10 <= q_len <= 500
    a_ok = 20 <= a_len <= 3000

    # Score: 1.0 if in ideal range, degrades outside
    q_score = 1.0 if 20 <= q_len <= 200 else (0.7 if q_ok else 0.0)
    a_score = 1.0 if 50 <= a_len <= 1000 else (0.6 if a_ok else 0.0)
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

# Phrases that indicate the answer is referencing source material instead of
# being self-contained.  Checked case-insensitively.
FORBIDDEN_SOURCE_PHRASES = [
    "according to the text",
    "according to the passage",
    "according to the article",
    "according to the study",
    "according to the document",
    "according to the source",
    "the text states",
    "the text mentions",
    "the text describes",
    "the text explains",
    "the text indicates",
    "the text suggests",
    "the text notes",
    "the passage states",
    "the passage mentions",
    "the passage describes",
    "the article states",
    "the article mentions",
    "the study found",
    "the study states",
    "the study shows",
    "the study suggests",
    "the study reports",
    "as mentioned in the text",
    "as stated in the text",
    "as described in the text",
    "as noted in the text",
    "based on the passage",
    "based on the text",
    "based on the article",
    "based on the given text",
    "based on the provided text",
    "in the text",
    "in the passage",
    "in the given text",
    "in the provided text",
    "from the text",
    "from the passage",
    "the given text",
    "the provided text",
    "the above text",
    "this text",
    "this passage",
]


def check_content_quality(question: str, answer: str) -> CheckResult:
    """Check for generic AI refusal patterns and low-quality indicators.

    Source-reference phrases are penalised in score but no longer cause an
    automatic FAIL — the quality score already captures the issue and the
    configurable ``min_quality_score`` threshold handles gating.

    Citation artifacts ([Evidence N], [1], etc.) are treated as a hard fail
    since they indicate the answer is not self-contained.
    """
    issues: list[str] = []
    # Critical issues (will fail the check)
    critical = False

    for kw in BAD_KEYWORDS:
        if kw.lower() in answer.lower():
            issues.append(f"Contains bad keyword: {kw}")
            critical = True

    # Check for citation artifacts that should have been stripped by the parser
    # If they survive to quality checking, the answer is not self-contained
    _CITATION_RE = re.compile(
        r"\[Evidence\s*\d+\]|\[Ref(?:erence)?\s*\d+\]|\[Source\s*\d+\]|\(\s*Evidence\s*\d+\s*\)",
        re.IGNORECASE,
    )
    citation_matches = _CITATION_RE.findall(answer)
    if citation_matches:
        issues.append(f"Contains {len(citation_matches)} citation artifact(s): {citation_matches[:3]}")
        critical = True

    # Check for forbidden source-reference phrases (soft penalty only)
    answer_lower = answer.lower()
    question_lower = question.lower()
    source_ref_count = 0
    for phrase in FORBIDDEN_SOURCE_PHRASES:
        if phrase in answer_lower:
            source_ref_count += 1
        if phrase in question_lower:
            source_ref_count += 1
    if source_ref_count > 0:
        issues.append(f"Contains {source_ref_count} source-reference phrase(s)")

    # Check for repetitive content
    words = answer.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            issues.append(f"Low word diversity: {unique_ratio:.2f}")
            critical = True

    passed = not critical
    # Score penalty: BAD_KEYWORDS = 0.3 each, citations = 0.4, source refs = 0.1 per phrase, low diversity = 0.3
    penalty = 0.0
    if any("bad keyword" in i.lower() for i in issues):
        penalty += 0.3 * sum(1 for i in issues if "bad keyword" in i.lower())
    if any("citation artifact" in i.lower() for i in issues):
        penalty += 0.4
    penalty += source_ref_count * 0.1
    if any("word diversity" in i.lower() for i in issues):
        penalty += 0.3
    score = max(0.0, 1.0 - penalty)

    return CheckResult(
        check_type="relevance",
        passed=passed,
        score=score,
        details={"issues": issues},
    )


def check_duplicate(
    question: str,
    existing_questions: list[str],
    threshold: float = 0.92,
) -> CheckResult:
    """Check if question is a near-duplicate of any existing question.

    Threshold raised to 0.92 (from 0.85) to allow more semantic variation \u2014
    only truly paraphrased questions are rejected.  The over-generation
    strategy compensates by producing 3x more prompts.
    """
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
    min_quality_score: float = 0.0,
) -> tuple[bool, float, list[CheckResult]]:
    """Run all quality checks on a Q&A pair.

    Returns
    -------
    passed : bool — True if no critical checks fail AND score >= min_quality_score
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

    # A pair passes if:
    #  1. No critical checks fail (length, duplicate — format and content-quality
    #     are "soft" and primarily influence the score)
    #  2. Quality score meets the configurable threshold
    critical_types = {"length", "duplicate", "relevance"}
    critical_passed = all(c.passed for c in checks if c.check_type in critical_types)
    quality_score = compute_quality_score(checks)
    passed = critical_passed and quality_score >= min_quality_score

    # Transparent scoring log — always emitted at DEBUG level
    breakdown = " | ".join(
        f"{c.check_type}={c.score:.2f}{'*' if not c.passed else ''}"
        for c in checks
    )
    logger.debug(
        "QA score=%.3f (pass=%s, crit=%s, min=%.2f): %s",
        quality_score, passed, critical_passed, min_quality_score, breakdown,
    )

    return passed, quality_score, checks
