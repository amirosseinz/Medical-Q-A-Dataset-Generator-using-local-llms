"""Tests for quality checker service."""
import pytest
from app.services.quality_checker import (
    check_length,
    check_format,
    check_content_quality,
    check_duplicate,
    evaluate_qa_pair,
    compute_quality_score,
)


class TestCheckLength:
    def test_valid_lengths(self):
        result = check_length("What is diabetes mellitus?", "Diabetes mellitus is a chronic condition affecting blood sugar regulation.")
        assert result.passed is True
        assert result.score > 0.5

    def test_short_question(self):
        result = check_length("What?", "A valid answer with enough content to pass the minimum.")
        assert result.passed is False

    def test_short_answer(self):
        result = check_length("What is the treatment for hypertension?", "Medication.")
        assert result.passed is False


class TestCheckFormat:
    def test_question_with_mark(self):
        result = check_format("What is the recommended dosage?")
        assert result.passed is True

    def test_question_without_mark_but_starts_with_qword(self):
        result = check_format("Explain the mechanism of action")
        assert result.passed is True

    def test_not_a_question(self):
        result = check_format("The patient was given medication")
        assert result.passed is False


class TestCheckContentQuality:
    def test_good_content(self):
        result = check_content_quality(
            "What are the risk factors?",
            "Major risk factors include age, family history, and obesity."
        )
        assert result.passed is True

    def test_ai_refusal(self):
        result = check_content_quality(
            "What is the treatment?",
            "I am a language model and cannot provide medical advice."
        )
        assert result.passed is False


class TestCheckDuplicate:
    def test_no_duplicate(self):
        result = check_duplicate("What is diabetes?", ["What is cancer?", "How to treat flu?"])
        assert result.passed is True

    def test_exact_duplicate(self):
        result = check_duplicate("What is diabetes?", ["What is diabetes?"])
        assert result.passed is False

    def test_fuzzy_duplicate(self):
        result = check_duplicate(
            "What is the treatment for diabetes?",
            ["What is the treatment for diabetes mellitus?"],
            threshold=0.80,
        )
        # Similarity should be high
        assert result.score < 0.5  # Low score means high similarity


class TestEvaluateQAPair:
    def test_good_pair(self):
        passed, score, checks = evaluate_qa_pair(
            "What are the symptoms of heart failure?",
            "Common symptoms of heart failure include shortness of breath, fatigue, swelling in legs, and rapid heartbeat.",
        )
        assert passed is True
        assert score > 0.5
        assert len(checks) == 3  # length, format, relevance (no duplicate list)

    def test_bad_pair(self):
        passed, score, checks = evaluate_qa_pair("Hi", "No")
        assert passed is False
