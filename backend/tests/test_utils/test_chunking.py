"""Tests for text chunking utilities."""
import pytest
from app.utils.chunking import chunk_by_word_count, chunk_by_paragraph, chunk_by_section, create_chunks


class TestChunkByWordCount:
    def test_empty_text(self):
        assert chunk_by_word_count("") == []

    def test_short_text_below_min_words(self):
        assert chunk_by_word_count("hello world", min_words=20) == []

    def test_basic_chunking(self):
        words = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_by_word_count(words, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 30

    def test_overlap(self):
        words = " ".join([f"word{i}" for i in range(60)])
        chunks = chunk_by_word_count(words, chunk_size=30, overlap=10)
        # With overlap, chunks should share some words
        if len(chunks) >= 2:
            first_words = set(chunks[0].split()[-10:])
            second_words = set(chunks[1].split()[:10])
            assert len(first_words & second_words) > 0


class TestChunkByParagraph:
    def test_empty_text(self):
        assert chunk_by_paragraph("") == []

    def test_single_paragraph(self):
        text = " ".join(["word"] * 50)
        chunks = chunk_by_paragraph(text, target_size=100)
        assert len(chunks) == 1

    def test_multiple_paragraphs(self):
        para1 = " ".join(["alpha"] * 100)
        para2 = " ".join(["beta"] * 100)
        text = f"{para1}\n\n{para2}"
        chunks = chunk_by_paragraph(text, target_size=120)
        assert len(chunks) >= 2


class TestCreateChunks:
    def test_dispatcher_word_count(self):
        text = " ".join(["word"] * 100)
        chunks = create_chunks(text, strategy="word_count", chunk_size=30)
        assert len(chunks) > 1

    def test_dispatcher_paragraph(self):
        text = "Para one " * 30 + "\n\n" + "Para two " * 30
        chunks = create_chunks(text, strategy="paragraph", chunk_size=40)
        assert len(chunks) >= 1
