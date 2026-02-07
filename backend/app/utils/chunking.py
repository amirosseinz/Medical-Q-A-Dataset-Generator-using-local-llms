"""Text chunking strategies — word-count, paragraph-aware, and section-aware."""
from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)


def chunk_by_word_count(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_words: int = 20,
) -> list[str]:
    """Split text into overlapping chunks based on word count.

    Ported from original ``split_into_chunks()``.
    """
    if not text:
        return []
    words = text.split()
    chunks: list[str] = []
    i = 0
    step = max(chunk_size - overlap, 1)
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
        i += step
    return chunks


def chunk_by_paragraph(
    text: str,
    target_size: int = 500,
    min_words: int = 20,
) -> list[str]:
    """Split text into chunks that respect paragraph boundaries.

    Paragraphs are detected by double-newlines or indentation changes.
    Small consecutive paragraphs are merged until *target_size* is reached.
    """
    if not text:
        return []
    # Split on double newlines or lines that look like section breaks
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_chunk_parts: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_word_count + para_words > target_size and current_chunk_parts:
            merged = "\n\n".join(current_chunk_parts)
            if len(merged.split()) >= min_words:
                chunks.append(merged)
            current_chunk_parts = [para]
            current_word_count = para_words
        else:
            current_chunk_parts.append(para)
            current_word_count += para_words

    # Flush remaining
    if current_chunk_parts:
        merged = "\n\n".join(current_chunk_parts)
        if len(merged.split()) >= min_words:
            chunks.append(merged)

    return chunks


def chunk_by_section(
    text: str,
    target_size: int = 500,
    min_words: int = 20,
) -> list[str]:
    """Split text by detected section headings, then sub-chunk large sections.

    Uses heuristics to detect headings (ALL CAPS lines, numbered sections,
    lines ending with a colon).
    """
    if not text:
        return []

    heading_pattern = re.compile(
        r"^(?:"
        r"[A-Z][A-Z\s]{4,}$|"         # ALL CAPS lines (at least 5 chars)
        r"\d+\.[\d.]*\s+.+$|"          # Numbered sections: 1. Title, 2.1 Sub
        r".{5,60}:\s*$"                 # Lines ending with colon
        r")",
        re.MULTILINE,
    )

    lines = text.split("\n")
    sections: list[str] = []
    current_section_lines: list[str] = []

    for line in lines:
        if heading_pattern.match(line.strip()) and current_section_lines:
            sections.append("\n".join(current_section_lines))
            current_section_lines = [line]
        else:
            current_section_lines.append(line)

    if current_section_lines:
        sections.append("\n".join(current_section_lines))

    # Sub-chunk large sections using paragraph chunking
    chunks: list[str] = []
    for section in sections:
        word_count = len(section.split())
        if word_count <= target_size:
            if word_count >= min_words:
                chunks.append(section.strip())
        else:
            sub_chunks = chunk_by_paragraph(section, target_size, min_words)
            chunks.extend(sub_chunks)

    return chunks


def create_chunks(
    text: str,
    strategy: str = "word_count",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Dispatch to the appropriate chunking strategy.

    Parameters
    ----------
    strategy : word_count | paragraph | section
    """
    if strategy == "paragraph":
        return chunk_by_paragraph(text, target_size=chunk_size)
    elif strategy == "section":
        return chunk_by_section(text, target_size=chunk_size)
    else:
        return chunk_by_word_count(text, chunk_size=chunk_size, overlap=overlap)
