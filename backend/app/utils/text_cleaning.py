"""Text cleaning utilities — ported from original app.py clean_text()."""
import re


def clean_text(text: str) -> str:
    """Clean and prepare raw text for Q&A generation.

    Handles common PDF extraction artifacts, citation markers,
    hyphenated line-breaks, and excessive whitespace.
    """
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove page headers/footers
    text = re.sub(r"Page \d+ of \d+", "", text)
    # Remove citation numbers [1], [2,3], etc.
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    # Remove PDF CID references (cid:XXX)
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Rejoin hyphenated words split across lines
    text = re.sub(r"-\s*\n", "", text)
    # Collapse multiple newlines to single
    text = re.sub(r"\n+", "\n", text).strip()
    # Remove very short lines (likely headers/footers)
    lines = text.split("\n")
    lines = [l for l in lines if len(l.strip()) > 10]
    return "\n".join(lines)
