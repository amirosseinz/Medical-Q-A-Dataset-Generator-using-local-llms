"""Export service — multi-format dataset export.

Preserves original CSV/JSON export logic and adds:
  - JSONL (one object per line)
  - Parquet (via pandas + pyarrow)
  - Alpaca format (instruction/input/output)
  - OpenAI fine-tune format (messages array)
  - Train/val/test split
"""
from __future__ import annotations

import csv
import io
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QAPairForExport:
    """Minimal Q&A pair representation for export."""
    question: str
    answer: str
    source_type: str = ""
    quality_score: float | None = None
    validation_status: str = ""
    metadata: dict | None = None


def split_dataset(
    pairs: list[QAPairForExport],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> dict[str, list[QAPairForExport]]:
    """Split pairs into train/val/test sets."""
    shuffled = list(pairs)
    random.seed(seed)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


# ── Format converters ──────────────────────────────────────────────────

def to_csv(pairs: list[QAPairForExport], include_metadata: bool = False) -> str:
    """Standard CSV: question, answer, source, quality_score, status."""
    buf = io.StringIO()
    fieldnames = ["question", "answer", "source_type", "quality_score", "validation_status"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for p in pairs:
        writer.writerow({
            "question": p.question,
            "answer": p.answer,
            "source_type": p.source_type,
            "quality_score": p.quality_score,
            "validation_status": p.validation_status,
        })
    return buf.getvalue()


def to_training_csv(pairs: list[QAPairForExport]) -> str:
    """Single-column training CSV matching the original format:
    ``question: ... answer: ...`` with no header."""
    buf = io.StringIO()
    for p in pairs:
        buf.write(f"question: {p.question} answer: {p.answer}\n")
    return buf.getvalue()


def to_json(pairs: list[QAPairForExport]) -> str:
    """Full JSON array with all fields."""
    data = [
        {
            "question": p.question,
            "answer": p.answer,
            "source_type": p.source_type,
            "quality_score": p.quality_score,
            "validation_status": p.validation_status,
        }
        for p in pairs
    ]
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_jsonl(pairs: list[QAPairForExport]) -> str:
    """JSONL — one JSON object per line."""
    lines = []
    for p in pairs:
        obj = {"question": p.question, "answer": p.answer}
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def to_alpaca(pairs: list[QAPairForExport]) -> str:
    """Alpaca instruction format JSONL."""
    lines = []
    for p in pairs:
        obj = {
            "instruction": p.question,
            "input": "",
            "output": p.answer,
        }
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def to_openai(pairs: list[QAPairForExport]) -> str:
    """OpenAI fine-tuning JSONL format."""
    lines = []
    for p in pairs:
        obj = {
            "messages": [
                {"role": "system", "content": "You are a knowledgeable medical assistant."},
                {"role": "user", "content": p.question},
                {"role": "assistant", "content": p.answer},
            ]
        }
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def to_parquet_bytes(pairs: list[QAPairForExport]) -> bytes:
    """Parquet format bytes (requires pandas + pyarrow)."""
    import pandas as pd
    data = [
        {
            "question": p.question,
            "answer": p.answer,
            "source_type": p.source_type,
            "quality_score": p.quality_score,
            "validation_status": p.validation_status,
        }
        for p in pairs
    ]
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()


# ── Dispatcher ─────────────────────────────────────────────────────────

EXPORTERS = {
    "csv": to_csv,
    "json": to_json,
    "jsonl": to_jsonl,
    "alpaca": to_alpaca,
    "openai": to_openai,
}

BINARY_EXPORTERS = {
    "parquet": to_parquet_bytes,
}

EXTENSIONS = {
    "csv": ".csv",
    "json": ".json",
    "jsonl": ".jsonl",
    "alpaca": ".jsonl",
    "openai": ".jsonl",
    "parquet": ".parquet",
}


def export_dataset(
    pairs: list[QAPairForExport],
    fmt: str,
    output_dir: Path,
    base_name: str = "dataset",
) -> Path:
    """Export pairs to the given format, write to output_dir, return path."""
    ext = EXTENSIONS.get(fmt, ".txt")
    output_path = output_dir / f"{base_name}{ext}"

    if fmt in BINARY_EXPORTERS:
        data = BINARY_EXPORTERS[fmt](pairs)
        output_path.write_bytes(data)
    elif fmt in EXPORTERS:
        data = EXPORTERS[fmt](pairs)
        output_path.write_text(data, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    logger.info(f"Exported {len(pairs)} pairs to {output_path} ({fmt})")
    return output_path
