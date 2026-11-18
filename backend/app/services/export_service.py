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
import math
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QAPairForExport:
    """Minimal Q&A pair representation for export.

    RAG fields (criterion 8): when a pair was generated via RAG, the metadata
    dict contains citation_chunk_ids, citations (with scores and previews),
    and query_term. These are surfaced in the export so every dataset row
    carries its evidence provenance.
    """
    question: str
    answer: str
    source_type: str = ""
    quality_score: float | None = None
    validation_status: str = ""
    metadata: dict | None = None

    # ── Convenience accessors for RAG evidence ──

    @property
    def is_rag(self) -> bool:
        return bool(self.metadata and self.metadata.get("rag"))

    @property
    def citation_chunk_ids(self) -> list[str]:
        if self.metadata:
            return self.metadata.get("citation_chunk_ids", [])
        return []

    @property
    def retrieval_scores(self) -> list[float]:
        if self.metadata:
            return self.metadata.get("retrieval_scores", [])
        return []

    @property
    def citations(self) -> list[dict]:
        if self.metadata:
            return self.metadata.get("citations", [])
        return []

    @property
    def query_term(self) -> str:
        if self.metadata:
            return self.metadata.get("query_term", "")
        return ""


def split_dataset(
    pairs: list[QAPairForExport],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    split_strategy: str = "row",
) -> dict[str, list[QAPairForExport]]:
    """Split pairs into train/val/test sets."""
    if split_strategy == "source":
        return split_by_source(pairs, train, val, test, seed)
    if split_strategy != "row":
        raise ValueError("Unknown split strategy.")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def split_by_source(
    pairs: list[QAPairForExport], train: float, val: float, test: float, seed: int,
) -> dict[str, list[QAPairForExport]]:
    """Keep connected cited-source components intact, including multi-source rows."""
    fractions = {"train": train, "val": val, "test": test}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions.values()):
        raise ValueError("Split fractions must be finite values between zero and one.")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to one.")

    parents = list(range(len(pairs)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, pair in enumerate(pairs):
        metadata = pair.metadata or {}
        context = metadata.get("retrieved_context")
        passages = context.get("passages") if isinstance(context, dict) else metadata.get("citations")
        if not isinstance(passages, list) or not passages:
            raise ValueError(f"Row {index + 1} requires complete cited source IDs for source splitting.")
        for passage in passages:
            source_id = passage.get("source_id") if isinstance(passage, dict) else None
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"Row {index + 1} has an incomplete cited source ID.")
            if source_id in owners:
                parents[root(index)] = root(owners[source_id])
            else:
                owners[source_id] = index

    components = {}
    for index, pair in enumerate(pairs):
        components.setdefault(root(index), []).append(pair)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    # Place large components first. Randomisation breaks ties reproducibly.
    groups.sort(key=len, reverse=True)
    splits = {name: [] for name in fractions}
    targets = {name: len(pairs) * fraction for name, fraction in fractions.items() if fraction > 0}
    for group in groups:
        selected = max(targets, key=lambda name: targets[name] - len(splits[name]))
        splits[selected].extend(group)
    return splits


# ── Format converters ──────────────────────────────────────────────────

def to_csv(pairs: list[QAPairForExport], include_metadata: bool = False) -> str:
    """Standard CSV: question, answer, source, quality_score, status, + RAG evidence fields."""
    buf = io.StringIO()
    fieldnames = [
        "question", "answer", "source_type", "quality_score", "validation_status",
        "rag", "query_term", "citation_chunk_ids", "retrieval_scores",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for p in pairs:
        writer.writerow({
            "question": p.question,
            "answer": p.answer,
            "source_type": p.source_type,
            "quality_score": p.quality_score,
            "validation_status": p.validation_status,
            "rag": p.is_rag,
            "query_term": p.query_term,
            "citation_chunk_ids": ";".join(p.citation_chunk_ids) if p.citation_chunk_ids else "",
            "retrieval_scores": ";".join(f"{s:.4f}" for s in p.retrieval_scores) if p.retrieval_scores else "",
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
    """Full JSON array with all fields including RAG evidence."""
    data = []
    for p in pairs:
        obj = {
            "question": p.question,
            "answer": p.answer,
            "source_type": p.source_type,
            "quality_score": p.quality_score,
            "validation_status": p.validation_status,
        }
        if p.is_rag:
            obj["rag"] = True
            obj["query_term"] = p.query_term
            obj["citation_chunk_ids"] = p.citation_chunk_ids
            obj["retrieval_scores"] = p.retrieval_scores
            obj["citations"] = p.citations
            if retrieved_context := (p.metadata or {}).get("retrieved_context"):
                obj["retrieved_context"] = retrieved_context
        data.append(obj)
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_jsonl(pairs: list[QAPairForExport]) -> str:
    """JSONL — one JSON object per line, with RAG evidence when available."""
    lines = []
    for p in pairs:
        obj = {"question": p.question, "answer": p.answer}
        if p.is_rag:
            obj["rag"] = True
            obj["query_term"] = p.query_term
            obj["citation_chunk_ids"] = p.citation_chunk_ids
            obj["retrieval_scores"] = p.retrieval_scores
            if retrieved_context := (p.metadata or {}).get("retrieved_context"):
                obj["retrieved_context"] = retrieved_context
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
    """Parquet format bytes (requires pandas + pyarrow), includes RAG evidence."""
    import pandas as pd
    data = []
    for p in pairs:
        row = {
            "question": p.question,
            "answer": p.answer,
            "source_type": p.source_type,
            "quality_score": p.quality_score,
            "validation_status": p.validation_status,
            "rag": p.is_rag,
            "query_term": p.query_term,
            "citation_chunk_ids": json.dumps(p.citation_chunk_ids) if p.citation_chunk_ids else "",
            "retrieval_scores": json.dumps(p.retrieval_scores) if p.retrieval_scores else "",
        }
        data.append(row)
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
