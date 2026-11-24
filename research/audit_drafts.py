"""Audit stored RAG draft context against a consistent, read-only SQLite snapshot."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from app.services.prompt_builder import format_evidence_for_prompt

REVIEW_LABELS = {"supported", "contradicted", "insufficient", "conflicting"}


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def valid_manifest(manifest: object) -> bool:
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int:
        return False
    if manifest["version"] != 1 or manifest.get("verification_status") != "retrieval_only":
        return False
    passages = manifest.get("passages")
    if not isinstance(passages, list) or not passages:
        return False
    for passage in passages:
        if not isinstance(passage, dict):
            return False
        if any(not isinstance(passage.get(key), str) or not passage[key]
               for key in ("chunk_id", "source_id", "chunk_sha256")):
            return False
        if not re.fullmatch(r"[0-9a-f]{64}", passage["chunk_sha256"]):
            return False
    value = manifest.get("context_sha256")
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_context(connection: sqlite3.Connection, project_id: str, metadata: str | None) -> dict:
    try:
        data = json.loads(metadata) if metadata is not None else None
    except (TypeError, json.JSONDecodeError):
        return {"status": "invalid_metadata"}
    if data is None:
        return {"status": "missing_manifest"}
    if not isinstance(data, dict):
        return {"status": "invalid_metadata"}
    if "retrieved_context" not in data:
        return {"status": "missing_manifest"}
    manifest = data["retrieved_context"]
    if not valid_manifest(manifest):
        return {"status": "invalid_manifest"}

    chunks, missing, changed = [], [], []
    for passage in manifest["passages"]:
        chunk = connection.execute(
            "SELECT c.content, s.filename FROM chunks c JOIN sources s ON s.id = c.source_id "
            "WHERE c.id = ? AND c.source_id = ? AND c.project_id = ? AND s.project_id = ?",
            (passage["chunk_id"], passage["source_id"], project_id, project_id),
        ).fetchone()
        if chunk is None:
            missing.append(passage["chunk_id"])
            continue
        content, filename = chunk
        if digest(content) != passage["chunk_sha256"]:
            changed.append(passage["chunk_id"])
        chunks.append(SimpleNamespace(source_filename=filename, content=content))
    if missing:
        return {"status": "missing_source", "missing_chunk_ids": missing, "changed_chunk_ids": changed}
    actual_digest = digest(format_evidence_for_prompt(chunks))
    if changed or actual_digest != manifest["context_sha256"]:
        return {"status": "context_mismatch", "changed_chunk_ids": changed,
                "assembled_context_matches": actual_digest == manifest["context_sha256"]}
    return {"status": "context_matches", "passage_count": len(chunks),
            "context_sha256": actual_digest}


def index_reviews(reviews: list[dict]) -> dict[str, dict]:
    indexed = {}
    for number, review in enumerate(reviews, start=1):
        if not isinstance(review, dict):
            raise ValueError(f"Review {number} must be an object.")
        pair_id = review.get("qa_pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Review {number} requires a qa_pair_id.")
        if pair_id in indexed:
            raise ValueError(f"Review {number} duplicates a qa_pair_id. Supply one adjudicated label per draft.")
        if not isinstance(review.get("label"), str) or review["label"] not in REVIEW_LABELS:
            raise ValueError(f"Review {number} has an unknown evidence label.")
        for key in ("draft_sha256", "context_sha256"):
            value = review.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"Review {number} requires a valid {key}.")
        indexed[pair_id] = review
    return indexed


def review_status(row: dict, review: dict | None) -> str:
    if review is None:
        return "unreviewed"
    if review["draft_sha256"] != row["draft_sha256"]:
        return "stale_review"
    if row["status"] != "context_matches":
        return "context_unverified"
    if review["context_sha256"] != row["context_sha256"]:
        return "stale_review"
    return review["label"]


def audit(database: Path, project_id: str, reviews: list[dict] | None = None) -> dict:
    indexed_reviews = index_reviews(reviews) if reviews is not None else None
    # mode=ro also prevents a typo from silently creating an empty database.
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        if connection.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
            raise ValueError("Project does not exist in the database.")
        records = connection.execute(
            'SELECT id, question, answer, metadata FROM qa_pairs WHERE project_id = ? ORDER BY id',
            (project_id,),
        )
        rows = []
        for pair_id, question, answer, metadata in records:
            result = check_context(connection, project_id, metadata)
            # Bind the report to the text revision without reproducing the text.
            revision = json.dumps([question, answer], ensure_ascii=False, separators=(",", ":"))
            rows.append({"qa_pair_id": pair_id, "draft_sha256": digest(revision), **result})
        report = {
            "protocol": "medqa-draft-context-audit-v1",
            "scope": "stored_retrieval_context_integrity",
            "project_id": project_id,
            "total_drafts": len(rows),
            "counts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "rows": rows,
        }
        if indexed_reviews is not None:
            unknown = set(indexed_reviews) - {row["qa_pair_id"] for row in rows}
            if unknown:
                raise ValueError(f"{len(unknown)} review IDs do not belong to the audited project.")
            for row in rows:
                row["review_status"] = review_status(row, indexed_reviews.get(row["qa_pair_id"]))
            counts = Counter(row["review_status"] for row in rows)
            current = sum(counts[label] for label in REVIEW_LABELS)
            report["evidence_reviews"] = {
                "counts": dict(sorted(counts.items())),
                "current_reviews": current,
                "coverage": current / len(rows) if rows else None,
                "supported_fraction_among_current_reviews": counts["supported"] / current if current else None,
            }
        return report
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--reviews", type=Path, help="Optional JSONL of adjudicated evidence-review labels")
    args = parser.parse_args()
    try:
        reviews = None
        if args.reviews is not None:
            reviews = [json.loads(line) for line in args.reviews.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        result = audit(args.database, args.project_id, reviews)
    except (sqlite3.Error, ValueError, OSError) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    matched = result["counts"].get("context_matches", 0)
    if args.reviews is not None:
        matched = result["evidence_reviews"]["counts"].get("supported", 0)
    return int(result["total_drafts"] == 0 or matched != result["total_drafts"])


if __name__ == "__main__":
    raise SystemExit(main())
