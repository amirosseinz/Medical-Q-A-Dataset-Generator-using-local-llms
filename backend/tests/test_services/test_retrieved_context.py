import json
import sys
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch

from app.services.export_service import QAPairForExport, to_json, to_jsonl
from app.services.prompt_builder import format_evidence_for_prompt, prepare_rag_prompts


def test_retrieval_context_records_rotated_prompt_order_and_exact_text():
    chunks = [
        SimpleNamespace(
            chunk_id="chunk-1", source_id="source-1", source_filename="A", content="First Δ", score=0.9,
        ),
        SimpleNamespace(
            chunk_id="chunk-2", source_id="source-2", source_filename="B", content="Second", score=0.8,
        ),
    ]
    retrieval = SimpleNamespace(
        has_evidence=True, chunks=chunks, citation_ids=lambda: ["chunk-1", "chunk-2"],
        citation_metadata=lambda: [],
    )
    rag = SimpleNamespace(retrieve_for_topic=lambda **kwargs: retrieval)
    with patch.dict(sys.modules, {"app.services.rag_service": rag}):
        original, rotated = prepare_rag_prompts(
            medical_terms=["topic"], domain="domain", question_types=["factual"],
            difficulty_levels=["beginner"], target_prompts=2, existing_questions=[],
            project_id="project", db=None, settings=SimpleNamespace(RAG_TOP_K=2, RAG_MIN_SCORE=0.25),
        )

    manifest = rotated.metadata["retrieved_context"]
    evidence = format_evidence_for_prompt(chunks[::-1])
    assert evidence in rotated.prompt
    assert manifest["verification_status"] == "retrieval_only"
    assert manifest["context_sha256"] == sha256(evidence.encode("utf-8")).hexdigest()
    assert manifest["context_sha256"] != original.metadata["retrieved_context"]["context_sha256"]
    assert manifest["passages"] == [
        {
            "chunk_id": chunk.chunk_id, "source_id": chunk.source_id,
            "chunk_sha256": sha256(chunk.content.encode("utf-8")).hexdigest(),
        }
        for chunk in chunks[::-1]
    ]
    assert rotated.metadata["citation_ids"] == ["chunk-1", "chunk-2"]


def test_json_and_jsonl_include_context_for_new_rows_and_accept_legacy_rows():
    context = {"version": 1, "verification_status": "retrieval_only", "passages": []}
    rows = [
        QAPairForExport("question", "answer", metadata={"rag": True, "retrieved_context": context}),
        QAPairForExport("old question", "old answer", metadata={"rag": True}),
        QAPairForExport("random question", "random answer"),
    ]
    for output in (json.loads(to_json(rows)), [json.loads(line) for line in to_jsonl(rows).splitlines()]):
        assert output[0]["retrieved_context"] == context
        assert all("retrieved_context" not in row for row in output[1:])
