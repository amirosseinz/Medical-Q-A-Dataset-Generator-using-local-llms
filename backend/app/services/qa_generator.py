"""Core Q&A generation orchestrator.

This is the main pipeline that coordinates all services:
  1. Process uploaded documents (PDF, XML, DOCX)
  2. Fetch PubMed abstracts
  3. Chunk all text
  4. Extract MedQuAD pairs from XML
  5. Generate AI Q&A pairs via Ollama
  6. Run quality checks
  7. Store results in the database

Ported from the original ``DatasetGenerator.generate_dataset()`` (lines 395-556).
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone

import redis
import json
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Source, Chunk, QAPair, GenerationJob, QualityCheck
from app.services.document_processor import (
    extract_text_from_pdf,
    extract_medquad_pairs,
    extract_text,
)
from app.services.pubmed_service import fetch_pubmed_abstracts
from app.services.ollama_service import generate_qa_batch
from app.services.quality_checker import evaluate_qa_pair
from app.utils.text_cleaning import clean_text
from app.utils.chunking import create_chunks
from app.utils.prompts import build_prompt

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Orchestrates the full dataset generation workflow."""

    def __init__(self, db: Session, job_id: str, project_id: str, config: dict):
        self.db = db
        self.job_id = job_id
        self.project_id = project_id
        self.config = config
        self.settings = get_settings()
        self._redis: redis.Redis | None = None

    @property
    def redis_client(self) -> redis.Redis | None:
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.settings.REDIS_URL)
            except Exception:
                logger.warning("Could not connect to Redis for progress updates")
        return self._redis

    def update_progress(self, message: str, percentage: int):
        """Update job progress in DB and publish to Redis for WebSocket clients."""
        try:
            job = self.db.query(GenerationJob).filter(GenerationJob.id == self.job_id).first()
            if job:
                job.progress_pct = percentage
                job.current_message = message
                if percentage > 0 and job.status == "queued":
                    job.status = "in_progress"
                    job.started_at = datetime.now(timezone.utc)
                self.db.commit()
        except Exception as e:
            logger.error(f"DB progress update failed: {e}")

        # Publish to Redis channel for WebSocket
        try:
            if self.redis_client:
                payload = json.dumps({
                    "job_id": self.job_id,
                    "percentage": percentage,
                    "message": message,
                    "status": "in_progress" if 0 < percentage < 100 else (
                        "completed" if percentage >= 100 else "queued"
                    ),
                })
                self.redis_client.publish(f"job:{self.job_id}", payload)
        except Exception as e:
            logger.debug(f"Redis publish failed: {e}")

        logger.info(f"Job {self.job_id[:8]}: {message} ({percentage}%)")

    def is_cancelled(self) -> bool:
        """Check if the job has been cancelled."""
        try:
            job = self.db.query(GenerationJob).filter(GenerationJob.id == self.job_id).first()
            return job is not None and job.status == "cancelled"
        except Exception:
            return False

    async def run(self) -> dict:
        """Execute the full generation pipeline."""
        try:
            self.update_progress("Starting dataset generation...", 2)

            medical_terms = [
                t.strip() for t in self.config.get("medical_terms", "").split(",") if t.strip()
            ]
            domain = self.config.get("medical_domain", "medical conditions")

            if not medical_terms:
                # Fall back to project domain
                from app.models import Project
                project = self.db.query(Project).filter(Project.id == self.project_id).first()
                if project and project.domain:
                    medical_terms = [t.strip() for t in project.domain.split(",") if t.strip()]
                if not medical_terms:
                    medical_terms = ["medical conditions"]

            # ── Step 1: Process MedQuAD XML files ──────────────────────
            medquad_pairs = []
            sources = self.db.query(Source).filter(
                Source.project_id == self.project_id,
                Source.file_type == "xml",
            ).all()

            if sources:
                self.update_progress("Processing MedQuAD XML files...", 8)
                for src in sources:
                    if src.filepath:
                        pairs = extract_medquad_pairs(src.filepath, medical_terms)
                        for p in pairs:
                            medquad_pairs.append({
                                "question": p.question,
                                "answer": p.answer,
                                "source_type": "medquad",
                                "source_file": p.source_file,
                            })
                        src.processing_status = "completed"
                self.db.commit()
                logger.info(f"Extracted {len(medquad_pairs)} MedQuAD pairs")

            if self.is_cancelled():
                return self._finish_cancelled()

            # ── Step 2: Extract text from PDF/DOCX files ───────────────
            all_chunks_with_meta: list[tuple[str, str, str | None]] = []  # (chunk, source_type, filename)

            doc_sources = self.db.query(Source).filter(
                Source.project_id == self.project_id,
                Source.file_type.in_(["pdf", "docx"]),
            ).all()

            if doc_sources:
                self.update_progress("Extracting text from documents...", 15)
                for i, src in enumerate(doc_sources):
                    if src.filepath:
                        doc = extract_text(src.filepath)
                        if doc.text:
                            cleaned = clean_text(doc.text)
                            chunks = create_chunks(
                                cleaned,
                                strategy=self.config.get("chunking_strategy", "word_count"),
                                chunk_size=self.config.get("chunk_size", 500),
                                overlap=self.config.get("chunk_overlap", 50),
                            )
                            # Store chunks in DB
                            for idx, chunk_text in enumerate(chunks):
                                chunk_record = Chunk(
                                    source_id=src.id,
                                    project_id=self.project_id,
                                    chunk_index=idx,
                                    content=chunk_text,
                                    word_count=len(chunk_text.split()),
                                )
                                self.db.add(chunk_record)
                                all_chunks_with_meta.append((chunk_text, "PDF", src.filename))
                            src.processing_status = "completed"
                        else:
                            src.processing_status = "failed"
                            src.error_message = doc.error or "No text extracted"
                    pct = 15 + int((i + 1) / len(doc_sources) * 15)
                    self.update_progress(f"Processed document {i + 1}/{len(doc_sources)}", pct)
                self.db.commit()

            if self.is_cancelled():
                return self._finish_cancelled()

            # ── Step 3: Fetch PubMed abstracts ─────────────────────────
            if self.config.get("use_pubmed", True) and medical_terms:
                self.update_progress("Fetching PubMed abstracts...", 35)
                pubmed_result = await fetch_pubmed_abstracts(
                    medical_terms=medical_terms,
                    email=self.config.get("email", self.settings.PUBMED_EMAIL),
                    retmax=self.config.get("pubmed_retmax", 1000),
                )
                if pubmed_result.abstracts:
                    combined_text = "\n\n".join(pubmed_result.abstracts)
                    cleaned = clean_text(combined_text)
                    chunks = create_chunks(
                        cleaned,
                        strategy=self.config.get("chunking_strategy", "word_count"),
                        chunk_size=self.config.get("chunk_size", 500),
                        overlap=self.config.get("chunk_overlap", 50),
                    )
                    # Create a "virtual" source for PubMed
                    pubmed_source = Source(
                        project_id=self.project_id,
                        filename="pubmed_abstracts",
                        file_type="pubmed",
                        processing_status="completed",
                        metadata_json={
                            "total_ids": pubmed_result.total_ids_found,
                            "total_fetched": pubmed_result.total_fetched,
                        },
                    )
                    self.db.add(pubmed_source)
                    self.db.flush()

                    for idx, chunk_text in enumerate(chunks):
                        chunk_record = Chunk(
                            source_id=pubmed_source.id,
                            project_id=self.project_id,
                            chunk_index=idx,
                            content=chunk_text,
                            word_count=len(chunk_text.split()),
                        )
                        self.db.add(chunk_record)
                        all_chunks_with_meta.append((chunk_text, "PubMed", None))
                    self.db.commit()
                    logger.info(f"PubMed: {len(chunks)} chunks from {pubmed_result.total_fetched} abstracts")

            if self.is_cancelled():
                return self._finish_cancelled()

            # ── Step 4: Store MedQuAD pairs in DB ──────────────────────
            self.update_progress("Storing MedQuAD pairs...", 50)
            existing_questions: list[str] = []
            for mp in medquad_pairs:
                passed, score, checks = evaluate_qa_pair(mp["question"], mp["answer"], existing_questions)
                qa = QAPair(
                    project_id=self.project_id,
                    question=mp["question"],
                    answer=mp["answer"],
                    source_type="medquad",
                    quality_score=score,
                    validation_status="approved" if passed else "pending",
                    metadata_json={"source_file": mp.get("source_file")},
                )
                self.db.add(qa)
                self.db.flush()  # Ensure qa.id is populated before creating QualityChecks
                existing_questions.append(mp["question"])

                # Store quality checks
                for check in checks:
                    qc = QualityCheck(
                        qa_pair_id=qa.id,
                        check_type=check.check_type,
                        passed=check.passed,
                        score=check.score,
                        details=check.details,
                    )
                    self.db.add(qc)
            self.db.commit()

            # ── Step 5: Generate AI Q&A pairs ──────────────────────────
            remaining_pairs = max(
                0, self.config.get("target_pairs", 1000) - len(medquad_pairs)
            )

            if remaining_pairs > 0 and all_chunks_with_meta and self.config.get("use_ollama", True):
                self.update_progress("Preparing AI generation...", 55)

                # Build prompts for each chunk
                question_types = self.config.get("question_types", ["factual", "reasoning"])
                difficulty_levels = self.config.get("difficulty_levels", ["intermediate"])
                model_name = self.config.get("ollama_model", "llama3")
                ollama_url = self.config.get("ollama_url", self.settings.OLLAMA_URL)

                # Limit chunks to process — use proportional sampling so every
                # source type (PDF, PubMed, etc.) is represented fairly instead
                # of being drowned out by whichever has the most chunks.
                target_chunks = min(len(all_chunks_with_meta), int(remaining_pairs * 1.5))

                # Group chunks by source type
                from collections import defaultdict
                source_buckets: dict[str, list[tuple[str, str, str | None]]] = defaultdict(list)
                for item in all_chunks_with_meta:
                    source_buckets[item[1]].append(item)

                # Shuffle each bucket independently
                for bucket in source_buckets.values():
                    random.shuffle(bucket)

                # Allocate slots proportionally, but guarantee at least 1 from
                # each source type (as long as target_chunks allows it)
                n_types = len(source_buckets)
                min_per_type = min(1, target_chunks // max(n_types, 1))
                remaining_slots = target_chunks
                chunks_to_process: list[tuple[str, str, str | None]] = []

                # First pass: guarantee minimum from each type
                for stype, bucket in source_buckets.items():
                    take = min(min_per_type, len(bucket))
                    chunks_to_process.extend(bucket[:take])
                    remaining_slots -= take

                # Second pass: fill remaining slots proportionally
                if remaining_slots > 0:
                    total_remaining_in_buckets = sum(
                        len(b) - min_per_type for b in source_buckets.values() if len(b) > min_per_type
                    )
                    for stype, bucket in source_buckets.items():
                        leftover = bucket[min_per_type:]
                        if not leftover or total_remaining_in_buckets == 0:
                            continue
                        share = max(1, int(remaining_slots * len(leftover) / total_remaining_in_buckets))
                        take = min(share, len(leftover), remaining_slots)
                        chunks_to_process.extend(leftover[:take])
                        remaining_slots -= take
                        if remaining_slots <= 0:
                            break

                # Final shuffle so PDF and PubMed chunks are interleaved
                random.shuffle(chunks_to_process)
                logger.info(
                    f"Chunk sampling: {target_chunks} target from "
                    + ", ".join(f"{k}={len(v)}" for k, v in source_buckets.items())
                    + f" → selected {len(chunks_to_process)}"
                )

                chunks_with_prompts: list[tuple[str, str]] = []
                chunk_meta_map: list[tuple[str, str | None]] = []  # (source_type, filename)

                for i, (chunk_text, source_type, filename) in enumerate(chunks_to_process):
                    qt = question_types[i % len(question_types)]
                    dl = difficulty_levels[i % len(difficulty_levels)]
                    prompt = build_prompt(qt, chunk_text, domain, dl)
                    chunks_with_prompts.append((chunk_text, prompt))
                    chunk_meta_map.append((source_type, filename))

                # Progress callback
                async def on_progress(completed: int, total: int, pair):
                    pct = 60 + int((completed / total) * 30)
                    self.update_progress(
                        f"Generating AI Q&A: {completed}/{total} chunks processed", pct
                    )

                self.update_progress("Generating AI Q&A pairs with Ollama...", 60)
                ai_pairs = await generate_qa_batch(
                    chunks_with_prompts=chunks_with_prompts,
                    ollama_url=ollama_url,
                    model_name=model_name,
                    temperature=self.config.get("temperature", 0.7),
                    top_p=self.config.get("top_p", 0.9),
                    max_concurrent=self.config.get("max_workers", 3),
                    max_pairs=remaining_pairs,
                    progress_callback=on_progress,
                )

                # Store AI pairs with quality checks
                self.update_progress("Running quality checks on AI pairs...", 92)
                for i, pair in enumerate(ai_pairs):
                    if self.is_cancelled():
                        break
                    source_type, filename = chunk_meta_map[i] if i < len(chunk_meta_map) else ("unknown", None)
                    qa_source = f"{'pdf' if source_type == 'PDF' else 'pubmed'}_ollama"

                    passed, score, checks = evaluate_qa_pair(
                        pair.question, pair.answer, existing_questions
                    )

                    if not passed:
                        continue

                    qa = QAPair(
                        project_id=self.project_id,
                        question=pair.question,
                        answer=pair.answer,
                        source_type=qa_source,
                        model_used=model_name,
                        quality_score=score,
                        validation_status="pending",
                        metadata_json={
                            "original_source": source_type,
                            "original_file": filename,
                        },
                    )
                    self.db.add(qa)
                    self.db.flush()  # Ensure qa.id is populated before creating QualityChecks
                    existing_questions.append(pair.question)

                    for check in checks:
                        qc = QualityCheck(
                            qa_pair_id=qa.id,
                            check_type=check.check_type,
                            passed=check.passed,
                            score=check.score,
                            details=check.details,
                        )
                        self.db.add(qc)

                self.db.commit()

            # ── Step 6: Finalize ───────────────────────────────────────
            self.update_progress("Finalizing dataset...", 97)

            total_pairs = self.db.query(QAPair).filter(
                QAPair.project_id == self.project_id
            ).count()

            # Build summary
            from sqlalchemy import func
            source_counts = dict(
                self.db.query(QAPair.source_type, func.count(QAPair.id))
                .filter(QAPair.project_id == self.project_id)
                .group_by(QAPair.source_type)
                .all()
            )
            status_counts = dict(
                self.db.query(QAPair.validation_status, func.count(QAPair.id))
                .filter(QAPair.project_id == self.project_id)
                .group_by(QAPair.validation_status)
                .all()
            )

            result = {
                "total_pairs": total_pairs,
                "sources": source_counts,
                "quality_breakdown": status_counts,
            }

            # Mark job as completed
            job = self.db.query(GenerationJob).filter(GenerationJob.id == self.job_id).first()
            if job:
                job.status = "completed"
                job.progress_pct = 100
                job.current_message = "Dataset generation completed!"
                job.completed_at = datetime.now(timezone.utc)
            self.db.commit()

            self.update_progress("Dataset generation completed!", 100)

            # Publish completion to Redis
            try:
                if self.redis_client:
                    payload = json.dumps({
                        "job_id": self.job_id,
                        "percentage": 100,
                        "message": "Dataset generation completed!",
                        "status": "completed",
                        "results": result,
                    })
                    self.redis_client.publish(f"job:{self.job_id}", payload)
            except Exception:
                pass

            return result

        except Exception as e:
            logger.exception(f"Generation pipeline error: {e}")
            return self._finish_error(str(e))

    def _finish_error(self, error_msg: str) -> dict:
        try:
            job = self.db.query(GenerationJob).filter(GenerationJob.id == self.job_id).first()
            if job:
                job.status = "failed"
                job.error_message = error_msg
                job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
        except Exception:
            pass

        try:
            if self.redis_client:
                payload = json.dumps({
                    "job_id": self.job_id,
                    "percentage": 0,
                    "message": f"Error: {error_msg}",
                    "status": "failed",
                })
                self.redis_client.publish(f"job:{self.job_id}", payload)
        except Exception:
            pass

        return {"error": error_msg, "total_pairs": 0}

    def _finish_cancelled(self) -> dict:
        try:
            if self.redis_client:
                payload = json.dumps({
                    "job_id": self.job_id,
                    "percentage": 0,
                    "message": "Job cancelled by user",
                    "status": "cancelled",
                })
                self.redis_client.publish(f"job:{self.job_id}", payload)
        except Exception:
            pass
        return {"cancelled": True, "total_pairs": 0}
