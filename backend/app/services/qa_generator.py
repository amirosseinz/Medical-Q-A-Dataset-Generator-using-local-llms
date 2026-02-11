"""Core Q&A generation orchestrator — thin coordination layer.

Delegates to specialized services:
  - ``ProgressTracker``       — unified progress reporting (Redis + DB)
  - ``prompt_builder``        — evidence formatting (no citation tags) + prompt construction
  - ``AdaptiveBatchGenerator``— mini-batching, adaptive over-gen, early stopping
  - ``QualityValidator``      — completeness checks, semantic dedup, quality scoring

Pipeline steps:
  1. Process MedQuAD XML files
  2. Extract text from PDF / DOCX documents
  3. Fetch PubMed articles (abstracts + full-text)
  4. Store MedQuAD pairs
  5. Generate AI Q&A pairs (RAG or random)
  6. Validate, deduplicate, and store
  7. Finalize and report
"""
from __future__ import annotations

import logging

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Source, Chunk, QAPair, GenerationJob, QualityCheck
from app.services.document_processor import extract_medquad_pairs, extract_text
from app.services.pubmed_service import fetch_pubmed_abstracts, chunk_article_by_sections
from app.services.quality_checker import evaluate_qa_pair
from app.services.progress_tracker import ProgressTracker
from app.services.prompt_builder import (
    prepare_rag_prompts,
    prepare_random_prompts,
    PreparedPrompt,
)
from app.services.batch_generator import AdaptiveBatchGenerator
from app.services.quality_validator import QualityValidator
from app.utils.text_cleaning import clean_text
from app.utils.chunking import create_chunks

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Orchestrates the full dataset generation workflow.

    This is a thin coordination layer (~400 lines) that delegates heavy
    logic to specialised services.
    """

    def __init__(self, db: Session, job_id: str, project_id: str, config: dict):
        self.db = db
        self.job_id = job_id
        self.project_id = project_id
        self.config = config
        self.settings = get_settings()

        # Unified progress tracker
        self.progress = ProgressTracker(db, job_id, self.settings.REDIS_URL)

        # Resolve LLM provider and API key
        self.provider = config.get("provider", "ollama")
        self.api_key = ""
        if self.provider != "ollama":
            self.api_key = self._resolve_api_key()
            if not self.api_key:
                error_msg = (
                    f"No API key configured for provider '{self.provider}'. "
                    f"Please add your API key in Settings before starting generation."
                )
                logger.error(error_msg)
                self.progress.finish_failed(error_msg)
                raise ValueError(error_msg)
            logger.info("API key resolved for provider '%s'", self.provider)

    # ── Key resolution ─────────────────────────────────────────────────

    def _resolve_api_key(self) -> str:
        from app.services.api_key_service import APIKeyService
        service = APIKeyService(self.db)
        return service.get_key(
            provider=self.provider,
            api_key_id=self.config.get("api_key_id"),
        )

    # ── Main entry point ───────────────────────────────────────────────

    async def run(self) -> dict:
        """Execute the full generation pipeline."""
        try:
            self.progress.update("Starting dataset generation...", 2)

            medical_terms = [
                t.strip()
                for t in self.config.get("medical_terms", "").split(",")
                if t.strip()
            ]
            domain = self.config.get("medical_domain", "medical conditions")

            if not medical_terms:
                from app.models import Project
                project = self.db.query(Project).filter(
                    Project.id == self.project_id
                ).first()
                if project and project.domain:
                    medical_terms = [
                        t.strip() for t in project.domain.split(",") if t.strip()
                    ]
                if not medical_terms:
                    medical_terms = ["medical conditions"]

            # ── Step 1: MedQuAD XML ────────────────────────────────────
            medquad_pairs = self._process_medquad(medical_terms)

            if self.progress.is_cancelled():
                return self._finish_cancelled()

            # ── Step 2: PDF / DOCX documents ───────────────────────────
            all_chunks_with_meta = self._process_documents()

            if self.progress.is_cancelled():
                return self._finish_cancelled()

            # ── Step 3: PubMed articles ────────────────────────────────
            if self.config.get("use_pubmed", True) and medical_terms:
                await self._fetch_pubmed(medical_terms, all_chunks_with_meta)

            if self.progress.is_cancelled():
                return self._finish_cancelled()

            # ── Step 4: Store MedQuAD pairs ────────────────────────────
            existing_questions = self._store_medquad(medquad_pairs)

            # ── Step 5: AI generation ──────────────────────────────────
            remaining_pairs = max(
                0, self.config.get("target_pairs", 50) - len(medquad_pairs),
            )

            if remaining_pairs > 0 and self.config.get("use_ollama", True):
                self.progress.update("Preparing AI generation...", 55)
                model_name = self.config.get("ollama_model", "llama3")
                ollama_url = self.config.get("ollama_url", self.settings.OLLAMA_URL)
                question_types = self.config.get("question_types", ["factual", "reasoning"])
                difficulty_levels = self.config.get("difficulty_levels", ["intermediate"])

                # Ollama health check
                if self.provider == "ollama":
                    from app.services.ollama_service import health_check
                    self.progress.update("Checking Ollama connectivity...", 55)
                    if not await health_check(ollama_url, retries=3, delay=5.0):
                        return self._finish_error(
                            "Ollama server is not responding. "
                            "Please ensure Ollama is running and try again."
                        )

                rag_enabled = self.settings.RAG_ENABLED and all_chunks_with_meta

                if rag_enabled:
                    await self._run_rag_generation(
                        medical_terms, domain, question_types, difficulty_levels,
                        model_name, ollama_url, remaining_pairs, existing_questions,
                    )
                elif all_chunks_with_meta:
                    await self._run_random_generation(
                        all_chunks_with_meta, domain, question_types, difficulty_levels,
                        model_name, ollama_url, remaining_pairs, existing_questions,
                    )

            # ── Step 6: Finalize ───────────────────────────────────────
            return self._finalize()

        except Exception as e:
            logger.exception("Generation pipeline error: %s", e)
            return self._finish_error(str(e))

    # ── Step implementations ───────────────────────────────────────────

    def _process_medquad(self, medical_terms: list[str]) -> list[dict]:
        """Step 1: Extract Q&A pairs from MedQuAD XML sources."""
        medquad_pairs: list[dict] = []
        sources = self.db.query(Source).filter(
            Source.project_id == self.project_id,
            Source.file_type == "xml",
        ).all()

        if not sources:
            return medquad_pairs

        self.progress.update("Processing MedQuAD XML files...", 8)
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
        logger.info("Extracted %d MedQuAD pairs", len(medquad_pairs))
        return medquad_pairs

    def _process_documents(self) -> list[tuple[str, str, str | None]]:
        """Step 2: Extract text from PDF/DOCX and create chunks."""
        all_chunks: list[tuple[str, str, str | None]] = []
        doc_sources = self.db.query(Source).filter(
            Source.project_id == self.project_id,
            Source.file_type.in_(["pdf", "docx"]),
        ).all()

        if not doc_sources:
            return all_chunks

        self.progress.update("Extracting text from documents...", 15)
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
                    for idx, chunk_text in enumerate(chunks):
                        self.db.add(Chunk(
                            source_id=src.id,
                            project_id=self.project_id,
                            chunk_index=idx,
                            content=chunk_text,
                            word_count=len(chunk_text.split()),
                        ))
                        all_chunks.append((chunk_text, "PDF", src.filename))
                    src.processing_status = "completed"
                else:
                    src.processing_status = "failed"
                    src.error_message = doc.error or "No text extracted"
            pct = 15 + int((i + 1) / len(doc_sources) * 15)
            self.progress.update(f"Processed document {i + 1}/{len(doc_sources)}", pct)
        self.db.commit()
        return all_chunks

    async def _fetch_pubmed(
        self,
        medical_terms: list[str],
        all_chunks: list[tuple[str, str, str | None]],
    ) -> None:
        """Step 3: Fetch PubMed articles and create chunks."""
        self.progress.update("Fetching PubMed articles...", 35)
        pubmed_result = await fetch_pubmed_abstracts(
            medical_terms=medical_terms,
            email=self.config.get("email", self.settings.PUBMED_EMAIL),
            retmax=self.config.get("pubmed_retmax", 1000),
            fetch_full_text=self.config.get("pubmed_full_text", True),
            max_full_text=self.config.get("pubmed_max_full_text", 50),
        )

        if pubmed_result.articles:
            pubmed_source = Source(
                project_id=self.project_id,
                filename="pubmed_articles",
                file_type="pubmed",
                processing_status="completed",
                metadata_json={
                    "total_ids": pubmed_result.total_ids_found,
                    "total_fetched": pubmed_result.total_fetched,
                    "full_text_count": pubmed_result.full_text_count,
                    "abstract_only_count": pubmed_result.abstract_only_count,
                },
            )
            self.db.add(pubmed_source)
            self.db.flush()

            chunk_idx = 0
            for article in pubmed_result.articles:
                if article.has_full_text:
                    article_chunks = chunk_article_by_sections(
                        article,
                        max_chunk_words=self.config.get("chunk_size", 500),
                        overlap_words=self.config.get("chunk_overlap", 50),
                    )
                else:
                    cleaned = clean_text(article.abstract)
                    std_chunks = create_chunks(
                        cleaned,
                        strategy=self.config.get("chunking_strategy", "word_count"),
                        chunk_size=self.config.get("chunk_size", 500),
                        overlap=self.config.get("chunk_overlap", 50),
                    )
                    article_chunks = [(c, "abstract") for c in std_chunks]

                for chunk_text, _section in article_chunks:
                    self.db.add(Chunk(
                        source_id=pubmed_source.id,
                        project_id=self.project_id,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        word_count=len(chunk_text.split()),
                    ))
                    all_chunks.append((chunk_text, "PubMed", article.display_name))
                    chunk_idx += 1

            self.db.commit()
            logger.info(
                "PubMed: %d chunks from %d articles (%d full-text, %d abstract-only)",
                chunk_idx, len(pubmed_result.articles),
                pubmed_result.full_text_count, pubmed_result.abstract_only_count,
            )
        elif pubmed_result.abstracts:
            # Legacy fallback
            combined = "\n\n".join(pubmed_result.abstracts)
            cleaned = clean_text(combined)
            chunks = create_chunks(cleaned,
                strategy=self.config.get("chunking_strategy", "word_count"),
                chunk_size=self.config.get("chunk_size", 500),
                overlap=self.config.get("chunk_overlap", 50),
            )
            pubmed_source = Source(
                project_id=self.project_id, filename="pubmed_abstracts",
                file_type="pubmed", processing_status="completed",
                metadata_json={"total_ids": pubmed_result.total_ids_found,
                               "total_fetched": pubmed_result.total_fetched},
            )
            self.db.add(pubmed_source)
            self.db.flush()
            for idx, chunk_text in enumerate(chunks):
                self.db.add(Chunk(
                    source_id=pubmed_source.id, project_id=self.project_id,
                    chunk_index=idx, content=chunk_text,
                    word_count=len(chunk_text.split()),
                ))
                all_chunks.append((chunk_text, "PubMed", None))
            self.db.commit()
            logger.info("PubMed (legacy): %d chunks", len(chunks))

    def _store_medquad(self, medquad_pairs: list[dict]) -> list[str]:
        """Step 4: Store MedQuAD pairs with quality checks."""
        self.progress.update("Storing MedQuAD pairs...", 50)
        existing_questions: list[str] = []
        for mp in medquad_pairs:
            passed, score, checks = evaluate_qa_pair(
                mp["question"], mp["answer"], existing_questions,
            )
            qa = QAPair(
                project_id=self.project_id,
                question=mp["question"],
                answer=mp["answer"],
                source_type="medquad",
                source_document=mp.get("source_file", "MedQuAD"),
                source_metadata={"source_file": mp.get("source_file")},
                quality_score=score,
                validation_status="approved" if passed else "pending",
                metadata_json={"source_file": mp.get("source_file")},
            )
            self.db.add(qa)
            self.db.flush()
            existing_questions.append(mp["question"])
            for check in checks:
                self.db.add(QualityCheck(
                    qa_pair_id=qa.id,
                    check_type=check.check_type,
                    passed=check.passed,
                    score=check.score,
                    details=check.details,
                ))
        self.db.commit()
        return existing_questions

    # ── Generation paths ───────────────────────────────────────────────

    async def _run_rag_generation(
        self,
        medical_terms: list[str],
        domain: str,
        question_types: list[str],
        difficulty_levels: list[str],
        model_name: str,
        ollama_url: str,
        remaining_pairs: int,
        existing_questions: list[str],
    ) -> list:
        """RAG path: build FAISS index → retrieve → generate → validate → store."""
        from app.services.rag_service import get_project_index

        settings = self.settings
        self.progress.update("Building RAG vector index...", 56)

        # Load chunks and build FAISS index
        project_chunks = self.db.query(Chunk).filter(
            Chunk.project_id == self.project_id
        ).all()
        if not project_chunks:
            logger.warning("No chunks for RAG — empty result")
            return []

        chunk_ids = [c.id for c in project_chunks]
        chunk_texts = [c.content for c in project_chunks]
        project_sources = {
            s.id: s for s in self.db.query(Source).filter(
                Source.project_id == self.project_id
            ).all()
        }
        chunk_metadata = [
            {
                "source_id": c.source_id,
                "source_filename": project_sources.get(c.source_id, Source(filename="unknown")).filename,
                "chunk_index": c.chunk_index,
                "word_count": c.word_count,
                "content_preview": c.content[:200],
            }
            for c in project_chunks
        ]

        index = get_project_index(self.project_id)
        index.build_index(chunk_ids, chunk_texts, chunk_metadata)
        logger.info("FAISS index: %d vectors for %d terms", index.size, len(medical_terms))

        # ── Prepare prompts (NO citation tags in evidence) ─────────────
        self.progress.update("Retrieving evidence per topic...", 58)
        batch_gen = AdaptiveBatchGenerator(
            target_pairs=remaining_pairs,
            initial_multiplier=settings.OVER_GEN_INITIAL_MULTIPLIER,
            min_multiplier=settings.OVER_GEN_MIN_MULTIPLIER,
            max_multiplier=settings.OVER_GEN_MAX_MULTIPLIER,
            adapt_interval=settings.OVER_GEN_ADAPT_INTERVAL,
            mini_batch_size=settings.MINI_BATCH_SIZE,
        )

        prepared = prepare_rag_prompts(
            medical_terms=medical_terms,
            domain=domain,
            question_types=question_types,
            difficulty_levels=difficulty_levels,
            target_prompts=batch_gen.compute_target_prompts(),
            existing_questions=existing_questions,
            project_id=self.project_id,
            db=self.db,
            settings=settings,
        )

        if not prepared:
            logger.warning("RAG retrieval produced no prompts")
            return []

        # ── Generate via LLM ──────────────────────────────────────────
        provider_label = self.provider.capitalize() if self.provider != "ollama" else "Ollama"
        self.progress.update(f"Generating RAG Q&A with {provider_label}...", 60)

        async def on_progress(completed: int, total: int, pair):
            pct = 60 + int((completed / max(total, 1)) * 30)
            self.progress.update(
                f"RAG: {completed}/{total} prompts | "
                f"{batch_gen.stats.parse_successes} parsed "
                f"(target: {remaining_pairs})",
                pct,
            )

        tuples = [p.as_tuple() for p in prepared]
        ai_pairs = await batch_gen.generate(
            chunks_with_prompts=tuples,
            provider=self.provider,
            model=model_name,
            api_key=self.api_key,
            ollama_url=ollama_url,
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
            max_concurrent=self.config.get("max_workers", 5),
            progress_callback=on_progress,
        )

        # ── Validate and store ─────────────────────────────────────────
        return self._validate_and_store(
            ai_pairs, prepared, existing_questions,
            remaining_pairs, model_name, is_rag=True,
            batch_gen=batch_gen,
        )

    async def _run_random_generation(
        self,
        all_chunks_with_meta: list[tuple[str, str, str | None]],
        domain: str,
        question_types: list[str],
        difficulty_levels: list[str],
        model_name: str,
        ollama_url: str,
        remaining_pairs: int,
        existing_questions: list[str],
    ) -> list:
        """Random-sampling fallback (RAG disabled)."""
        settings = self.settings
        batch_gen = AdaptiveBatchGenerator(
            target_pairs=remaining_pairs,
            initial_multiplier=settings.OVER_GEN_INITIAL_MULTIPLIER,
            mini_batch_size=settings.MINI_BATCH_SIZE,
        )

        prepared = prepare_random_prompts(
            all_chunks_with_meta=all_chunks_with_meta,
            domain=domain,
            question_types=question_types,
            difficulty_levels=difficulty_levels,
            target_prompts=batch_gen.compute_target_prompts(),
            existing_questions=existing_questions,
            config=self.config,
        )

        if not prepared:
            return []

        provider_label = self.provider.capitalize() if self.provider != "ollama" else "Ollama"
        self.progress.update(f"Generating Q&A with {provider_label}...", 60)

        async def on_progress(completed: int, total: int, pair):
            pct = 60 + int((completed / max(total, 1)) * 30)
            self.progress.update(
                f"AI Q&A: {completed}/{total} ({batch_gen.stats.parse_successes} parsed)",
                pct,
            )

        tuples = [p.as_tuple() for p in prepared]
        ai_pairs = await batch_gen.generate(
            chunks_with_prompts=tuples,
            provider=self.provider,
            model=model_name,
            api_key=self.api_key,
            ollama_url=ollama_url,
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
            max_concurrent=self.config.get("max_workers", 5),
            progress_callback=on_progress,
        )

        return self._validate_and_store(
            ai_pairs, prepared, existing_questions,
            remaining_pairs, model_name, is_rag=False,
            batch_gen=batch_gen,
        )

    # ── Shared validation + storage ────────────────────────────────────

    def _validate_and_store(
        self,
        ai_pairs: list,
        prepared: list[PreparedPrompt],
        existing_questions: list[str],
        remaining_pairs: int,
        model_name: str,
        is_rag: bool,
        batch_gen: AdaptiveBatchGenerator | None = None,
    ) -> list:
        """Validate generated pairs with QualityValidator and store accepted ones.

        Uses ``pair.prompt_index`` to map each pair back to the correct
        PreparedPrompt metadata (fixes the index-mismatch bug that occurred
        when some prompts failed to parse).
        """
        self.progress.update("Running quality checks...", 92)

        validator = QualityValidator(
            min_quality_score=self.config.get(
                "min_quality_score", self.settings.MIN_QUALITY_SCORE,
            ),
            semantic_dup_threshold=self.settings.SEMANTIC_DUP_THRESHOLD,
            existing_questions=existing_questions,
        )

        stored_pairs = []
        batch_count = 0

        for pair in ai_pairs:
            if self.progress.is_cancelled():
                break
            if len(stored_pairs) >= remaining_pairs:
                logger.info(
                    "Target reached: %d/%d pairs after validating %d LLM responses",
                    len(stored_pairs), remaining_pairs, validator.total_validated,
                )
                break

            result = validator.validate(pair.question, pair.answer)

            if not result.passed:
                continue

            # Use prompt_index for correct metadata mapping
            idx = getattr(pair, 'prompt_index', -1)
            meta = prepared[idx].metadata if 0 <= idx < len(prepared) else {}

            if is_rag:
                source_tag = (
                    f"rag_{self.provider}"
                    if self.provider != "ollama"
                    else meta.get("source_type", "rag_ollama")
                )
                source_doc = meta.get("source_filename") or meta.get("query_term")
                qa = QAPair(
                    project_id=self.project_id,
                    generation_job_id=self.job_id,
                    chunk_id=meta.get("primary_chunk_id"),
                    question=pair.question,
                    answer=pair.answer,
                    source_type=source_tag,
                    source_document=source_doc,
                    source_metadata={
                        "query_term": meta.get("query_term"),
                        "retrieval_scores": meta.get("retrieval_scores", []),
                        "citation_chunk_ids": meta.get("citation_ids", []),
                        "rag": True,
                    },
                    model_used=model_name,
                    provider=self.provider,
                    quality_score=result.score,
                    validation_status="pending",
                    metadata_json={
                        "rag": True,
                        "provider": self.provider,
                        "query_term": meta.get("query_term"),
                        "question_type": meta.get("question_type"),
                        "retrieval_scores": meta.get("retrieval_scores", []),
                        "citation_chunk_ids": meta.get("citation_ids", []),
                        "citations": meta.get("citation_metadata", []),
                        "source_filename": meta.get("source_filename"),
                    },
                )
            else:
                source_type = meta.get("source_type", "unknown")
                filename = meta.get("source_filename")
                qa_source = f"{'pdf' if source_type == 'PDF' else 'pubmed'}_{self.provider}"
                qa = QAPair(
                    project_id=self.project_id,
                    generation_job_id=self.job_id,
                    question=pair.question,
                    answer=pair.answer,
                    source_type=qa_source,
                    source_document=filename,
                    source_metadata={
                        "original_source": source_type,
                        "original_file": filename,
                    },
                    model_used=model_name,
                    provider=self.provider,
                    quality_score=result.score,
                    validation_status="pending",
                    metadata_json={
                        "rag": False,
                        "provider": self.provider,
                        "original_source": source_type,
                        "original_file": filename,
                    },
                )

            self.db.add(qa)
            self.db.flush()
            stored_pairs.append(pair)

            for check in result.checks:
                self.db.add(QualityCheck(
                    qa_pair_id=qa.id,
                    check_type=check.check_type,
                    passed=check.passed,
                    score=check.score,
                    details=check.details,
                ))

            batch_count += 1
            if batch_count % 50 == 0:
                self.db.commit()

        self.db.commit()

        # Log summary from validator
        summary = validator.summary()
        path_label = "RAG" if is_rag else "Random"
        logger.info(
            "%s quality summary: %d stored, %d rejected "
            "(breakdown: %s) out of %d LLM responses (target=%d)",
            path_label,
            summary["total_accepted"],
            summary["total_rejected"],
            summary["rejection_breakdown"],
            len(ai_pairs),
            remaining_pairs,
        )

        # Wire acceptance count back to batch generator for adaptive multiplier
        if batch_gen is not None:
            batch_gen.update_accepted(summary["total_accepted"])
            batch_gen.stats.rejected += summary["total_rejected"]
            batch_gen.stats.duplicates += summary["rejection_breakdown"].get("duplicate", 0)
            batch_gen.stats.semantic_dups += summary["rejection_breakdown"].get("semantic_dup", 0)
            batch_gen.stats.quality_failures += summary["rejection_breakdown"].get("quality", 0)
            batch_gen.stats.not_found += summary["rejection_breakdown"].get("not_found", 0)
            batch_gen.stats.truncated += summary["rejection_breakdown"].get("completeness", 0)
            logger.info(
                "%s batch stats: %s",
                path_label, batch_gen.stats.summary_line(),
            )

        # Feed stats back to progress tracker
        self.progress.stats.update({
            "pairs_accepted": summary["total_accepted"],
            "pairs_rejected": summary["total_rejected"],
            **{f"rej_{k}": v for k, v in summary["rejection_breakdown"].items()},
        })

        return stored_pairs

    # ── Finalization helpers ───────────────────────────────────────────

    def _finalize(self) -> dict:
        """Step 6: Finalize — count pairs, build summary, mark job complete."""
        self.progress.update("Finalizing dataset...", 97)

        total_pairs = self.db.query(QAPair).filter(
            QAPair.project_id == self.project_id
        ).count()

        job = self.db.query(GenerationJob).filter(
            GenerationJob.id == self.job_id
        ).first()
        if job:
            gen_pairs = self.db.query(QAPair).filter(
                QAPair.generation_job_id == self.job_id
            ).count()
            job.qa_pair_count = gen_pairs

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

        self.progress.finish_completed(result)
        self.progress.update("Dataset generation completed!", 100)
        return result

    def _finish_error(self, error_msg: str) -> dict:
        self.progress.finish_failed(error_msg)
        return {"error": error_msg, "total_pairs": 0}

    def _finish_cancelled(self) -> dict:
        self.progress.finish_cancelled()
        return {"cancelled": True, "total_pairs": 0}
