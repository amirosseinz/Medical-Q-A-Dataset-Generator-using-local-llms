"""LLM review endpoints — review sessions, fact-check, progress tracking."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import QAPair
from app.models.review_session import ReviewSession
from app.services.api_key_service import APIKeyService
from app.services.llm_review_service import (
    review_qa_pair,
    review_batch,
    fact_check_qa_pair,
)
from app.services.llm_http import PROVIDER_MODELS
from app.services.model_fetcher import FALLBACK_MODELS
from app.services.rate_limit_handler import (
    RateLimitHandler,
    estimate_cost,
    get_concurrency_limit,
    get_request_spacing,
)
from app.schemas.llm_provider import (
    ReviewStartRequest,
    ReviewSessionResponse,
    FactCheckRequest,
    FactCheckResponse,
    CostEstimate,
    AutoApproveWorkflowRequest,
    AcceptSuggestionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---- Helpers ----

def _resolve_api_key(provider: str, api_key_id: str | None, api_key: str | None, db: Session) -> str:
    """Resolve API key using unified APIKeyService."""
    service = APIKeyService(db)
    return service.get_key(
        provider=provider,
        api_key_id=api_key_id,
        direct_key=api_key,
    )


# ---- Provider Info ----

class ProviderInfo(BaseModel):
    name: str
    models: list[str]
    requires_api_key: bool
    has_stored_key: bool = False
    stored_key_id: str | None = None
    models_source: str = "hardcoded"  # "hardcoded", "fetched", "none"
    models_fetched_at: str | None = None


@router.get("/review/providers", response_model=list[ProviderInfo])
async def list_providers(db: Session = Depends(get_db)):
    """List available LLM review providers with dynamically fetched models.

    Priority for model lists:
      1. Cached models from DB (fetched from provider API)
      2. Hardcoded PROVIDER_MODELS fallback
    """
    provider_names = ["openai", "anthropic", "gemini", "openrouter", "ollama"]
    providers = []

    service = APIKeyService(db)

    for name in provider_names:
        # Use unified service to find stored key (handles google↔gemini alias)
        stored = service.get_key_record(name)

        # Determine model list: prefer DB-cached models, then fallback
        models_source = "none"
        models_fetched_at = None
        if stored and stored.available_models and len(stored.available_models) > 0:
            models = stored.available_models
            models_source = "fetched"
            if hasattr(stored, "models_fetched_at") and stored.models_fetched_at:
                models_fetched_at = stored.models_fetched_at.isoformat() if hasattr(stored.models_fetched_at, "isoformat") else str(stored.models_fetched_at)
        elif stored:
            # Key exists but models not fetched yet
            models = PROVIDER_MODELS.get(name, FALLBACK_MODELS.get(name, []))
            models_source = "hardcoded"
        else:
            models = PROVIDER_MODELS.get(name, FALLBACK_MODELS.get(name, []))
            models_source = "hardcoded"

        providers.append(ProviderInfo(
            name=name,
            models=models,
            requires_api_key=name != "ollama",
            has_stored_key=stored is not None,
            stored_key_id=stored.id if stored else None,
            models_source=models_source,
            models_fetched_at=models_fetched_at,
        ))

    # Fetch Ollama models dynamically
    try:
        import httpx
        from app.config import get_settings
        settings = get_settings()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.OLLAMA_URL.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                ollama_models = [m["name"] for m in resp.json().get("models", [])]
                for p in providers:
                    if p.name == "ollama":
                        p.models = ollama_models
    except Exception:
        pass

    return providers


# ---- Cost Estimation ----

@router.get("/review/estimate-cost", response_model=CostEstimate)
async def estimate_review_cost(
    pair_count: int = 25,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
):
    """Estimate the cost of reviewing N pairs with a given model."""
    cost = estimate_cost(model, pair_count)
    spacing = get_request_spacing(provider, model)
    concurrency = get_concurrency_limit(provider, model)
    time_per_pair = max(spacing, 2.0)  # At least 2 seconds per pair (LLM response time)
    est_time = (pair_count / max(concurrency, 1)) * time_per_pair

    return CostEstimate(
        provider=provider,
        model=model,
        pair_count=pair_count,
        estimated_cost_usd=cost,
        estimated_time_seconds=round(est_time, 1),
    )


# ---- Review Sessions ----

@router.post("/projects/{project_id}/review/start", response_model=ReviewSessionResponse)
async def start_review_session(
    project_id: str,
    request: ReviewStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start an asynchronous LLM review session with rate limit handling."""
    # Validate pairs exist
    pairs = (
        db.query(QAPair)
        .filter(QAPair.project_id == project_id, QAPair.id.in_(request.qa_pair_ids))
        .all()
    )
    if not pairs:
        raise HTTPException(status_code=404, detail="No matching Q&A pairs found")

    # Resolve API key before starting background task
    resolved_key = _resolve_api_key(request.provider, request.api_key_id, request.api_key, db)
    if request.provider != "ollama" and not resolved_key:
        raise HTTPException(
            status_code=400,
            detail=f"No API key available for {request.provider}. Add one in Settings or provide directly.",
        )

    # Create review session
    session = ReviewSession(
        project_id=project_id,
        provider=request.provider,
        model_name=request.model or (PROVIDER_MODELS.get(request.provider, [""])[0] if PROVIDER_MODELS.get(request.provider) else ""),
        status="pending",
        total_pairs=len(pairs),
        qa_pair_ids=[p.id for p in pairs],
        completed_pair_ids=[],
        results=[],
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # Launch background task
    background_tasks.add_task(
        _run_review_session,
        session_id=session.id,
        project_id=project_id,
        provider=request.provider,
        api_key=resolved_key,
        model=request.model or session.model_name,
        ollama_url=request.ollama_url,
        speed=request.speed,
        pair_data=[{"id": p.id, "question": p.question, "answer": p.answer} for p in pairs],
    )

    return ReviewSessionResponse.model_validate(session)


@router.get("/review/sessions/{session_id}", response_model=ReviewSessionResponse)
def get_review_session(session_id: str, db: Session = Depends(get_db)):
    """Poll review session progress."""
    session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Review session not found")
    return ReviewSessionResponse.model_validate(session)


@router.post("/review/sessions/{session_id}/cancel")
def cancel_review_session(session_id: str, db: Session = Depends(get_db)):
    """Cancel an in-progress review session (completed work is preserved)."""
    session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Review session not found")
    if session.status in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="Session already finished")
    session.status = "cancelled"
    session.current_message = "Cancelled by user"
    session.completed_at = datetime.now(timezone.utc)
    db.commit()
    return {"message": "Session cancelled", "completed_pairs": session.completed_pairs}


@router.post("/review/sessions/{session_id}/resume", response_model=ReviewSessionResponse)
async def resume_review_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Resume a failed/cancelled review session from where it stopped."""
    session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Review session not found")
    if session.status == "in_progress":
        raise HTTPException(status_code=400, detail="Session already running")
    if session.status == "completed":
        raise HTTPException(status_code=400, detail="Session already completed")

    # Find remaining pairs
    completed_ids = set(session.completed_pair_ids or [])
    remaining_ids = [pid for pid in (session.qa_pair_ids or []) if pid not in completed_ids]
    if not remaining_ids:
        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)
        db.commit()
        return ReviewSessionResponse.model_validate(session)

    # Fetch remaining pairs
    remaining_pairs = (
        db.query(QAPair)
        .filter(QAPair.id.in_(remaining_ids))
        .all()
    )

    # Resolve API key
    resolved_key = _resolve_api_key(session.provider, None, None, db)
    if session.provider != "ollama" and not resolved_key:
        raise HTTPException(status_code=400, detail=f"No stored API key for {session.provider}")

    session.status = "pending"
    session.current_message = f"Resuming from {session.completed_pairs}/{session.total_pairs}"
    session.error_message = None
    db.commit()
    db.refresh(session)

    background_tasks.add_task(
        _run_review_session,
        session_id=session.id,
        project_id=session.project_id,
        provider=session.provider,
        api_key=resolved_key,
        model=session.model_name,
        ollama_url="http://host.docker.internal:11434",
        speed="normal",
        pair_data=[{"id": p.id, "question": p.question, "answer": p.answer} for p in remaining_pairs],
        is_resume=True,
    )

    return ReviewSessionResponse.model_validate(session)


# ---- Legacy synchronous review (kept for backward compatibility) ----

class LegacyReviewRequest(BaseModel):
    qa_pair_ids: list[str]
    provider: str = "openai"
    api_key: str = ""
    model: str = ""
    ollama_url: str = "http://host.docker.internal:11434"


class LegacyReviewResult(BaseModel):
    qa_pair_id: str
    accuracy: float = 0
    completeness: float = 0
    clarity: float = 0
    relevance: float = 0
    overall: float = 0
    recommendation: str = "revise"
    feedback: str = ""
    error: str | None = None


class LegacyReviewResponse(BaseModel):
    results: list[LegacyReviewResult]
    total_reviewed: int
    avg_overall: float | None = None


@router.post("/projects/{project_id}/review", response_model=LegacyReviewResponse)
async def review_qa_pairs_sync(
    project_id: str,
    request: LegacyReviewRequest,
    db: Session = Depends(get_db),
):
    """Review selected Q&A pairs synchronously (legacy, for small batches)."""
    pairs = (
        db.query(QAPair)
        .filter(QAPair.project_id == project_id, QAPair.id.in_(request.qa_pair_ids))
        .all()
    )
    if not pairs:
        raise HTTPException(status_code=404, detail="No matching Q&A pairs found")

    # Resolve API key
    resolved_key = _resolve_api_key(request.provider, None, request.api_key, db)
    if request.provider != "ollama" and not resolved_key:
        raise HTTPException(status_code=400, detail=f"No API key for {request.provider}")

    pair_dicts = [{"id": p.id, "question": p.question, "answer": p.answer} for p in pairs]
    raw_results = await review_batch(
        pairs=pair_dicts,
        provider=request.provider,
        api_key=resolved_key,
        model=request.model,
        ollama_url=request.ollama_url,
        speed="normal",
    )

    results: list[LegacyReviewResult] = []
    overall_scores: list[float] = []
    for raw in raw_results:
        qa_id = raw.get("qa_pair_id", "")
        result = LegacyReviewResult(
            qa_pair_id=qa_id,
            accuracy=raw.get("accuracy", 0),
            completeness=raw.get("completeness", 0),
            clarity=raw.get("clarity", 0),
            relevance=raw.get("relevance", 0),
            overall=raw.get("overall", 0),
            recommendation=raw.get("recommendation", "revise"),
            feedback=raw.get("feedback", ""),
            error=raw.get("error"),
        )
        results.append(result)
        if not raw.get("error"):
            overall_scores.append(result.overall)

        # Save review to QAPair metadata
        qa_pair = next((p for p in pairs if p.id == qa_id), None)
        if qa_pair:
            meta = qa_pair.metadata_json or {}
            meta["llm_review"] = {
                "provider": request.provider,
                "model": request.model,
                "accuracy": result.accuracy,
                "completeness": result.completeness,
                "clarity": result.clarity,
                "relevance": result.relevance,
                "overall": result.overall,
                "recommendation": result.recommendation,
                "feedback": result.feedback,
            }
            qa_pair.metadata_json = meta

    db.commit()

    return LegacyReviewResponse(
        results=results,
        total_reviewed=len(results),
        avg_overall=round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else None,
    )


# ---- Auto-Approve ----

@router.post("/projects/{project_id}/review/auto-approve")
async def auto_approve_reviewed(
    project_id: str,
    min_overall: float = 7.0,
    db: Session = Depends(get_db),
):
    """Auto-approve Q&A pairs that scored above threshold in LLM review."""
    pairs = (
        db.query(QAPair)
        .filter(QAPair.project_id == project_id, QAPair.validation_status == "pending")
        .all()
    )

    approved_count = 0
    rejected_count = 0
    for pair in pairs:
        meta = pair.metadata_json or {}
        review = meta.get("llm_review")
        if not review:
            continue
        overall = review.get("overall", 0)
        if overall >= min_overall:
            pair.validation_status = "approved"
            approved_count += 1
        elif overall < 4.0:
            pair.validation_status = "rejected"
            rejected_count += 1

    db.commit()
    return {
        "approved": approved_count,
        "rejected": rejected_count,
        "message": f"Auto-approved {approved_count}, auto-rejected {rejected_count} pairs",
    }


# ---- Fact-Check ----

@router.post("/projects/{project_id}/fact-check", response_model=FactCheckResponse)
async def fact_check(
    project_id: str,
    request: FactCheckRequest,
    db: Session = Depends(get_db),
):
    """Fact-check a single Q&A pair using an LLM."""
    pair = db.query(QAPair).filter(
        QAPair.id == request.qa_pair_id,
        QAPair.project_id == project_id,
    ).first()
    if not pair:
        raise HTTPException(status_code=404, detail="Q&A pair not found")

    resolved_key = _resolve_api_key(request.provider, request.api_key_id, request.api_key, db)
    if request.provider != "ollama" and not resolved_key:
        raise HTTPException(status_code=400, detail=f"No API key for {request.provider}")

    try:
        rate_handler = RateLimitHandler(request.provider, request.model)
        result = await fact_check_qa_pair(
            question=pair.question,
            answer=pair.answer,
            provider=request.provider,
            api_key=resolved_key,
            model=request.model,
            ollama_url=request.ollama_url,
            rate_handler=rate_handler,
        )

        cost = estimate_cost(request.model, 1)

        # Save fact-check result to pair metadata
        meta = pair.metadata_json or {}
        meta["fact_check"] = result
        pair.metadata_json = meta
        db.commit()

        return FactCheckResponse(
            qa_pair_id=pair.id,
            factual_accuracy=result.get("factual_accuracy", 5),
            analysis=result.get("analysis", []),
            suggested_answer=result.get("suggested_answer"),
            confidence=result.get("confidence", 0.5),
            cost_usd=cost,
        )

    except Exception as e:
        logger.error(f"Fact-check failed for pair {pair.id}: {e}")
        return FactCheckResponse(
            qa_pair_id=pair.id,
            error=str(e),
        )


# ---- Background Review Session Runner ----

async def _run_review_session(
    session_id: str,
    project_id: str,
    provider: str,
    api_key: str,
    model: str,
    ollama_url: str,
    speed: str,
    pair_data: list[dict[str, str]],
    is_resume: bool = False,
) -> None:
    """Background task: process review session with rate limiting."""
    db = SessionLocal()
    try:
        session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
        if not session:
            return

        session.status = "in_progress"
        session.started_at = session.started_at or datetime.now(timezone.utc)
        session.current_message = f"Reviewing 0/{len(pair_data)} pairs..."
        db.commit()

        rate_handler = RateLimitHandler(provider, model, speed=speed)
        completed_count = session.completed_pairs if is_resume else 0
        all_results = list(session.results or []) if is_resume else []
        completed_ids = list(session.completed_pair_ids or []) if is_resume else []
        overall_scores: list[float] = []

        for existing in all_results:
            if not existing.get("error") and existing.get("overall"):
                overall_scores.append(float(existing["overall"]))

        for i, pair in enumerate(pair_data):
            # Check if session was cancelled
            db.expire(session)
            if session.status == "cancelled":
                logger.info(f"Review session {session_id} cancelled at pair {i+1}/{len(pair_data)}")
                break

            try:
                result = await review_qa_pair(
                    question=pair["question"],
                    answer=pair["answer"],
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    ollama_url=ollama_url,
                    rate_handler=rate_handler,
                )
                result["qa_pair_id"] = pair["id"]
                all_results.append(result)
                completed_ids.append(pair["id"])
                completed_count += 1

                # Classify recommendation
                rec = result.get("recommendation", "revise")
                if rec == "approve":
                    session.approved_count += 1
                elif rec == "reject":
                    session.rejected_count += 1
                else:
                    session.revise_count += 1

                if not result.get("error"):
                    overall_scores.append(float(result.get("overall", 0)))

                # Save review result to QAPair metadata
                qa_pair = db.query(QAPair).filter(QAPair.id == pair["id"]).first()
                if qa_pair:
                    meta = qa_pair.metadata_json or {}
                    meta["llm_review"] = {
                        "provider": provider,
                        "model": model,
                        "accuracy": result.get("accuracy", 0),
                        "completeness": result.get("completeness", 0),
                        "clarity": result.get("clarity", 0),
                        "relevance": result.get("relevance", 0),
                        "overall": result.get("overall", 0),
                        "recommendation": rec,
                        "feedback": result.get("feedback", ""),
                    }
                    qa_pair.metadata_json = meta

            except Exception as e:
                logger.error(f"Review failed for pair {pair['id']}: {e}")
                error_result = {
                    "qa_pair_id": pair["id"],
                    "error": str(e),
                    "accuracy": 0, "completeness": 0, "clarity": 0,
                    "relevance": 0, "overall": 0,
                    "recommendation": "revise",
                    "feedback": f"Review failed: {e}",
                }
                all_results.append(error_result)
                completed_ids.append(pair["id"])
                completed_count += 1
                session.failed_pairs += 1

            # Update session progress (commit every pair for real-time polling)
            session.completed_pairs = completed_count
            session.completed_pair_ids = completed_ids
            session.results = all_results
            session.avg_overall_score = (
                round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else None
            )
            msg_parts = [f"Reviewed {completed_count}/{session.total_pairs} pairs"]
            if rate_handler.state.is_rate_limited:
                msg_parts.append(rate_handler.state.current_message)
            session.current_message = ". ".join(msg_parts)
            db.commit()

        # Finalize
        if session.status != "cancelled":
            session.status = "completed"
            session.current_message = (
                f"Review complete: {session.approved_count} approved, "
                f"{session.revise_count} need revision, "
                f"{session.rejected_count} rejected"
            )
        session.completed_at = datetime.now(timezone.utc)
        session.total_cost_usd = estimate_cost(model, completed_count)
        db.commit()

    except Exception as e:
        logger.exception(f"Review session {session_id} failed: {e}")
        try:
            session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
            if session:
                session.status = "failed"
                session.error_message = str(e)
                session.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


# ---- Accept Fact-Check Suggestion ----

@router.post("/qa-pairs/{pair_id}/accept-suggestion", response_model=AcceptSuggestionResponse)
async def accept_suggestion(pair_id: str, db: Session = Depends(get_db)):
    """Apply the suggested answer from a fact-check result."""
    qa = db.query(QAPair).filter(QAPair.id == pair_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A pair not found")

    meta = qa.metadata_json or {}
    fact_check_data = meta.get("fact_check")
    if not fact_check_data:
        raise HTTPException(status_code=400, detail="No fact-check result found for this pair")

    suggested = fact_check_data.get("suggested_answer")
    if not suggested:
        raise HTTPException(status_code=400, detail="No suggested answer available")

    old_answer = qa.answer
    qa.answer = suggested
    qa.human_edited = True

    # Track suggestion acceptance in metadata
    meta["suggestion_applied"] = {
        "old_answer": old_answer,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "source": "fact_check",
        "fact_check_score": fact_check_data.get("factual_accuracy", 0),
    }
    qa.metadata_json = meta
    db.commit()

    return AcceptSuggestionResponse(
        qa_pair_id=pair_id,
        old_answer=old_answer,
        new_answer=suggested,
        applied=True,
        message="Suggestion applied successfully",
    )


@router.post("/qa-pairs/{pair_id}/revert-suggestion", response_model=AcceptSuggestionResponse)
async def revert_suggestion(pair_id: str, db: Session = Depends(get_db)):
    """Revert the last accepted suggestion (undo)."""
    qa = db.query(QAPair).filter(QAPair.id == pair_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A pair not found")

    meta = qa.metadata_json or {}
    suggestion_data = meta.get("suggestion_applied")
    if not suggestion_data or "old_answer" not in suggestion_data:
        raise HTTPException(status_code=400, detail="No suggestion to revert")

    current_answer = qa.answer
    qa.answer = suggestion_data["old_answer"]

    # Remove the suggestion_applied record
    del meta["suggestion_applied"]
    qa.metadata_json = meta
    db.commit()

    return AcceptSuggestionResponse(
        qa_pair_id=pair_id,
        old_answer=current_answer,
        new_answer=suggestion_data["old_answer"],
        applied=True,
        message="Suggestion reverted",
    )


# ---- Auto-Approve Workflow ----

@router.post("/projects/{project_id}/review/auto-approve-workflow", response_model=ReviewSessionResponse)
async def start_auto_approve_workflow(
    project_id: str,
    request: AutoApproveWorkflowRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start automated review → fact-check → approve workflow."""
    pairs = (
        db.query(QAPair)
        .filter(QAPair.project_id == project_id, QAPair.id.in_(request.qa_pair_ids))
        .all()
    )
    if not pairs:
        raise HTTPException(status_code=404, detail="No matching Q&A pairs found")

    resolved_key = _resolve_api_key(request.provider, request.api_key_id, request.api_key, db)
    if request.provider != "ollama" and not resolved_key:
        raise HTTPException(
            status_code=400,
            detail=f"No API key available for {request.provider}. Add one in Settings.",
        )

    session = ReviewSession(
        project_id=project_id,
        provider=request.provider,
        model_name=request.model or (PROVIDER_MODELS.get(request.provider, [""])[0] if PROVIDER_MODELS.get(request.provider) else ""),
        status="pending",
        total_pairs=len(pairs),
        qa_pair_ids=[p.id for p in pairs],
        completed_pair_ids=[],
        results=[],
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    background_tasks.add_task(
        _run_auto_approve_workflow,
        session_id=session.id,
        project_id=project_id,
        provider=request.provider,
        api_key=resolved_key,
        model=request.model or session.model_name,
        ollama_url=request.ollama_url,
        speed=request.speed,
        pair_data=[{"id": p.id, "question": p.question, "answer": p.answer} for p in pairs],
        threshold=request.threshold,
        auto_accept_suggestions=request.auto_accept_suggestions,
        suggestion_threshold_min=request.suggestion_threshold_min,
        suggestion_threshold_max=request.suggestion_threshold_max,
    )

    return ReviewSessionResponse.model_validate(session)


async def _run_auto_approve_workflow(
    session_id: str,
    project_id: str,
    provider: str,
    api_key: str,
    model: str,
    ollama_url: str,
    speed: str,
    pair_data: list[dict[str, str]],
    threshold: float = 7.0,
    auto_accept_suggestions: bool = False,
    suggestion_threshold_min: float = 6.0,
    suggestion_threshold_max: float = 6.9,
) -> None:
    """Background task: review → fact-check → auto-approve workflow."""
    db = SessionLocal()
    try:
        session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
        if not session:
            return

        session.status = "in_progress"
        session.started_at = datetime.now(timezone.utc)
        session.current_message = f"Starting workflow for {len(pair_data)} pairs..."
        db.commit()

        rate_handler = RateLimitHandler(provider, model, speed=speed)
        completed_count = 0
        all_results: list[dict] = []
        completed_ids: list[str] = []
        overall_scores: list[float] = []
        auto_approved = 0
        auto_skipped = 0
        suggestions_applied = 0

        for i, pair in enumerate(pair_data):
            db.expire(session)
            if session.status == "cancelled":
                logger.info(f"Auto-approve workflow {session_id} cancelled at pair {i+1}")
                break

            pair_result: dict[str, Any] = {"qa_pair_id": pair["id"], "phase": "reviewing"}

            try:
                # Phase 1: LLM Review
                session.current_message = f"Reviewing pair {i+1}/{len(pair_data)}..."
                db.commit()

                review = await review_qa_pair(
                    question=pair["question"],
                    answer=pair["answer"],
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    ollama_url=ollama_url,
                    rate_handler=rate_handler,
                )
                pair_result.update(review)
                pair_result["phase"] = "fact_checking"

                rec = review.get("recommendation", "revise")
                if rec == "approve":
                    session.approved_count += 1
                elif rec == "reject":
                    session.rejected_count += 1
                else:
                    session.revise_count += 1

                if not review.get("error"):
                    overall_scores.append(float(review.get("overall", 0)))

                # Save review to metadata
                qa_pair = db.query(QAPair).filter(QAPair.id == pair["id"]).first()
                if qa_pair:
                    meta = qa_pair.metadata_json or {}
                    meta["llm_review"] = {
                        "provider": provider, "model": model,
                        "accuracy": review.get("accuracy", 0),
                        "completeness": review.get("completeness", 0),
                        "clarity": review.get("clarity", 0),
                        "relevance": review.get("relevance", 0),
                        "overall": review.get("overall", 0),
                        "recommendation": rec,
                        "feedback": review.get("feedback", ""),
                    }
                    qa_pair.metadata_json = meta

                # Phase 2: Fact Check
                session.current_message = f"Fact-checking pair {i+1}/{len(pair_data)}..."
                db.commit()

                fc_result = await fact_check_qa_pair(
                    question=pair["question"],
                    answer=pair["answer"],
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    ollama_url=ollama_url,
                    rate_handler=rate_handler,
                )
                pair_result["fact_check"] = fc_result
                pair_result["phase"] = "deciding"

                fc_score = float(fc_result.get("factual_accuracy", 0))
                suggested_answer = fc_result.get("suggested_answer")

                # Save fact-check to metadata
                if qa_pair:
                    meta = qa_pair.metadata_json or {}
                    meta["fact_check"] = fc_result
                    qa_pair.metadata_json = meta

                # Phase 3: Decision
                decision = "pending"
                if fc_score >= threshold:
                    decision = "approved"
                    auto_approved += 1
                    if qa_pair:
                        qa_pair.validation_status = "approved"
                elif (
                    auto_accept_suggestions
                    and suggested_answer
                    and suggestion_threshold_min <= fc_score <= suggestion_threshold_max
                ):
                    # Borderline: accept suggestion and re-evaluate
                    if qa_pair:
                        old_answer = qa_pair.answer
                        qa_pair.answer = suggested_answer
                        qa_pair.human_edited = True
                        meta = qa_pair.metadata_json or {}
                        meta["suggestion_applied"] = {
                            "old_answer": old_answer,
                            "applied_at": datetime.now(timezone.utc).isoformat(),
                            "source": "auto_approve_workflow",
                            "fact_check_score": fc_score,
                        }
                        qa_pair.metadata_json = meta
                        suggestions_applied += 1

                    # Re-evaluate with new answer
                    try:
                        session.current_message = f"Re-checking pair {i+1}/{len(pair_data)} after suggestion..."
                        db.commit()
                        re_fc = await fact_check_qa_pair(
                            question=pair["question"],
                            answer=suggested_answer,
                            provider=provider,
                            api_key=api_key,
                            model=model,
                            ollama_url=ollama_url,
                            rate_handler=rate_handler,
                        )
                        re_score = float(re_fc.get("factual_accuracy", 0))
                        pair_result["re_evaluation"] = re_fc
                        if re_score >= threshold:
                            decision = "approved"
                            auto_approved += 1
                            if qa_pair:
                                qa_pair.validation_status = "approved"
                        else:
                            decision = "pending"
                            auto_skipped += 1
                    except Exception as re_err:
                        logger.warning(f"Re-evaluation failed for {pair['id']}: {re_err}")
                        decision = "pending"
                        auto_skipped += 1
                else:
                    decision = "pending"
                    auto_skipped += 1

                pair_result["decision"] = decision
                pair_result["fact_check_score"] = fc_score
                pair_result["phase"] = "complete"

                # Audit trail
                if qa_pair:
                    meta = qa_pair.metadata_json or {}
                    meta["auto_approve_audit"] = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "review_overall": review.get("overall", 0),
                        "fact_check_score": fc_score,
                        "threshold": threshold,
                        "decision": decision,
                        "suggestion_applied": decision == "approved" and suggestions_applied > 0,
                        "provider": provider,
                        "model": model,
                    }
                    qa_pair.metadata_json = meta

            except Exception as e:
                logger.error(f"Auto-approve workflow failed for pair {pair['id']}: {e}")
                pair_result["error"] = str(e)
                pair_result["decision"] = "error"
                pair_result["phase"] = "error"
                session.failed_pairs += 1
                auto_skipped += 1

            completed_count += 1
            completed_ids.append(pair["id"])
            all_results.append(pair_result)

            session.completed_pairs = completed_count
            session.completed_pair_ids = completed_ids
            session.results = all_results
            session.avg_overall_score = (
                round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else None
            )
            session.current_message = (
                f"Processed {completed_count}/{len(pair_data)} | "
                f"Approved: {auto_approved} | Skipped: {auto_skipped}"
            )
            db.commit()

        if session.status != "cancelled":
            session.status = "completed"
            parts = [
                f"Workflow complete: {auto_approved} auto-approved, {auto_skipped} need review",
            ]
            if suggestions_applied > 0:
                parts.append(f"{suggestions_applied} suggestions applied")
            session.current_message = ". ".join(parts)
        session.completed_at = datetime.now(timezone.utc)
        session.total_cost_usd = estimate_cost(model, completed_count * 2)  # review + fact-check
        db.commit()

    except Exception as e:
        logger.exception(f"Auto-approve workflow {session_id} failed: {e}")
        try:
            session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
            if session:
                session.status = "failed"
                session.error_message = str(e)
                session.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
