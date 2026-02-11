"""Q&A pair CRUD, search, filter, and batch operations."""
from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import QAPair
from app.schemas.qa_pair import (
    QAPairResponse,
    QAPairUpdate,
    QAPairBatchUpdate,
    QAPairStats,
    EnhancedAnalytics,
    FileAnalytics,
)
from app.schemas.common import PaginatedResponse

router = APIRouter()


@router.get("/projects/{project_id}/qa-pairs", response_model=PaginatedResponse[QAPairResponse])
def list_qa_pairs(
    project_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    source_type: str | None = None,
    source_document: str | None = None,
    validation_status: str | None = None,
    min_quality_score: float | None = Query(None, ge=0.0, le=1.0),
    max_quality_score: float | None = Query(None, ge=0.0, le=1.0),
    search: str | None = None,
    model_used: str | None = None,
    generation_job_id: str | None = None,
    db: Session = Depends(get_db),
):
    """List Q&A pairs with filtering, search, and pagination."""
    query = db.query(QAPair).filter(QAPair.project_id == project_id)

    if generation_job_id:
        query = query.filter(QAPair.generation_job_id == generation_job_id)

    if source_type:
        query = query.filter(QAPair.source_type == source_type)
    if source_document:
        query = query.filter(QAPair.source_document == source_document)
    if validation_status:
        query = query.filter(QAPair.validation_status == validation_status)
    if min_quality_score is not None:
        query = query.filter(QAPair.quality_score >= min_quality_score)
    if max_quality_score is not None:
        query = query.filter(QAPair.quality_score <= max_quality_score)
    if model_used:
        query = query.filter(QAPair.model_used == model_used)
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (QAPair.question.ilike(search_term)) | (QAPair.answer.ilike(search_term))
        )

    total = query.count()
    items = (
        query.order_by(QAPair.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/projects/{project_id}/qa-pairs/stats", response_model=QAPairStats)
def get_qa_pair_stats(project_id: str, db: Session = Depends(get_db)):
    """Get aggregate statistics for a project's Q&A pairs."""
    total = db.query(func.count(QAPair.id)).filter(QAPair.project_id == project_id).scalar() or 0
    approved = db.query(func.count(QAPair.id)).filter(
        QAPair.project_id == project_id, QAPair.validation_status == "approved"
    ).scalar() or 0
    pending = db.query(func.count(QAPair.id)).filter(
        QAPair.project_id == project_id, QAPair.validation_status == "pending"
    ).scalar() or 0
    rejected = db.query(func.count(QAPair.id)).filter(
        QAPair.project_id == project_id, QAPair.validation_status == "rejected"
    ).scalar() or 0
    avg_score = db.query(func.avg(QAPair.quality_score)).filter(
        QAPair.project_id == project_id, QAPair.quality_score.isnot(None)
    ).scalar()

    # Source type breakdown
    source_counts = dict(
        db.query(QAPair.source_type, func.count(QAPair.id))
        .filter(QAPair.project_id == project_id)
        .group_by(QAPair.source_type)
        .all()
    )

    # Model breakdown
    model_counts = dict(
        db.query(QAPair.model_used, func.count(QAPair.id))
        .filter(QAPair.project_id == project_id, QAPair.model_used.isnot(None))
        .group_by(QAPair.model_used)
        .all()
    )

    # Source document breakdown (top 50)
    source_doc_counts = dict(
        db.query(QAPair.source_document, func.count(QAPair.id))
        .filter(QAPair.project_id == project_id, QAPair.source_document.isnot(None))
        .group_by(QAPair.source_document)
        .order_by(func.count(QAPair.id).desc())
        .limit(50)
        .all()
    )

    return QAPairStats(
        total=total,
        approved=approved,
        pending=pending,
        rejected=rejected,
        avg_quality_score=round(avg_score, 4) if avg_score else None,
        by_source_type=source_counts,
        by_model=model_counts,
        by_source_document=source_doc_counts,
    )


@router.get("/projects/{project_id}/qa-pairs/analytics", response_model=EnhancedAnalytics)
def get_enhanced_analytics(project_id: str, db: Session = Depends(get_db)):
    """Get enhanced analytics with per-file breakdown, quality histogram, and timeline."""
    pairs = (
        db.query(QAPair)
        .filter(QAPair.project_id == project_id)
        .all()
    )

    # Per-file breakdown using source_document field (preferred) or metadata fallback
    file_data: dict[str, dict] = defaultdict(lambda: {
        "pair_count": 0, "total_quality": 0.0, "quality_count": 0,
        "approved": 0, "rejected": 0, "pending": 0, "source_type": "unknown",
    })
    quality_buckets = [0] * 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    day_counts: dict[str, int] = defaultdict(int)

    for qa in pairs:
        # Per-file: prefer source_document column, fall back to metadata
        meta = qa.metadata_json or {}
        fname = qa.source_document or meta.get("original_file") or meta.get("source_filename") or meta.get("source_file") or f"({qa.source_type})"
        entry = file_data[fname]
        entry["pair_count"] += 1
        entry["source_type"] = meta.get("original_source", qa.source_type)
        if qa.quality_score is not None:
            entry["total_quality"] += qa.quality_score
            entry["quality_count"] += 1
            # Quality histogram
            bucket = min(int(qa.quality_score * 10), 9)
            quality_buckets[bucket] += 1
        if qa.validation_status == "approved":
            entry["approved"] += 1
        elif qa.validation_status == "rejected":
            entry["rejected"] += 1
        else:
            entry["pending"] += 1

        # Timeline
        if qa.created_at:
            day = qa.created_at.strftime("%Y-%m-%d")
            day_counts[day] += 1

    by_file = [
        FileAnalytics(
            filename=fname,
            source_type=data["source_type"],
            pair_count=data["pair_count"],
            avg_quality=round(data["total_quality"] / data["quality_count"], 4)
            if data["quality_count"] > 0 else None,
            approved=data["approved"],
            rejected=data["rejected"],
            pending=data["pending"],
        )
        for fname, data in sorted(file_data.items(), key=lambda x: -x[1]["pair_count"])
    ]

    quality_histogram = [
        {"range": f"{i/10:.1f}-{(i+1)/10:.1f}", "count": quality_buckets[i]}
        for i in range(10)
        if quality_buckets[i] > 0
    ]

    generation_timeline = [
        {"date": date, "count": count}
        for date, count in sorted(day_counts.items())
    ]

    return EnhancedAnalytics(
        by_file=by_file,
        quality_histogram=quality_histogram,
        generation_timeline=generation_timeline,
    )


@router.put("/qa-pairs/{pair_id}", response_model=QAPairResponse)
def update_qa_pair(pair_id: str, payload: QAPairUpdate, db: Session = Depends(get_db)):
    """Update a single Q&A pair (edit question/answer, change status)."""
    qa = db.query(QAPair).filter(QAPair.id == pair_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A pair not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "question" in update_data or "answer" in update_data:
        update_data["human_edited"] = True
    for field, value in update_data.items():
        setattr(qa, field, value)
    db.commit()
    db.refresh(qa)
    return qa


@router.post("/qa-pairs/batch-update")
def batch_update_qa_pairs(payload: QAPairBatchUpdate, db: Session = Depends(get_db)):
    """Batch approve/reject multiple Q&A pairs."""
    updated = (
        db.query(QAPair)
        .filter(QAPair.id.in_(payload.ids))
        .update({QAPair.validation_status: payload.validation_status}, synchronize_session="fetch")
    )
    db.commit()
    return {"updated": updated, "status": payload.validation_status}


@router.delete("/qa-pairs/{pair_id}", status_code=204)
def delete_qa_pair(pair_id: str, db: Session = Depends(get_db)):
    qa = db.query(QAPair).filter(QAPair.id == pair_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A pair not found")
    db.delete(qa)
    db.commit()
