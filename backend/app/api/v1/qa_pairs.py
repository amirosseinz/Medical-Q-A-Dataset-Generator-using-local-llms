"""Q&A pair CRUD, search, filter, and batch operations."""
from __future__ import annotations

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
)
from app.schemas.common import PaginatedResponse, ValidationStatus, SourceType

router = APIRouter()


@router.get("/projects/{project_id}/qa-pairs", response_model=PaginatedResponse[QAPairResponse])
def list_qa_pairs(
    project_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    source_type: str | None = None,
    validation_status: str | None = None,
    min_quality_score: float | None = Query(None, ge=0.0, le=1.0),
    max_quality_score: float | None = Query(None, ge=0.0, le=1.0),
    search: str | None = None,
    model_used: str | None = None,
    db: Session = Depends(get_db),
):
    """List Q&A pairs with filtering, search, and pagination."""
    query = db.query(QAPair).filter(QAPair.project_id == project_id)

    if source_type:
        query = query.filter(QAPair.source_type == source_type)
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

    return QAPairStats(
        total=total,
        approved=approved,
        pending=pending,
        rejected=rejected,
        avg_quality_score=round(avg_score, 4) if avg_score else None,
        by_source_type=source_counts,
        by_model=model_counts,
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
