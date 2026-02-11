"""Project CRUD endpoints."""
from __future__ import annotations

import logging
import shutil

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Project, QAPair, Source, Chunk, GenerationJob, QualityCheck
from app.models.review_session import ReviewSession
from app.models.user_feedback import UserFeedback
from app.models.llm_api_usage import LLMApiUsage
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse, ProjectListItem
from app.schemas.common import PaginatedResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    project = Project(
        name=payload.name,
        domain=payload.domain,
        description=payload.description,
        config=payload.config,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _enrich_project(project, db)


@router.get("", response_model=PaginatedResponse[ProjectListItem])
def list_projects(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    status: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Project)
    if status:
        query = query.filter(Project.status == status)
    total = query.count()
    projects = (
        query.order_by(Project.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    items = []
    for p in projects:
        qa_count = db.query(func.count(QAPair.id)).filter(QAPair.project_id == p.id).scalar() or 0
        src_count = db.query(func.count(Source.id)).filter(Source.project_id == p.id).scalar() or 0
        avg_score = db.query(func.avg(QAPair.quality_score)).filter(
            QAPair.project_id == p.id, QAPair.quality_score.isnot(None)
        ).scalar()
        items.append(ProjectListItem(
            id=p.id,
            name=p.name,
            domain=p.domain,
            status=p.status,
            created_at=p.created_at,
            updated_at=p.updated_at,
            total_qa_pairs=qa_count,
            total_sources=src_count,
            avg_quality_score=round(avg_score, 4) if avg_score else None,
        ))
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return _enrich_project(project, db)


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    db.commit()
    db.refresh(project)
    return _enrich_project(project, db)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    settings = get_settings()

    try:
        # 1. Cancel any running/queued generation jobs via Celery
        active_jobs = (
            db.query(GenerationJob)
            .filter(
                GenerationJob.project_id == project_id,
                GenerationJob.status.in_(["queued", "in_progress"]),
            )
            .all()
        )
        for job in active_jobs:
            if job.celery_task_id:
                try:
                    from app.celery_app import celery_app
                    celery_app.control.revoke(job.celery_task_id, terminate=True)
                    logger.info("Revoked Celery task %s for project %s", job.celery_task_id, project_id)
                except Exception as exc:
                    logger.warning("Failed to revoke task %s: %s", job.celery_task_id, exc)
            job.status = "cancelled"
        if active_jobs:
            db.flush()

        # 2. Remove filesystem artefacts
        for dir_path in [
            settings.upload_path / project_id,
            settings.faiss_path / project_id,
            settings.output_path / project_id,
        ]:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    logger.info("Removed directory %s", dir_path)
                except Exception as exc:
                    logger.warning("Failed to remove %s: %s", dir_path, exc)

        # 3. Manually delete child records in correct dependency order
        #    (SQLite's ALTER TABLE migrations may lack ON DELETE clauses)
        qa_pair_ids = [
            r[0] for r in db.query(QAPair.id).filter(QAPair.project_id == project_id).all()
        ]
        if qa_pair_ids:
            # Delete quality_checks & user_feedback that reference qa_pairs
            db.query(QualityCheck).filter(QualityCheck.qa_pair_id.in_(qa_pair_ids)).delete(synchronize_session=False)
            db.query(UserFeedback).filter(UserFeedback.qa_pair_id.in_(qa_pair_ids)).delete(synchronize_session=False)
        # Delete QA pairs themselves
        db.query(QAPair).filter(QAPair.project_id == project_id).delete(synchronize_session=False)

        # Delete LLM API usage records referencing this project or its review sessions
        review_session_ids = [
            r[0] for r in db.query(ReviewSession.id).filter(ReviewSession.project_id == project_id).all()
        ]
        db.query(LLMApiUsage).filter(LLMApiUsage.project_id == project_id).delete(synchronize_session=False)
        if review_session_ids:
            db.query(LLMApiUsage).filter(LLMApiUsage.review_session_id.in_(review_session_ids)).delete(synchronize_session=False)

        # Delete review sessions
        db.query(ReviewSession).filter(ReviewSession.project_id == project_id).delete(synchronize_session=False)

        # Delete generation jobs (qa_pairs already removed)
        db.query(GenerationJob).filter(GenerationJob.project_id == project_id).delete(synchronize_session=False)

        # Delete chunks and sources
        db.query(Chunk).filter(Chunk.project_id == project_id).delete(synchronize_session=False)
        db.query(Source).filter(Source.project_id == project_id).delete(synchronize_session=False)

        # 4. Finally delete the project itself
        db.delete(project)
        db.commit()
        logger.info("Deleted project %s (%s)", project.name, project_id)

    except Exception as exc:
        db.rollback()
        logger.exception("Failed to delete project %s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {exc}")


def _enrich_project(project: Project, db: Session) -> ProjectResponse:
    """Add computed counts to a project response."""
    qa_count = db.query(func.count(QAPair.id)).filter(QAPair.project_id == project.id).scalar() or 0
    approved = db.query(func.count(QAPair.id)).filter(
        QAPair.project_id == project.id, QAPair.validation_status == "approved"
    ).scalar() or 0
    avg_score = db.query(func.avg(QAPair.quality_score)).filter(
        QAPair.project_id == project.id, QAPair.quality_score.isnot(None)
    ).scalar()
    src_count = db.query(func.count(Source.id)).filter(Source.project_id == project.id).scalar() or 0

    return ProjectResponse(
        id=project.id,
        name=project.name,
        domain=project.domain,
        description=project.description,
        status=project.status,
        config=project.config,
        created_at=project.created_at,
        updated_at=project.updated_at,
        total_sources=src_count,
        total_qa_pairs=qa_count,
        total_approved=approved,
        avg_quality_score=round(avg_score, 4) if avg_score else None,
    )
