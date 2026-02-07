"""Generation job endpoints — start, cancel, and check progress."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Project, GenerationJob
from app.schemas.generation import (
    GenerationConfig,
    GenerationStartResponse,
    JobProgressResponse,
    JobResponse,
)
from app.tasks.generation import run_generation

router = APIRouter()


@router.post("/projects/{project_id}/generate", response_model=GenerationStartResponse)
def start_generation(
    project_id: str,
    config: GenerationConfig,
    db: Session = Depends(get_db),
):
    """Start a dataset generation job for a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Auto-fill medical_terms from project domain if not provided
    if not config.medical_terms.strip():
        config.medical_terms = project.domain or "medical conditions"

    # Auto-fill email from settings if not provided
    if not config.email.strip():
        from app.config import get_settings
        config.email = get_settings().PUBMED_EMAIL or "user@example.com"

    # Check for existing active jobs
    active = (
        db.query(GenerationJob)
        .filter(
            GenerationJob.project_id == project_id,
            GenerationJob.status.in_(["queued", "in_progress"]),
        )
        .first()
    )
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Project already has an active job ({active.id[:8]}). Cancel it first.",
        )

    # Create job record
    job = GenerationJob(
        project_id=project_id,
        status="queued",
        config=config.model_dump(),
    )
    db.add(job)

    # Update project
    project.status = "active"
    project.config = config.model_dump()
    db.commit()
    db.refresh(job)

    # Dispatch Celery task
    celery_task = run_generation.delay(job.id, project_id, config.model_dump())
    job.celery_task_id = celery_task.id
    db.commit()

    return GenerationStartResponse(
        job_id=job.id,
        project_id=project_id,
        message="Generation job queued successfully",
    )


@router.get("/jobs/{job_id}/progress", response_model=JobProgressResponse)
def get_job_progress(job_id: str, db: Session = Depends(get_db)):
    """Get current progress for a generation job."""
    job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Calculate estimated time remaining
    estimated = None
    if job.started_at and job.progress_pct > 0 and job.status == "in_progress":
        elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()
        if job.progress_pct < 100:
            total_estimated = elapsed / (job.progress_pct / 100)
            estimated = max(0, total_estimated - elapsed)

    return JobProgressResponse(
        job_id=job.id,
        project_id=job.project_id,
        status=job.status,
        progress_pct=job.progress_pct,
        current_message=job.current_message,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        estimated_time_remaining_seconds=estimated,
    )


@router.delete("/jobs/{job_id}", status_code=200)
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a running generation job."""
    job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("queued", "in_progress"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in '{job.status}' state")

    job.status = "cancelled"
    job.completed_at = datetime.now(timezone.utc)
    job.current_message = "Cancelled by user"
    db.commit()

    # Revoke Celery task if possible
    if job.celery_task_id:
        try:
            from app.celery_app import celery_app
            celery_app.control.revoke(job.celery_task_id, terminate=True)
        except Exception:
            pass

    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/projects/{project_id}/jobs", response_model=list[JobResponse])
def list_project_jobs(project_id: str, db: Session = Depends(get_db)):
    """List all generation jobs for a project."""
    jobs = (
        db.query(GenerationJob)
        .filter(GenerationJob.project_id == project_id)
        .order_by(GenerationJob.created_at.desc())
        .all()
    )
    return jobs
