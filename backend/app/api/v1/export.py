"""Export endpoints — download datasets in various formats."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Project, QAPair
from app.schemas.qa_pair import ExportRequest
from app.services.export_service import (
    QAPairForExport,
    export_dataset,
    split_dataset,
    EXTENSIONS,
)

router = APIRouter()


@router.post("/projects/{project_id}/export")
def export_project(
    project_id: str,
    request: ExportRequest,
    db: Session = Depends(get_db),
):
    """Export a project's Q&A pairs in the specified format."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Build query with filters
    query = db.query(QAPair).filter(QAPair.project_id == project_id)
    if request.generation_job_id:
        query = query.filter(QAPair.generation_job_id == request.generation_job_id)
    if request.validation_statuses:
        query = query.filter(QAPair.validation_status.in_([s.value for s in request.validation_statuses]))
    if request.min_quality_score is not None:
        query = query.filter(QAPair.quality_score >= request.min_quality_score)

    qa_records = query.all()
    if not qa_records:
        raise HTTPException(status_code=404, detail="No Q&A pairs match the filter criteria")

    pairs = [
        QAPairForExport(
            question=qa.question,
            answer=qa.answer,
            source_type=qa.source_type,
            quality_score=qa.quality_score,
            validation_status=qa.validation_status,
            metadata=qa.metadata_json,
        )
        for qa in qa_records
    ]

    settings = get_settings()
    output_dir = settings.output_path / project_id
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain_slug = project.domain.replace(" ", "_").replace("/", "_")

    # Handle train/val/test split
    if request.train_split < 1.0:
        splits = split_dataset(pairs, request.train_split, request.val_split, request.test_split)
        files = {}
        for split_name, split_pairs in splits.items():
            if split_pairs:
                base = f"{domain_slug}_{timestamp}_{split_name}"
                path = export_dataset(split_pairs, request.format, output_dir, base)
                files[split_name] = path.name
        return {
            "message": f"Exported {len(pairs)} pairs with train/val/test split",
            "files": files,
            "total_pairs": len(pairs),
            "download_base": f"/api/v1/projects/{project_id}/download",
        }
    else:
        base = f"{domain_slug}_{timestamp}"
        path = export_dataset(pairs, request.format, output_dir, base)
        return {
            "message": f"Exported {len(pairs)} pairs as {request.format}",
            "filename": path.name,
            "total_pairs": len(pairs),
            "download_url": f"/api/v1/projects/{project_id}/download/{path.name}",
        }


@router.get("/projects/{project_id}/download/{filename}")
def download_file(project_id: str, filename: str):
    """Download an exported dataset file."""
    settings = get_settings()
    file_path = settings.output_path / project_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    # Security: ensure we're not serving files outside the output dir
    try:
        file_path.resolve().relative_to(settings.output_path.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )
