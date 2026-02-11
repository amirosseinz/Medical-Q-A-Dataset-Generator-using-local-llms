"""File upload / source management endpoints."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Project, Source
from app.schemas.source import SourceResponse, SourceUploadResponse
from app.utils.helpers import make_unique_filename, is_allowed_file

router = APIRouter()


@router.post("/projects/{project_id}/sources/upload", response_model=SourceUploadResponse)
async def upload_sources(
    project_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Upload one or more files (PDF, XML, DOCX) to a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    settings = get_settings()
    upload_dir = settings.upload_path / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded: list[SourceResponse] = []
    errors: list[str] = []

    for f in files:
        if not f.filename or not is_allowed_file(f.filename):
            errors.append(f"Rejected '{f.filename}': unsupported file type")
            continue

        unique_name = make_unique_filename(f.filename)
        dest = upload_dir / unique_name

        try:
            content = await f.read()
            # Check size
            if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                errors.append(f"'{f.filename}' exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")
                continue

            dest.write_bytes(content)

            ext = Path(f.filename).suffix.lower().lstrip(".")
            source = Source(
                project_id=project_id,
                filepath=str(dest),
                filename=f.filename,
                file_type=ext,
                size_bytes=len(content),
                processing_status="pending",
            )
            db.add(source)
            db.flush()

            uploaded.append(SourceResponse(
                id=source.id,
                project_id=source.project_id,
                filepath=source.filepath,
                filename=source.filename,
                file_type=source.file_type,
                size_bytes=source.size_bytes,
                processing_status=source.processing_status,
                error_message=source.error_message,
                metadata_json=source.metadata_json,
                created_at=source.created_at,
            ))
        except Exception as e:
            errors.append(f"Failed to save '{f.filename}': {e}")

    db.commit()
    return SourceUploadResponse(uploaded=uploaded, errors=errors)


@router.get("/projects/{project_id}/sources", response_model=list[SourceResponse])
def list_sources(project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    sources = db.query(Source).filter(Source.project_id == project_id).order_by(Source.created_at.desc()).all()
    return sources


@router.delete("/sources/{source_id}", status_code=204)
def delete_source(source_id: str, db: Session = Depends(get_db)):
    source = db.query(Source).filter(Source.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    # Delete file from disk
    if source.filepath and os.path.exists(source.filepath):
        try:
            os.remove(source.filepath)
        except Exception:
            pass
    db.delete(source)
    db.commit()
