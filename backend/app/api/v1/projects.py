"""Project CRUD endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Project, QAPair, Source
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse, ProjectListItem
from app.schemas.common import PaginatedResponse

router = APIRouter()


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
        items.append(ProjectListItem(
            id=p.id,
            name=p.name,
            domain=p.domain,
            status=p.status,
            created_at=p.created_at,
            updated_at=p.updated_at,
            total_qa_pairs=qa_count,
            total_sources=src_count,
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
    db.delete(project)
    db.commit()


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
