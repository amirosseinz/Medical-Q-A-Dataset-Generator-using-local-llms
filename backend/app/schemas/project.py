"""Project schemas for request/response validation."""
from datetime import datetime
from pydantic import BaseModel, Field
from app.schemas.common import ProjectStatus


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    domain: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    config: dict | None = None


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    domain: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    status: ProjectStatus | None = None
    config: dict | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    domain: str
    description: str | None
    status: ProjectStatus
    config: dict | None
    created_at: datetime
    updated_at: datetime
    # Computed counts (filled by the API layer)
    total_sources: int = 0
    total_qa_pairs: int = 0
    total_approved: int = 0
    avg_quality_score: float | None = None

    model_config = {"from_attributes": True}


class ProjectListItem(BaseModel):
    id: str
    name: str
    domain: str
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    total_qa_pairs: int = 0
    total_sources: int = 0
    avg_quality_score: float | None = None

    model_config = {"from_attributes": True}
