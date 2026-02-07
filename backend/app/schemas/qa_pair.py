"""Q&A pair schemas for CRUD, search, filtering, and batch operations."""
from datetime import datetime
from pydantic import BaseModel, Field
from app.schemas.common import SourceType, ValidationStatus


class QAPairResponse(BaseModel):
    id: str
    project_id: str
    chunk_id: str | None
    question: str
    answer: str
    source_type: SourceType
    model_used: str | None
    prompt_template: str | None
    quality_score: float | None
    validation_status: ValidationStatus
    human_edited: bool
    metadata_json: dict | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QAPairUpdate(BaseModel):
    question: str | None = Field(None, min_length=1)
    answer: str | None = Field(None, min_length=1)
    validation_status: ValidationStatus | None = None
    human_edited: bool | None = None


class QAPairBatchUpdate(BaseModel):
    ids: list[str]
    validation_status: ValidationStatus


class QAPairFilterParams(BaseModel):
    source_type: SourceType | None = None
    validation_status: ValidationStatus | None = None
    min_quality_score: float | None = Field(None, ge=0.0, le=1.0)
    max_quality_score: float | None = Field(None, ge=0.0, le=1.0)
    search: str | None = None
    model_used: str | None = None


class QAPairStats(BaseModel):
    total: int = 0
    approved: int = 0
    pending: int = 0
    rejected: int = 0
    avg_quality_score: float | None = None
    by_source_type: dict[str, int] = {}
    by_model: dict[str, int] = {}


class ExportRequest(BaseModel):
    format: str = "csv"  # csv | json | jsonl | parquet | alpaca | openai
    validation_statuses: list[ValidationStatus] | None = None
    min_quality_score: float | None = None
    train_split: float = Field(default=0.8, ge=0.0, le=1.0)
    val_split: float = Field(default=0.1, ge=0.0, le=1.0)
    test_split: float = Field(default=0.1, ge=0.0, le=1.0)
    include_metadata: bool = False
