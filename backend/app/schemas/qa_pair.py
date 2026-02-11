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
    source_type: str  # Dynamic: medquad | pdf_ollama | rag_openrouter | etc.
    source_document: str | None = None  # Specific source: filename, article title, etc.
    source_metadata: dict | None = None  # DOI, authors, PMID, file path, etc.
    model_used: str | None
    provider: str | None = None
    prompt_template: str | None
    quality_score: float | None
    validation_status: ValidationStatus
    human_edited: bool
    metadata_json: dict | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class QAPairUpdate(BaseModel):
    question: str | None = Field(None, min_length=1)
    answer: str | None = Field(None, min_length=1)
    validation_status: ValidationStatus | None = None
    human_edited: bool | None = None


class QAPairBatchUpdate(BaseModel):
    ids: list[str]
    validation_status: ValidationStatus


class QAPairFilterParams(BaseModel):
    source_type: str | None = None  # Dynamic source types
    source_document: str | None = None  # Filter by specific source document
    validation_status: ValidationStatus | None = None
    min_quality_score: float | None = Field(None, ge=0.0, le=1.0)
    max_quality_score: float | None = Field(None, ge=0.0, le=1.0)
    search: str | None = None
    model_used: str | None = None

    model_config = {"protected_namespaces": ()}


class QAPairStats(BaseModel):
    total: int = 0
    approved: int = 0
    pending: int = 0
    rejected: int = 0
    avg_quality_score: float | None = None
    by_source_type: dict[str, int] = {}
    by_model: dict[str, int] = {}
    by_source_document: dict[str, int] = {}  # Counts per specific source document


class FileAnalytics(BaseModel):
    filename: str
    source_type: str
    pair_count: int
    avg_quality: float | None = None
    approved: int = 0
    rejected: int = 0
    pending: int = 0


class EnhancedAnalytics(BaseModel):
    """Extended analytics with per-file breakdown and quality distribution."""
    by_file: list[FileAnalytics] = []
    quality_histogram: list[dict] = []  # [{range: "0.0-0.1", count: 5}, ...]
    generation_timeline: list[dict] = []  # [{date: "2025-01-15", count: 12}, ...]


class ExportRequest(BaseModel):
    format: str = "csv"  # csv | json | jsonl | parquet | alpaca | openai
    validation_statuses: list[ValidationStatus] | None = None
    min_quality_score: float | None = None
    generation_job_id: str | None = None
    train_split: float = Field(default=0.8, ge=0.0, le=1.0)
    val_split: float = Field(default=0.1, ge=0.0, le=1.0)
    test_split: float = Field(default=0.1, ge=0.0, le=1.0)
    include_metadata: bool = False
