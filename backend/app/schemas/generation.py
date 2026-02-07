"""Generation configuration and job schemas."""
from datetime import datetime
from pydantic import BaseModel, Field
from app.schemas.common import (
    JobStatus,
    ChunkingStrategy,
    DifficultyLevel,
    QuestionType,
)


class GenerationConfig(BaseModel):
    """Configuration for a dataset generation run."""
    # Required fields
    medical_terms: str = Field(default="", description="Comma-separated medical keywords (auto-filled from project domain if empty)")
    email: str = Field(default="", description="Email for PubMed API")

    # Sources
    use_pubmed: bool = False
    use_ollama: bool = True

    # Ollama
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # Generation params
    target_pairs: int = Field(default=1000, ge=1, le=50000)
    pubmed_retmax: int = Field(default=1000, ge=1, le=10000)
    chunk_size: int = Field(default=500, ge=50, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    max_workers: int = Field(default=3, ge=1, le=20)
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.WORD_COUNT

    # Quality
    min_quality_score: float = Field(default=0.6, ge=0.0, le=1.0)

    # Question diversity
    difficulty_levels: list[DifficultyLevel] = [DifficultyLevel.INTERMEDIATE]
    question_types: list[QuestionType] = [QuestionType.FACTUAL, QuestionType.REASONING]


class GenerationStartResponse(BaseModel):
    job_id: str
    project_id: str
    message: str


class JobProgressResponse(BaseModel):
    job_id: str
    project_id: str
    status: JobStatus
    progress_pct: int
    current_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    estimated_time_remaining_seconds: float | None = None


class JobResponse(BaseModel):
    id: str
    project_id: str
    celery_task_id: str | None
    status: JobStatus
    progress_pct: int
    current_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    config: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}
