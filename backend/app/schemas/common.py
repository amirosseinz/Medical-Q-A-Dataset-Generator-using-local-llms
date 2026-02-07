"""Shared / common schemas: pagination, enums, base models."""
from datetime import datetime
from enum import Enum
from typing import Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


# ── Enums ──────────────────────────────────────────────────────────────

class ProjectStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    MEDQUAD = "medquad"
    PDF_OLLAMA = "pdf_ollama"
    PUBMED_OLLAMA = "pubmed_ollama"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class FileType(str, Enum):
    PDF = "pdf"
    XML = "xml"
    DOCX = "docx"
    PUBMED = "pubmed"


class ExportFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    ALPACA = "alpaca"
    OPENAI = "openai"


class QualityCheckType(str, Enum):
    LENGTH = "length"
    FORMAT = "format"
    RELEVANCE = "relevance"
    GRAMMAR = "grammar"
    DUPLICATE = "duplicate"


class ChunkingStrategy(str, Enum):
    WORD_COUNT = "word_count"
    PARAGRAPH = "paragraph"
    SECTION = "section"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class QuestionType(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    COMPARISON = "comparison"
    APPLICATION = "application"


# ── Pagination ─────────────────────────────────────────────────────────

class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=200)


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Common Responses ───────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
