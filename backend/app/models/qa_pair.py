"""QAPair model — stores generated question-answer pairs."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Float, Boolean, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class QAPair(Base):
    __tablename__ = "qa_pairs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # medquad | pdf_ollama | pubmed_ollama
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_template: Mapped[str | None] = mapped_column(String(50), nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending | approved | rejected
    human_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    project = relationship("Project", back_populates="qa_pairs")
    chunk = relationship("Chunk", back_populates="qa_pairs")
    quality_checks = relationship("QualityCheck", back_populates="qa_pair", cascade="all, delete-orphan", lazy="dynamic")
    feedback = relationship("UserFeedback", back_populates="qa_pair", cascade="all, delete-orphan", lazy="dynamic")

    __table_args__ = (
        Index("ix_qa_pairs_project_id", "project_id"),
        Index("ix_qa_pairs_validation_status", "validation_status"),
        Index("ix_qa_pairs_quality_score", "quality_score"),
        Index("ix_qa_pairs_source_type", "source_type"),
    )

    def __repr__(self) -> str:
        return f"<QAPair {self.id[:8]} ({self.source_type})>"
