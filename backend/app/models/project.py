"""Project model — top-level entity that groups sources, chunks, Q&A pairs, and jobs."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")  # draft | active | archived
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    sources = relationship("Source", back_populates="project", cascade="all, delete-orphan", lazy="dynamic")
    chunks = relationship("Chunk", back_populates="project", cascade="all, delete-orphan", lazy="dynamic")
    qa_pairs = relationship("QAPair", back_populates="project", cascade="all, delete-orphan", lazy="dynamic")
    generation_jobs = relationship("GenerationJob", back_populates="project", cascade="all, delete-orphan", lazy="dynamic")

    __table_args__ = (
        Index("ix_projects_status", "status"),
        Index("ix_projects_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Project {self.name!r} ({self.id[:8]})>"
