"""ReviewSession model — tracks LLM review sessions for progress and resumability."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Text, Integer, Float, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ReviewSession(Base):
    __tablename__ = "review_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # pending | in_progress | completed | failed | cancelled | rate_limited

    total_pairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_pairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_pairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    approved_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    revise_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    avg_overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    current_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Track which pairs to review and results
    qa_pair_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    completed_pair_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    results: Mapped[list | None] = mapped_column(JSON, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_review_sessions_project_id", "project_id"),
        Index("ix_review_sessions_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<ReviewSession {self.id[:8]} ({self.status})>"
