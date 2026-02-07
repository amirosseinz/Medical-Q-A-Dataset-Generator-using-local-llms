"""UserFeedback model — stores human review data for Q&A pairs."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Integer, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    qa_pair_id: Mapped[str] = mapped_column(String(36), ForeignKey("qa_pairs.id", ondelete="CASCADE"), nullable=False)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5
    issues: Mapped[list | None] = mapped_column(JSON, nullable=True)  # ["too_short", "off_topic", ...]
    corrected_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    qa_pair = relationship("QAPair", back_populates="feedback")

    __table_args__ = (
        Index("ix_user_feedback_qa_pair_id", "qa_pair_id"),
    )

    def __repr__(self) -> str:
        return f"<UserFeedback rating={self.rating}>"
