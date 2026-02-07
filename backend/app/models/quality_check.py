"""QualityCheck model — stores individual automated quality validations for Q&A pairs."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Boolean, Float, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class QualityCheck(Base):
    __tablename__ = "quality_checks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    qa_pair_id: Mapped[str] = mapped_column(String(36), ForeignKey("qa_pairs.id", ondelete="CASCADE"), nullable=False)
    check_type: Mapped[str] = mapped_column(String(50), nullable=False)  # length | format | relevance | grammar | duplicate
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    qa_pair = relationship("QAPair", back_populates="quality_checks")

    __table_args__ = (
        Index("ix_quality_checks_qa_pair_id", "qa_pair_id"),
    )

    def __repr__(self) -> str:
        return f"<QualityCheck {self.check_type} passed={self.passed}>"
