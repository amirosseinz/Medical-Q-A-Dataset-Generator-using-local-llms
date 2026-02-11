"""LLMApiUsage model — tracks LLM API call usage and costs."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class LLMApiUsage(Base):
    __tablename__ = "llm_api_usage"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    api_key_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("llm_api_keys.id", ondelete="SET NULL"), nullable=True
    )
    project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    review_session_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("review_sessions.id", ondelete="SET NULL"), nullable=True
    )
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    api_key = relationship("LLMApiKey", back_populates="usage_records")

    __table_args__ = (
        Index("ix_llm_api_usage_api_key_id", "api_key_id"),
        Index("ix_llm_api_usage_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<LLMApiUsage {self.model_used} tokens={self.input_tokens + self.output_tokens}>"
