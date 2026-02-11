"""LLMApiKey model — stores encrypted API keys for LLM providers."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Text, Boolean, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class LLMApiKey(Base):
    __tablename__ = "llm_api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    provider_name: Mapped[str] = mapped_column(String(50), nullable=False)
    api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    organization_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    models_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    available_models: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    rate_limits: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    usage_records = relationship(
        "LLMApiUsage", back_populates="api_key", cascade="all, delete-orphan", lazy="dynamic"
    )

    __table_args__ = (Index("ix_llm_api_keys_provider", "provider_name"),)

    def __repr__(self) -> str:
        return f"<LLMApiKey {self.provider_name} ({self.id[:8]})>"
