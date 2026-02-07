"""Application configuration using Pydantic Settings."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # Application
    APP_NAME: str = "Medical Q&A Dataset Generator"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "sqlite:///./data/medqa.db"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # Ollama
    OLLAMA_URL: str = "http://host.docker.internal:11434"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # File storage
    UPLOAD_DIR: str = "./data/uploads"
    OUTPUT_DIR: str = "./data/outputs"
    MAX_UPLOAD_SIZE_MB: int = 100

    # Celery
    CELERY_CONCURRENCY: int = 2

    # PubMed
    PUBMED_EMAIL: str = "user@example.com"
    PUBMED_RATE_LIMIT: float = 0.34  # ~3 requests/sec

    # Generation defaults
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    DEFAULT_TARGET_PAIRS: int = 1000
    DEFAULT_MAX_WORKERS: int = 3
    DEFAULT_TEMPERATURE: float = 0.7

    # Cleanup
    UPLOAD_RETENTION_DAYS: int = 7
    JOB_RETENTION_DAYS: int = 30
    MAX_PROJECT_STORAGE_MB: int = 500

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_path(self) -> Path:
        p = Path(self.OUTPUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
