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

    # RAG — Retrieval-Augmented Generation
    # Medical embedding model — PubMedBERT for domain-specific retrieval.
    # Falls back to all-MiniLM-L6-v2 automatically if unavailable.
    EMBEDDING_MODEL_NAME: str = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    FAISS_INDEX_DIR: str = "./data/faiss"
    RAG_TOP_K: int = 5          # number of chunks to retrieve per query
    RAG_MIN_SCORE: float = 0.25 # cosine similarity floor (0–1)
    RAG_ENABLED: bool = True    # set False to fall back to random sampling

    # GPU acceleration
    GPU_DEVICE: str = "auto"          # "auto", "cuda", "cuda:0", "cuda:1", "cpu"
    EMBEDDING_BATCH_SIZE: int = 64     # reduce if GPU OOM occurs
    GPU_MEMORY_FRACTION: float = 0.8   # max fraction of GPU memory to use

    # Generation pipeline — adaptive over-generation
    OVER_GEN_INITIAL_MULTIPLIER: float = 2.0   # start generating 2× target
    OVER_GEN_MIN_MULTIPLIER: float = 1.3       # floor after successful adaptation
    OVER_GEN_MAX_MULTIPLIER: float = 3.5       # ceiling for low-success runs
    OVER_GEN_ADAPT_INTERVAL: int = 10          # re-evaluate after N prompts
    MINI_BATCH_SIZE: int = 15                  # prompts per mini-batch

    # Quality thresholds
    MIN_QUALITY_SCORE: float = 0.4             # composite quality floor
    SEMANTIC_DUP_THRESHOLD: float = 0.92       # cosine similarity dedup cutoff
    STRING_DUP_THRESHOLD: float = 0.92         # difflib dedup cutoff
    MIN_QUESTION_LENGTH: int = 30              # chars
    MIN_ANSWER_LENGTH: int = 50                # chars

    # LLM generation defaults
    LLM_MAX_TOKENS: int = 1024                 # max output tokens for Q&A gen
    LLM_RETRY_TEMPERATURE: float = 0.5         # lower temp on retry attempts

    # Cleanup
    SECRET_KEY: str = "change-me-in-production-use-a-strong-random-secret-key"

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

    @property
    def faiss_path(self) -> Path:
        p = Path(self.FAISS_INDEX_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
