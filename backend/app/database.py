"""SQLAlchemy database engine, session factory, and connection management.

Provides the shared engine, scoped session factory, and declarative base
for all ORM models. SQLite connections enable WAL mode and foreign keys
via event listeners. Schema migrations are handled externally by Alembic.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def _get_engine():
    settings = get_settings()
    connect_args = {}
    if settings.DATABASE_URL.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args=connect_args,
        echo=settings.DEBUG,
    )
    # Enable WAL mode and foreign keys for SQLite
    if settings.DATABASE_URL.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA encoding='UTF-8';")
            cursor.close()
            # Ensure Python sqlite3 returns str (UTF-8) for TEXT columns
            dbapi_conn.text_factory = str
    return engine


engine = _get_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables from ORM metadata (dev convenience)."""
    Base.metadata.create_all(bind=engine)
    _run_migrations()


def _run_migrations():
    """Add new columns to existing tables (SQLite ALTER TABLE)."""
    import sqlite3
    from app.config import get_settings
    settings = get_settings()
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    if not db_path:
        return
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        migrations = [
            "ALTER TABLE generation_jobs ADD COLUMN generation_number INTEGER",
            "ALTER TABLE generation_jobs ADD COLUMN qa_pair_count INTEGER",
            "ALTER TABLE generation_jobs ADD COLUMN output_files JSON",
            "ALTER TABLE qa_pairs ADD COLUMN generation_job_id TEXT REFERENCES generation_jobs(id)",
            "ALTER TABLE llm_api_keys ADD COLUMN models_fetched_at DATETIME",
            "ALTER TABLE llm_api_keys ADD COLUMN model_details JSON",
            "ALTER TABLE qa_pairs ADD COLUMN provider TEXT DEFAULT 'ollama'",
            "ALTER TABLE qa_pairs ADD COLUMN source_document TEXT",
            "ALTER TABLE qa_pairs ADD COLUMN source_meta JSON",
        ]
        for sql in migrations:
            try:
                cursor.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists
        conn.commit()
        conn.close()
    except Exception:
        pass  # Non-SQLite or DB not created yet
