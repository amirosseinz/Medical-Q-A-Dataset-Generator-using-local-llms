"""Startup cleanup utilities shared between FastAPI lifespan and Celery worker_init.

Centralises the orphaned-job recovery logic so it is not duplicated across
entry points.  Both the API server and the task worker call
``cleanup_orphaned_jobs()`` on startup to mark stale jobs as failed, ensuring
the frontend never shows phantom active jobs that can never complete.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def cleanup_orphaned_jobs() -> int:
    """Mark any jobs stuck in queued/in_progress as failed.

    Returns the number of orphaned jobs cleaned up.

    This handles unclean shutdowns where the worker was killed mid-task.
    Without this, the frontend would show phantom "active" jobs that
    can never complete, and the active-job guard would block new generation.
    """
    from app.database import SessionLocal
    from app.models import GenerationJob

    count = 0
    db = SessionLocal()
    try:
        orphaned = (
            db.query(GenerationJob)
            .filter(GenerationJob.status.in_(["queued", "in_progress"]))
            .all()
        )
        for job in orphaned:
            logger.warning(
                "Marking orphaned job %s (status=%s) as failed on startup",
                job.id[:8], job.status,
            )
            job.status = "failed"
            job.error_message = (
                "Job was interrupted by a server restart. "
                "Please start a new generation."
            )
            job.completed_at = datetime.now(timezone.utc)
        if orphaned:
            db.commit()
            count = len(orphaned)
            logger.info("Cleaned up %d orphaned job(s)", count)
    except Exception as exc:
        logger.warning("Could not clean up orphaned jobs: %s", exc)
    finally:
        db.close()
    return count
