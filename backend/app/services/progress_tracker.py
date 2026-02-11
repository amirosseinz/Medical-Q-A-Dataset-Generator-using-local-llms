"""Unified progress tracking for long-running generation pipelines.

Provides a single ``ProgressTracker`` interface that broadcasts stage
progress over Redis pub/sub (for live WebSocket updates) and persists
snapshots to the database (for resilience across reconnects). All
pipeline stages report through this instead of accessing Redis or the
GenerationJob model directly.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis
from sqlalchemy.orm import Session

from app.models import GenerationJob

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Single entry-point for progress reporting throughout the generation pipeline.

    Primary: Redis pub/sub for real-time WebSocket delivery.
    Secondary: GenerationJob DB row for persistence / polling fallback.
    """

    def __init__(self, db: Session, job_id: str, redis_url: str):
        self.db = db
        self.job_id = job_id
        self._redis: redis.Redis | None = None
        self._redis_url = redis_url

        # Running statistics (populated by the pipeline as it progresses)
        self.stats: dict[str, int] = {
            "prompts_sent": 0,
            "prompts_succeeded": 0,
            "prompts_failed": 0,
            "pairs_accepted": 0,
            "pairs_rejected": 0,
            "duplicates": 0,
            "semantic_dups": 0,
            "quality_failures": 0,
            "truncated": 0,
            "not_found": 0,
        }

    # ── Redis connection (lazy, tolerant of failure) ───────────────────

    @property
    def redis_client(self) -> redis.Redis | None:
        if self._redis is None:
            try:
                self._redis = redis.from_url(self._redis_url)
            except Exception:
                logger.warning("Could not connect to Redis for progress updates")
        return self._redis

    # ── Public API ─────────────────────────────────────────────────────

    def update(
        self,
        message: str,
        percentage: int,
        status: str = "in_progress",
    ) -> None:
        """Push a progress update to both DB and Redis."""
        self._update_db(message, percentage, status)
        self._publish_redis(message, percentage, status)
        logger.info("Job %s: %s (%d%%)", self.job_id[:8], message, percentage)

    def is_cancelled(self) -> bool:
        """Check if the user has cancelled the job."""
        try:
            job = self.db.query(GenerationJob).filter(
                GenerationJob.id == self.job_id
            ).first()
            return job is not None and job.status == "cancelled"
        except Exception:
            return False

    def finish_completed(self, result: dict[str, Any]) -> None:
        """Mark the job as completed with final results."""
        try:
            job = self.db.query(GenerationJob).filter(
                GenerationJob.id == self.job_id
            ).first()
            if job:
                job.status = "completed"
                job.progress_pct = 100
                job.current_message = "Dataset generation completed!"
                job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
        except Exception as e:
            logger.error("DB completion update failed: %s", e)

        self._publish_redis(
            "Dataset generation completed!", 100, "completed", extra={"results": result},
        )

    def finish_failed(self, error_msg: str) -> None:
        """Mark the job as failed."""
        try:
            job = self.db.query(GenerationJob).filter(
                GenerationJob.id == self.job_id
            ).first()
            if job:
                job.status = "failed"
                job.error_message = error_msg
                job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
        except Exception:
            pass

        self._publish_redis(f"Error: {error_msg}", 0, "failed")

    def finish_cancelled(self) -> None:
        """Record cancellation."""
        self._publish_redis("Job cancelled by user", 0, "cancelled")

    # ── Internal helpers ───────────────────────────────────────────────

    def _update_db(self, message: str, percentage: int, status: str) -> None:
        try:
            job = self.db.query(GenerationJob).filter(
                GenerationJob.id == self.job_id
            ).first()
            if job:
                job.progress_pct = percentage
                job.current_message = message
                if percentage > 0 and job.status == "queued":
                    job.status = "in_progress"
                    job.started_at = datetime.now(timezone.utc)
                self.db.commit()
        except Exception as e:
            logger.error("DB progress update failed: %s", e)

    def _publish_redis(
        self,
        message: str,
        percentage: int,
        status: str,
        extra: dict | None = None,
    ) -> None:
        try:
            rc = self.redis_client
            if rc:
                payload: dict[str, Any] = {
                    "job_id": self.job_id,
                    "percentage": percentage,
                    "message": message,
                    "status": status,
                    "stats": self.stats,
                }
                if extra:
                    payload.update(extra)
                rc.publish(f"job:{self.job_id}", json.dumps(payload))
        except Exception:
            pass
