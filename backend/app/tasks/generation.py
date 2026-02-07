"""Celery tasks for background dataset generation."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models import GenerationJob
from app.services.qa_generator import GenerationPipeline

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="generation.run_generation")
def run_generation(self, job_id: str, project_id: str, config: dict):
    """Celery task that executes the full generation pipeline.

    This wraps the async GenerationPipeline.run() in an event loop
    so Celery (which is sync) can drive it.
    """
    db = SessionLocal()
    try:
        # Update the job with the Celery task ID
        job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
        if job:
            job.celery_task_id = self.request.id
            job.status = "in_progress"
            job.started_at = datetime.now(timezone.utc)
            db.commit()

        pipeline = GenerationPipeline(
            db=db,
            job_id=job_id,
            project_id=project_id,
            config=config,
        )

        # Run the async pipeline in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(pipeline.run())
        finally:
            loop.close()

        return result

    except Exception as e:
        logger.exception(f"Celery task error for job {job_id}: {e}")
        try:
            job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
        return {"error": str(e)}
    finally:
        db.close()
