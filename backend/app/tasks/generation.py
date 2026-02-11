"""Celery tasks for background dataset generation.

Exposes the ``run_generation`` task consumed by Celery workers. Because
the generation pipeline is fully async, each task spins up a short-lived
event loop (``asyncio.run``) to bridge sync Celery with async service code.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models import GenerationJob
from app.services.qa_generator import GenerationPipeline
from app.utils.gpu_cleanup import release_gpu_memory

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
        # Build a user-friendly error message
        error_str = str(e)
        if "No API key configured" in error_str:
            friendly = error_str  # Already clear from APIKeyService
        elif "401" in error_str or "Unauthorized" in error_str:
            friendly = f"Authentication failed for LLM provider. Please check your API key in Settings. ({error_str})"
        elif "429" in error_str or "rate limit" in error_str.lower():
            friendly = f"Rate limit exceeded by LLM provider. Try again later or reduce max_workers. ({error_str})"
        elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
            friendly = f"Request to LLM provider timed out. The provider may be overloaded. ({error_str})"
        elif "connection" in error_str.lower() or "connect" in error_str.lower():
            friendly = f"Could not connect to LLM provider. Check your network and provider status. ({error_str})"
        else:
            friendly = f"Generation failed: {error_str}"
        try:
            job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = friendly
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
        return {"error": friendly}
    finally:
        release_gpu_memory("post-generation-task")
        db.close()
