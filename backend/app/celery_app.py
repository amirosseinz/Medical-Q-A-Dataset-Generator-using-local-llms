"""Celery application factory and worker configuration.

Defines the shared Celery instance used by all background tasks, along
with beat schedule, serialisation settings, and task routing. A
``worker_init`` signal hook eagerly loads ML models at startup so the
first task does not pay the cold-start penalty.
"""
import logging
from celery import Celery
from celery.signals import worker_init
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "medqa_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.generation"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=False,           # ACK immediately to prevent ghost re-delivery on restart
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 24 hours
    broker_connection_retry_on_startup=True,
)


@worker_init.connect
def setup_worker(**kwargs):
    """Run once when the Celery worker process starts.

    - Purge any stale tasks left in the broker queue from a previous
      unclean shutdown so they don't replay automatically.
    - Mark orphaned DB jobs (stuck in queued/in_progress) as failed
      so the frontend shows them correctly.
    - Suppress noisy HTTP loggers.
    """
    log = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # ── Purge stale tasks from Redis ───────────────────────────────────
    try:
        purged = celery_app.control.purge()
        if purged:
            log.warning(
                "Purged %d stale task(s) from broker queue on startup", purged,
            )
    except Exception as exc:
        log.warning("Could not purge broker queue: %s", exc)

    # ── Mark orphaned DB jobs as failed ────────────────────────────────
    from app.utils.startup import cleanup_orphaned_jobs
    cleanup_orphaned_jobs()
