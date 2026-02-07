"""Celery application configuration."""
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
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 24 hours
    broker_connection_retry_on_startup=True,
)


@worker_init.connect
def setup_worker_logging(**kwargs):
    """Suppress noisy HTTP loggers in Celery workers."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
