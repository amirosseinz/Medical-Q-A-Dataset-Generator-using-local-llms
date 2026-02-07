"""WebSocket endpoint for real-time job progress updates."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    """WebSocket endpoint that streams generation progress in real-time.

    Subscribes to a Redis pub/sub channel ``job:{job_id}`` and forwards
    messages to the connected WebSocket client.
    """
    await websocket.accept()

    settings = get_settings()
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.REDIS_URL)
        pubsub = r.pubsub()
        await pubsub.subscribe(f"job:{job_id}")

        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    await websocket.send_text(data)

                    # If job is complete/failed/cancelled, close after sending
                    try:
                        parsed = json.loads(data)
                        if parsed.get("status") in ("completed", "failed", "cancelled"):
                            await asyncio.sleep(0.5)
                            break
                    except json.JSONDecodeError:
                        pass
                else:
                    # Send heartbeat to detect disconnection
                    try:
                        await websocket.send_text(json.dumps({"heartbeat": True}))
                    except Exception:
                        break
                    await asyncio.sleep(1)

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for job {job_id}")
        finally:
            await pubsub.unsubscribe(f"job:{job_id}")
            await r.close()

    except ImportError:
        # redis.asyncio not available — fall back to polling DB
        logger.warning("redis.asyncio not available; falling back to DB polling for WebSocket")
        await _poll_db_fallback(websocket, job_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _poll_db_fallback(websocket: WebSocket, job_id: str):
    """Fallback: poll the database for progress updates when Redis is unavailable."""
    from app.database import SessionLocal
    from app.models import GenerationJob

    try:
        while True:
            db = SessionLocal()
            try:
                job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
                if not job:
                    await websocket.send_text(json.dumps({"error": "Job not found"}))
                    break

                payload = {
                    "job_id": job.id,
                    "percentage": job.progress_pct,
                    "message": job.current_message or "Processing...",
                    "status": job.status,
                }
                await websocket.send_text(json.dumps(payload))

                if job.status in ("completed", "failed", "cancelled"):
                    break
            finally:
                db.close()

            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
