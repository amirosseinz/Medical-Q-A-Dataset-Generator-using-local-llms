"""Aggregate API v1 router — mounts all sub-routers."""
from fastapi import APIRouter
from app.api.v1 import projects, sources, generation, qa_pairs, export, ollama, websocket

router = APIRouter(prefix="/api/v1")

router.include_router(projects.router, prefix="/projects", tags=["Projects"])
router.include_router(sources.router, tags=["Sources"])
router.include_router(generation.router, tags=["Generation"])
router.include_router(qa_pairs.router, tags=["Q&A Pairs"])
router.include_router(export.router, tags=["Export"])
router.include_router(ollama.router, prefix="/ollama", tags=["Ollama"])
router.include_router(websocket.router, tags=["WebSocket"])
