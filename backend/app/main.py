"""FastAPI application entry point and lifespan management.

Configures CORS, registers API routers, and manages the application
lifespan (database table creation, startup logging). Serves as the
single top-level module that wires together all sub-packages.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import create_tables
from app.api.v1.router import router as v1_router


def _setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy third-party HTTP loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _log_gpu_status():
    """Log GPU availability and details on startup."""
    log = logging.getLogger(__name__)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_version = torch.version.cuda or "unknown"
            log.info(f"GPU available: {gpu_name} ({vram_gb:.1f} GB VRAM, CUDA {cuda_version})")
            log.info(f"PyTorch CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
        else:
            log.info("No CUDA GPU detected — using CPU for embeddings and FAISS")
    except ImportError:
        log.info("PyTorch not installed — GPU acceleration unavailable")
    except Exception as e:
        log.warning(f"GPU detection error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup: create directories and DB tables."""
    settings = get_settings()
    _setup_logging(settings.LOG_LEVEL)

    # Ensure data directories exist
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.output_path.mkdir(parents=True, exist_ok=True)

    # Ensure DB directory exists
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    if db_path:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create tables
    create_tables()
    logging.getLogger(__name__).info("Database tables ready")

    # Mark orphaned jobs from previous unclean shutdown as failed
    from app.utils.startup import cleanup_orphaned_jobs
    cleanup_orphaned_jobs()

    # Log GPU status on startup
    _log_gpu_status()

    yield  # Application runs here

    # Graceful shutdown: close shared HTTP clients
    from app.services.http_client_manager import close_all_clients
    await close_all_clients()
    logging.getLogger(__name__).info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # UTF-8 Content-Type enforcement — ensures all text-based responses
    # include charset=utf-8 to prevent encoding/display issues
    @app.middleware("http")
    async def enforce_utf8_charset(request: Request, call_next):
        response = await call_next(request)
        ct = response.headers.get("content-type", "")
        # Add charset=utf-8 to JSON and text responses that lack it
        if ("charset" not in ct) and any(
            t in ct for t in ("application/json", "text/html", "text/plain")
        ):
            response.headers["content-type"] = ct + "; charset=utf-8"
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logging.getLogger(__name__).exception(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
            media_type="application/json; charset=utf-8",
        )

    # Health check
    @app.get("/health", tags=["Health"])
    async def health():
        from datetime import datetime, timezone
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # GPU status endpoint
    @app.get("/gpu-status", tags=["Health"])
    async def gpu_status():
        info: dict = {"gpu_available": False, "device": "cpu"}
        try:
            import torch
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_available"] = True
                info["device"] = "cuda"
                info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["vram_total_gb"] = round(props.total_memory / (1024 ** 3), 1)
                info["cuda_version"] = torch.version.cuda
                info["cudnn_version"] = str(torch.backends.cudnn.version())
                info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
                info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)
        except ImportError:
            info["pytorch_version"] = None
        except Exception as e:
            info["error"] = str(e)
        return info

    # Mount API routes
    app.include_router(v1_router)

    return app


app = create_app()
