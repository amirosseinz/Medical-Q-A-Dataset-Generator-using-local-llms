"""Ollama connectivity and model listing endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.ollama_service import test_connection
from app.services.http_client_manager import get_http_client
from app.config import get_settings

router = APIRouter()


class OllamaTestRequest(BaseModel):
    ollama_url: str | None = None


class OllamaTestResponse(BaseModel):
    success: bool
    models: list[str] = []
    error: str | None = None
    message: str | None = None


@router.post("/test", response_model=OllamaTestResponse)
async def test_ollama(request: OllamaTestRequest):
    """Test connectivity to an Ollama server and list available models."""
    settings = get_settings()
    url = request.ollama_url or settings.OLLAMA_URL
    result = await test_connection(url)
    if result["success"]:
        return OllamaTestResponse(
            success=True,
            models=result["models"],
            message=f"Connected! Found {len(result['models'])} models.",
        )
    return OllamaTestResponse(success=False, error=result["error"])


class OllamaModelInfo(BaseModel):
    name: str
    size: int = 0
    modified_at: str = ""


class OllamaStatusResponse(BaseModel):
    connected: bool
    url: str
    models: list[OllamaModelInfo] = []


@router.get("/status", response_model=OllamaStatusResponse)
async def ollama_status():
    """Return current Ollama connection status and available models."""
    settings = get_settings()
    url = settings.OLLAMA_URL
    # Get full model info from Ollama
    base_url = url.rstrip("/")
    try:
        client = get_http_client("ollama")
        resp = await client.get(f"{base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        models = [
            OllamaModelInfo(
                name=m.get("name", "Unknown"),
                size=m.get("size", 0),
                modified_at=m.get("modified_at", ""),
            )
            for m in data.get("models", [])
        ]
        return OllamaStatusResponse(connected=True, url=url, models=models)
    except Exception:
        return OllamaStatusResponse(connected=False, url=url, models=[])


@router.get("/models", response_model=OllamaTestResponse)
async def list_models(ollama_url: str | None = None):
    """List available Ollama models using default or provided URL."""
    settings = get_settings()
    url = ollama_url or settings.OLLAMA_URL
    result = await test_connection(url)
    if result["success"]:
        return OllamaTestResponse(success=True, models=result["models"])
    return OllamaTestResponse(success=False, error=result["error"])
