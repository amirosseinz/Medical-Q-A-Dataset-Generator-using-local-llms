"""LLM provider management endpoints — CRUD for API keys, test connections, model refresh."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.llm_api_key import LLMApiKey
from app.services.encryption_service import encrypt_value, decrypt_value, mask_api_key
from app.services.api_key_service import clear_key_cache
from app.services.model_fetcher import fetch_models, is_cache_fresh, FALLBACK_MODELS
from app.schemas.llm_provider import (
    LLMProviderCreate,
    LLMProviderUpdate,
    LLMProviderResponse,
    LLMProviderTestResult,
)

router = APIRouter()


def _to_response(key: LLMApiKey) -> LLMProviderResponse:
    """Convert a model instance to a response, masking the API key."""
    try:
        raw_key = decrypt_value(key.api_key_encrypted)
        masked = mask_api_key(raw_key)
    except Exception:
        masked = "••••••••"

    return LLMProviderResponse(
        id=key.id,
        provider_name=key.provider_name,
        display_name=key.display_name,
        organization_id=key.organization_id,
        masked_key=masked,
        is_valid=key.is_valid,
        enabled=key.enabled,
        is_default=key.is_default,
        available_models=key.available_models,
        model_details=key.model_details,
        models_fetched_at=key.models_fetched_at,
        rate_limits=key.rate_limits,
        last_tested_at=key.last_tested_at,
        error_message=key.error_message,
        created_at=key.created_at,
        updated_at=key.updated_at,
    )


@router.get("/settings/llm-providers", response_model=list[LLMProviderResponse])
def list_providers(db: Session = Depends(get_db)):
    """List all stored LLM provider API keys."""
    keys = db.query(LLMApiKey).order_by(LLMApiKey.provider_name, LLMApiKey.created_at).all()
    return [_to_response(k) for k in keys]


@router.post("/settings/llm-providers", response_model=LLMProviderResponse, status_code=201)
async def add_provider(payload: LLMProviderCreate, db: Session = Depends(get_db)):
    """Add a new LLM provider API key (encrypted). Auto-fetches available models."""
    if payload.provider_name not in ("openai", "anthropic", "google", "openrouter"):
        raise HTTPException(status_code=400, detail="Invalid provider name")

    # If setting as default, unset other defaults for this provider
    if payload.is_default:
        db.query(LLMApiKey).filter(
            LLMApiKey.provider_name == payload.provider_name,
            LLMApiKey.is_default == True,
        ).update({"is_default": False})

    encrypted = encrypt_value(payload.api_key)
    key = LLMApiKey(
        provider_name=payload.provider_name,
        api_key_encrypted=encrypted,
        organization_id=payload.organization_id,
        display_name=payload.display_name or payload.provider_name.capitalize(),
        is_default=payload.is_default,
    )
    db.add(key)
    db.commit()
    db.refresh(key)
    clear_key_cache(payload.provider_name)

    # Auto-fetch models in background (don't fail if it doesn't work)
    try:
        result = await fetch_models(payload.provider_name, payload.api_key)
        if result["models"]:
            now = datetime.now(timezone.utc)
            key.available_models = result["models"]
            key.model_details = result["model_details"]
            key.models_fetched_at = now
            key.is_valid = True
            key.last_tested_at = now
            db.commit()
            db.refresh(key)
    except Exception:
        pass  # Non-critical — models can be fetched later

    return _to_response(key)


@router.patch("/settings/llm-providers/{provider_id}", response_model=LLMProviderResponse)
def update_provider(
    provider_id: str,
    payload: LLMProviderUpdate,
    db: Session = Depends(get_db),
):
    """Update an existing LLM provider configuration."""
    key = db.query(LLMApiKey).filter(LLMApiKey.id == provider_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Provider not found")

    update_data = payload.model_dump(exclude_unset=True)

    if "api_key" in update_data and update_data["api_key"]:
        key.api_key_encrypted = encrypt_value(update_data.pop("api_key"))
        key.is_valid = True
        key.error_message = None

    if update_data.get("is_default"):
        db.query(LLMApiKey).filter(
            LLMApiKey.provider_name == key.provider_name,
            LLMApiKey.id != key.id,
            LLMApiKey.is_default == True,
        ).update({"is_default": False})

    for field, value in update_data.items():
        setattr(key, field, value)

    key.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(key)
    clear_key_cache(key.provider_name)
    return _to_response(key)


@router.delete("/settings/llm-providers/{provider_id}", status_code=204)
def delete_provider(provider_id: str, db: Session = Depends(get_db)):
    """Delete an LLM provider API key."""
    key = db.query(LLMApiKey).filter(LLMApiKey.id == provider_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Provider not found")
    provider_name = key.provider_name
    db.delete(key)
    db.commit()
    clear_key_cache(provider_name)


@router.post(
    "/settings/llm-providers/{provider_id}/test",
    response_model=LLMProviderTestResult,
)
async def test_provider(provider_id: str, db: Session = Depends(get_db)):
    """Test an LLM provider connection and fetch available models."""
    key = db.query(LLMApiKey).filter(LLMApiKey.id == provider_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Provider not found")

    try:
        raw_key = decrypt_value(key.api_key_encrypted)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to decrypt API key")

    result = await fetch_models(key.provider_name, raw_key)
    now = datetime.now(timezone.utc)

    if result["models"]:
        key.is_valid = True
        key.error_message = None
        key.available_models = result["models"]
        key.model_details = result["model_details"]
        key.models_fetched_at = now
        key.last_tested_at = now
        db.commit()
        return LLMProviderTestResult(
            success=True,
            message=f"Connected to {key.provider_name}. Found {len(result['models'])} models.",
            available_models=result["models"],
        )
    elif result.get("error"):
        error_msg = result["error"]
        key.is_valid = False
        key.error_message = error_msg
        key.last_tested_at = now
        db.commit()
        return LLMProviderTestResult(
            success=False,
            message=f"Connection failed: {error_msg}",
            error=error_msg,
        )
    else:
        # Connected but no models found — still valid
        key.is_valid = True
        key.error_message = None
        key.last_tested_at = now
        key.available_models = FALLBACK_MODELS.get(key.provider_name, [])
        key.models_fetched_at = now
        db.commit()
        return LLMProviderTestResult(
            success=True,
            message=f"Connected to {key.provider_name}. Using default model list.",
            available_models=key.available_models or [],
        )


@router.post(
    "/settings/llm-providers/{provider_id}/refresh-models",
    response_model=LLMProviderTestResult,
)
async def refresh_models(
    provider_id: str,
    force: bool = False,
    db: Session = Depends(get_db),
):
    """Refresh the available models list for a provider.

    Uses cached data if still fresh, unless force=True.
    """
    key = db.query(LLMApiKey).filter(LLMApiKey.id == provider_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Return cached if fresh (unless forced)
    if not force and is_cache_fresh(key.models_fetched_at, key.provider_name):
        return LLMProviderTestResult(
            success=True,
            message=f"Models cached ({len(key.available_models or [])} models). Cache still fresh.",
            available_models=key.available_models or [],
        )

    try:
        raw_key = decrypt_value(key.api_key_encrypted)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to decrypt API key")

    result = await fetch_models(key.provider_name, raw_key)
    now = datetime.now(timezone.utc)

    if result["models"]:
        key.available_models = result["models"]
        key.model_details = result["model_details"]
        key.models_fetched_at = now
        db.commit()
        return LLMProviderTestResult(
            success=True,
            message=f"Refreshed {len(result['models'])} models from {key.provider_name}.",
            available_models=result["models"],
        )
    else:
        # Fetch failed — keep old cached data if any
        error_msg = result.get("error", "No models returned")
        existing = key.available_models or []
        if existing:
            return LLMProviderTestResult(
                success=False,
                message=f"Refresh failed ({error_msg}). Showing {len(existing)} cached models.",
                available_models=existing,
                error=error_msg,
            )
        # No cache, use fallback
        fallback = FALLBACK_MODELS.get(key.provider_name, [])
        key.available_models = fallback
        key.models_fetched_at = now
        db.commit()
        return LLMProviderTestResult(
            success=False,
            message=f"Refresh failed ({error_msg}). Using {len(fallback)} default models.",
            available_models=fallback,
            error=error_msg,
        )


@router.get("/settings/llm-providers/{provider_id}/models")
async def get_provider_models(provider_id: str, db: Session = Depends(get_db)):
    """Get cached available models for a provider. Auto-refreshes if cache is stale."""
    key = db.query(LLMApiKey).filter(LLMApiKey.id == provider_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Provider not found")

    # If cache is fresh, return immediately
    if is_cache_fresh(key.models_fetched_at, key.provider_name) and key.available_models:
        return {
            "models": key.available_models,
            "model_details": key.model_details or {},
            "fetched_at": key.models_fetched_at.isoformat() if key.models_fetched_at else None,
            "cached": True,
        }

    # Auto-refresh if stale
    try:
        raw_key = decrypt_value(key.api_key_encrypted)
        result = await fetch_models(key.provider_name, raw_key)
        if result["models"]:
            now = datetime.now(timezone.utc)
            key.available_models = result["models"]
            key.model_details = result["model_details"]
            key.models_fetched_at = now
            db.commit()
            return {
                "models": result["models"],
                "model_details": result["model_details"],
                "fetched_at": now.isoformat(),
                "cached": False,
            }
    except Exception:
        pass

    # Return whatever we have (cached or fallback)
    models = key.available_models or FALLBACK_MODELS.get(key.provider_name, [])
    return {
        "models": models,
        "model_details": key.model_details or {},
        "fetched_at": key.models_fetched_at.isoformat() if key.models_fetched_at else None,
        "cached": True,
    }
