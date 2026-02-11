"""LLM provider schemas for API key management, review sessions, and fact-check."""
from datetime import datetime
from pydantic import BaseModel, Field


# ---- API Key Management ----

class LLMProviderCreate(BaseModel):
    provider_name: str = Field(..., description="Provider: openai, anthropic, google")
    api_key: str = Field(..., min_length=1)
    organization_id: str | None = None
    display_name: str | None = None
    is_default: bool = False


class LLMProviderUpdate(BaseModel):
    api_key: str | None = None
    organization_id: str | None = None
    display_name: str | None = None
    is_default: bool | None = None
    enabled: bool | None = None


class LLMProviderResponse(BaseModel):
    id: str
    provider_name: str
    display_name: str | None
    organization_id: str | None
    masked_key: str
    is_valid: bool
    enabled: bool
    is_default: bool
    available_models: list[str] | None
    model_details: dict | None = None
    models_fetched_at: datetime | None = None
    rate_limits: dict | None
    last_tested_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class LLMProviderTestResult(BaseModel):
    success: bool
    message: str
    available_models: list[str] = []
    error: str | None = None


# ---- Review Sessions ----

class ReviewStartRequest(BaseModel):
    qa_pair_ids: list[str]
    provider: str = "openai"
    api_key_id: str | None = None  # Use stored key
    api_key: str | None = None  # Or provide directly
    model: str = ""
    ollama_url: str = "http://host.docker.internal:11434"
    speed: str = "normal"  # "normal" | "fast" | "slow"


class ReviewSessionResponse(BaseModel):
    id: str
    project_id: str
    provider: str
    model_name: str
    status: str
    total_pairs: int
    completed_pairs: int
    failed_pairs: int
    approved_count: int
    revise_count: int
    rejected_count: int
    avg_overall_score: float | None
    total_cost_usd: float
    current_message: str | None
    error_message: str | None
    results: list[dict] | None
    completed_pair_ids: list[str] | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


# ---- Fact-Check ----

class FactCheckRequest(BaseModel):
    qa_pair_id: str
    provider: str = "openai"
    api_key_id: str | None = None
    api_key: str | None = None
    model: str = ""
    ollama_url: str = "http://host.docker.internal:11434"


class FactCheckResponse(BaseModel):
    qa_pair_id: str
    factual_accuracy: float = 0.0
    analysis: list[str] = []
    suggested_answer: str | None = None
    confidence: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


# ---- Cost Estimation ----

class CostEstimate(BaseModel):
    provider: str
    model: str
    pair_count: int
    estimated_cost_usd: float
    estimated_time_seconds: float


# ---- Auto-Approve Workflow ----

class AutoApproveWorkflowRequest(BaseModel):
    qa_pair_ids: list[str]
    provider: str = "openai"
    api_key_id: str | None = None
    api_key: str | None = None
    model: str = ""
    ollama_url: str = "http://host.docker.internal:11434"
    speed: str = "normal"
    threshold: float = Field(default=7.0, ge=6.0, le=10.0)
    auto_accept_suggestions: bool = False
    suggestion_threshold_min: float = Field(default=6.0, ge=0.0, le=10.0)
    suggestion_threshold_max: float = Field(default=6.9, ge=0.0, le=10.0)


class AcceptSuggestionResponse(BaseModel):
    qa_pair_id: str
    old_answer: str
    new_answer: str
    applied: bool
    message: str
