"""Source schemas."""
from datetime import datetime
from pydantic import BaseModel
from app.schemas.common import ProcessingStatus, FileType


class SourceResponse(BaseModel):
    id: str
    project_id: str
    filepath: str | None
    filename: str
    file_type: FileType
    size_bytes: int | None
    processing_status: ProcessingStatus
    error_message: str | None
    metadata_json: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SourceUploadResponse(BaseModel):
    uploaded: list[SourceResponse]
    errors: list[str] = []
