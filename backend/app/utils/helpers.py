"""General helper utilities."""
import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename


def make_unique_filename(original_name: str) -> str:
    """Return a sanitized filename with a UUID suffix to avoid collisions."""
    safe = secure_filename(original_name)
    name, ext = os.path.splitext(safe)
    return f"{name}_{uuid.uuid4().hex[:8]}{ext}"


def file_extension(filename: str) -> str:
    """Return lowercase extension without the dot."""
    return Path(filename).suffix.lstrip(".").lower()


ALLOWED_EXTENSIONS = {"pdf", "xml", "docx"}


def is_allowed_file(filename: str) -> bool:
    return file_extension(filename) in ALLOWED_EXTENSIONS
