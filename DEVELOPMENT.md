# Development Guide

## Local Development (without Docker)

### Backend

**Requirements:** Python 3.11+, Redis

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="sqlite:///./data/medqa.db"
export REDIS_URL="redis://localhost:6379/0"
export OLLAMA_URL="http://localhost:11434"
export UPLOAD_DIR="./data/uploads"
export OUTPUT_DIR="./data/outputs"

# Run the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal, start the Celery worker
celery -A app.celery_app worker --loglevel=info --concurrency=2
```

### Frontend

**Requirements:** Node.js 20+

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (proxies API to localhost:8000)
npm run dev
```

The frontend dev server runs on http://localhost:3000 with hot module replacement.

### Redis

Install and run Redis locally:

- **macOS:** `brew install redis && redis-server`
- **Linux:** `sudo apt install redis-server && redis-server`
- **Windows:** Use WSL or [Memurai](https://www.memurai.com/)

### Ollama

Ensure Ollama is running: `ollama serve`

## Database Migrations

Using Alembic:

```bash
cd backend

# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

## Running Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_utils/test_chunking.py
```

## Code Style

### Backend
- Python 3.11+ type hints throughout
- Async/await for I/O operations
- Pydantic for all request/response validation

### Frontend
- TypeScript strict mode
- Functional components with hooks
- TanStack Query for server state
- Zustand for client state

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite over PostgreSQL | Single-user local tool, simpler deployment |
| Celery+Redis over threading | Proper job lifecycle, cancellation, monitoring |
| PyMuPDF over PyPDF2 | Better text extraction, OCR fallback |
| httpx over requests | Async HTTP for non-blocking PubMed calls |
| TanStack Query over Redux | Server-state focused, automatic caching |
| shadcn/ui | Composable, accessible, customisable primitives |

## Adding a New Export Format

1. Add format to `ExportFormat` enum in `backend/app/schemas/common.py`
2. Implement export function in `backend/app/services/export_service.py`
3. Add case to `export_dataset()` dispatcher
4. Add option to frontend `SelectItem` in `ProjectDetailPage.tsx`

## Adding a New Source Type

1. Add type to `FileType` enum in `backend/app/schemas/common.py`
2. Implement extraction in `backend/app/services/document_processor.py`
3. Update allowed extensions in `backend/app/utils/helpers.py`
4. Update file accept config in `ProjectDetailPage.tsx` dropzone
