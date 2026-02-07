# Medical Q&A Dataset Generator

A production-ready tool for generating high-quality medical question-answer datasets from diverse sources using local LLMs via [Ollama](https://ollama.ai).

## Overview

This application processes medical documents (PDFs, MedQuAD XML, DOCX) and PubMed articles, then uses local large language models to generate structured Q&A pairs suitable for training and evaluating medical AI systems.

### Key Features

- **Multi-source ingestion** — PDF, MedQuAD XML, DOCX files, and PubMed articles
- **AI-powered generation** — Uses Ollama (local LLMs) with configurable prompts, difficulty levels, and question types
- **Quality assurance** — Automated quality scoring, length/format checks, duplicate detection
- **Human review** — Approve, reject, or edit pairs with batch operations
- **Multiple export formats** — CSV, JSON, JSONL, Alpaca, OpenAI, Parquet with train/val/test splitting
- **Real-time progress** — WebSocket-based live updates during generation
- **Analytics dashboard** — Quality metrics, source distribution, validation status charts
- **Docker deployment** — One-command start with Docker Compose

## Architecture

| Component | Technology |
|-----------|------------|
| Backend API | FastAPI (Python) |
| Task Queue | Celery + Redis |
| Database | SQLite (WAL mode) |
| Frontend | React + TypeScript + Tailwind CSS |
| LLM Engine | Ollama (local) |
| Deployment | Docker Compose |

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Ollama](https://ollama.ai) installed and running with at least one model

```bash
# Install a model (if you haven't already)
ollama pull llama3
```

### Start the Application

**Linux / macOS:**
```bash
chmod +x scripts/*.sh
./scripts/start.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\start.ps1
```

The application will be available at:
- **Frontend:** http://localhost:3000
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Stop

```bash
./scripts/stop.sh        # Linux/macOS
.\scripts\stop.ps1       # Windows
```

### Reset (remove all data)

```bash
./scripts/reset.sh       # Linux/macOS
.\scripts\reset.ps1      # Windows
```

## Project Structure

```
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/v1/          # API route handlers
│   │   ├── models/          # SQLAlchemy ORM models
│   │   ├── schemas/         # Pydantic request/response schemas
│   │   ├── services/        # Business logic services
│   │   ├── tasks/           # Celery background tasks
│   │   └── utils/           # Utilities (chunking, prompts, etc.)
│   ├── alembic/             # Database migrations
│   ├── tests/               # Backend tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                # React frontend
│   ├── src/
│   │   ├── components/      # UI components (layout, ui primitives)
│   │   ├── hooks/           # Custom React hooks
│   │   ├── lib/             # API client, WebSocket, utilities
│   │   ├── pages/           # Route pages
│   │   ├── store/           # Zustand state management
│   │   └── types/           # TypeScript type definitions
│   ├── Dockerfile
│   └── nginx.conf
├── scripts/                 # Start/stop/reset/logs scripts
├── docker-compose.yml
├── .env.example
└── README.md
```

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for local development setup without Docker.

## Documentation

- [INSTALL.md](INSTALL.md) — Detailed installation guide
- [USER_GUIDE.md](USER_GUIDE.md) — How to use the application
- [API_DOCS.md](API_DOCS.md) — REST API reference
- [DEVELOPMENT.md](DEVELOPMENT.md) — Developer setup & contributing

## License

This project is for academic research purposes (LJMU MSc).
