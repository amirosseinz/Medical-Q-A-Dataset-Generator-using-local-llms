# LLM-driven medical Q&A dataset generation with retrieval grounding and quality filtering

**An end-to-end AI pipeline for generating, validating, and curating high-quality medical question-answer datasets from scientific literature.**

MedQA Forge ingests medical documents (PDFs, DOCX, PubMed articles), builds a domain-specific vector index using PubMedBERT embeddings and FAISS, then leverages local or cloud LLMs to produce grounded Q&A pairs. Every pair passes through a multi-stage quality pipeline — length, format, medical relevance, semantic deduplication — before landing in a review dashboard where human experts approve, reject, or refine the output. The result is a ready-to-export dataset for fine-tuning medical language models.

---

## Why This Exists

High-quality medical Q&A data is the bottleneck for training domain-specific LLMs. Manual curation is slow and expensive. Existing synthetic data tools either ignore source grounding (hallucination risk) or lack quality filtering (noise). MedQA Forge closes both gaps:

- **Retrieval-Augmented Generation** grounds every question in real clinical text, reducing hallucination.
- **Adaptive over-generation with multi-stage filtering** ensures only clinically coherent, non-duplicate pairs survive.
- **Human-in-the-loop review** with optional LLM-assisted scoring preserves expert oversight without bottlenecking throughput.

The system is designed for researchers preparing fine-tuning datasets, teams building clinical NLP tools, and anyone who needs structured medical Q&A data at scale.

---

## Key Features

| Category | Capabilities |
|----------|-------------|
| **Document Ingestion** | PDF (with OCR fallback), DOCX, TXT, CSV, JSON, PubMed search (abstracts + full-text via PMC) |
| **RAG Pipeline** | PubMedBERT embeddings → FAISS vector index → context-grounded prompt construction |
| **LLM Providers** | Ollama (local), OpenAI, Anthropic, Google Gemini, OpenRouter — unified abstraction |
| **Adaptive Generation** | Dynamic over-generation multiplier (1.3×–3.5×) adjusting in real time based on acceptance rate |
| **Quality Pipeline** | Length validation, format checks, medical keyword scoring, string dedup, semantic dedup (cosine < 0.92) |
| **LLM Review** | Optional second-pass LLM scoring for accuracy, relevance, and completeness |
| **GPU Acceleration** | Auto-detected CUDA for embeddings + FAISS; transparent CPU fallback; memory released between batches |
| **Real-Time Progress** | WebSocket streaming + polling fallback to a React dashboard |
| **Export** | JSON, CSV, Hugging Face JSONL, Alpaca format, OpenAI fine-tune format, Parquet |
| **Project Organisation** | Multi-project workspace with per-project sources, jobs, Q&A pairs, and exports |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│   Dashboard │ Project Detail │ Review │ Analytics │ Settings    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST + WebSocket
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Projects │  │ Sources   │  │ Export   │  │ LLM Providers │  │
│  │ Q&A CRUD │  │ Generation│  │ Review   │  │ Ollama Status │  │
│  └──────────┘  └─────┬─────┘  └──────────┘  └───────────────┘  │
└────────────────────────┼────────────────────────────────────────┘
                         │ Celery task dispatch
┌────────────────────────▼────────────────────────────────────────┐
│                    Celery Worker (GPU-enabled)                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              GenerationPipeline Orchestrator                │ │
│  │                                                            │ │
│  │  1. Document Processing ──► Text extraction + chunking     │ │
│  │  2. PubMed Fetch ────────► Abstract + full-text retrieval  │ │
│  │  3. RAG Index Build ─────► PubMedBERT → FAISS (GPU/CPU)   │ │
│  │  4. Prompt Construction ─► RAG retrieval or random sample  │ │
│  │  5. LLM Generation ─────► Adaptive batching + retry        │ │
│  │  6. Quality Validation ──► Multi-check + semantic dedup    │ │
│  │  7. Storage ─────────────► SQLite via SQLAlchemy            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                              │
    ┌────▼────┐                    ┌────▼────┐
    │  Redis  │                    │ SQLite  │
    │ Broker  │                    │   DB    │
    └─────────┘                    └─────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Zustand, TanStack Query |
| API Server | FastAPI 0.109, Pydantic v2, WebSocket (Redis pub/sub bridge) |
| Task Queue | Celery 5.3 + Redis 7 |
| Database | SQLite (WAL mode) via SQLAlchemy 2.0 |
| Embeddings | sentence-transformers (PubMedBERT 768-dim), FAISS-GPU/CPU |
| GPU Runtime | PyTorch CUDA 12.4, faiss-gpu-cu12 |
| Containerisation | Docker Compose — 4 services: backend, celery_worker, frontend, redis |

---

## Data Flow

A complete generation run follows this pipeline:

```
User configures and starts generation via the frontend
  │
  ▼
POST /api/v1/projects/{id}/generate
  │  Validates provider API key, creates GenerationJob record
  │  Dispatches Celery task asynchronously
  ▼
Celery Worker picks up task
  │
  ├─► Phase 1: Document Processing
  │   Uploaded PDFs/DOCX → text extraction (PyMuPDF + OCR fallback)
  │   → cleaning (artifact removal) → chunking (configurable size/overlap)
  │
  ├─► Phase 2: PubMed Retrieval (optional)
  │   ESearch → EFetch abstracts → PMC full-text → section-based chunking
  │
  ├─► Phase 3: RAG Index Construction
  │   PubMedBERT encodes all chunks → L2-normalised 768-dim vectors
  │   → FAISS IndexFlatIP (GPU-accelerated when available)
  │
  ├─► Phase 4: Prompt Construction
  │   For each mini-batch: retrieve top-K chunks via FAISS similarity search
  │   → format evidence without citation tags → build typed medical prompts
  │
  ├─► Phase 5: LLM Generation (adaptive loop)
  │   Send prompts via unified provider interface → parse Q&A pairs
  │   → validate each pair → adjust over-generation multiplier
  │   → early-stop if acceptance rate < 5%
  │   → release GPU memory between batches
  │
  ├─► Phase 6: Quality Validation
  │   Length check → format check → content quality → string dedup
  │   → semantic dedup (embedding cosine similarity) → composite scoring
  │
  └─► Phase 7: Storage & Completion
      Accepted pairs stored with full provenance metadata
      → Job marked complete → WebSocket notification to frontend
```

---

## RAG Pipeline Deep Dive

The Retrieval-Augmented Generation pipeline is the core differentiator — it grounds Q&A generation in actual medical literature rather than relying on the LLM's parametric knowledge.

### Chunking Strategy

Documents are split into overlapping chunks to preserve context at boundaries. Three strategies are available:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| **Word Count** (default) | Fixed word windows with configurable overlap | General medical text |
| **Sentence** | Sentence boundary detection with grouping | Short abstracts |
| **Paragraph** | Double-newline splitting | Structured clinical guidelines |

Default configuration: 500-word chunks with 50-word overlap.

### Embedding Model

The system uses a medical-domain embedding model for accurate retrieval:

| Priority | Model | Dimensions | Training Data |
|----------|-------|-----------|--------------|
| Primary | `pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb` | 768 | 30M+ PubMed abstracts |
| Fallback | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | 768 | Biomedical literature |
| Final | `all-MiniLM-L6-v2` | 384 | General web text |

PubMedBERT significantly outperforms general-purpose models on medical term similarity — it correctly distinguishes "myocardial infarction" from "myocarditis" and understands abbreviations like MI, CHF, and T2DM.

### Vector Similarity Search

- Index type: `faiss.IndexFlatIP` (inner product on L2-normalised vectors = cosine similarity)
- One index per project, cached in memory for fast repeated queries
- GPU acceleration via `faiss.index_cpu_to_gpu()` when CUDA is available
- Retrieval: top-K chunks (default 5) above a minimum cosine threshold (default 0.25)
- Source diversity enforcement: maximum 3 chunks from any single document per query

### Evidence Injection

Retrieved chunks are formatted with simple separators (`--- Medical Source ---`) rather than numbered citation labels. This prevents the LLM from parroting `[Evidence 1]` artifacts in its output — a common failure mode in RAG systems.

---

## LLM Provider Architecture

All five providers are accessed through a unified interface, making the system provider-agnostic at the service layer.

### Unified Call Chain

```
prompt_builder.prepare_prompts()
  → batch_generator.generate_batch()
    → llm_generation_client.generate_text()
      → llm_http.call_provider(provider, model, prompt, api_key, ...)
        → _call_ollama() | _call_openai() | _call_anthropic() | _call_gemini() | _call_openrouter()
          → http_client_manager.get_http_client(provider)  # pooled, loop-aware
```

### Provider Details

| Provider | Endpoint | Auth | Default Model | Timeout |
|----------|----------|------|--------------|---------|
| **Ollama** | `{OLLAMA_URL}/api/generate` | None (local) | llama3 | 180s |
| **OpenAI** | `api.openai.com/v1/chat/completions` | Bearer token | gpt-4o-mini | 120s |
| **Anthropic** | `api.anthropic.com/v1/messages` | x-api-key header | claude-3-5-haiku | 120s |
| **Google Gemini** | `generativelanguage.googleapis.com/v1beta` | URL param | gemini-2.0-flash | 120s |
| **OpenRouter** | `openrouter.ai/api/v1/chat/completions` | Bearer token | meta-llama/llama-3-8b | 120s |

### Adding a New Provider

1. Add an `async def _call_newprovider(prompt, api_key, model, temperature, top_p, max_tokens) -> str` function in [backend/app/services/llm_http.py](backend/app/services/llm_http.py).
2. Register it in the `_CLOUD_CALLERS` dictionary.
3. Add default models to `PROVIDER_MODELS`.
4. Add timeout configuration in [backend/app/services/http_client_manager.py](backend/app/services/http_client_manager.py) `_PROVIDER_TIMEOUTS`.
5. Add the provider name to the frontend's provider list in the Settings and Generation config components.

No changes needed to the generation pipeline, quality validation, or task infrastructure — the abstraction handles routing automatically.

### API Key Management

- Keys are encrypted at rest using Fernet symmetric encryption
- Stored in the database with per-provider default key selection
- Resolution priority: direct key → specific key ID → default key → any enabled key
- In-memory cache with 5-minute TTL to avoid repeated DB + decryption calls
- Dynamic model list fetching from provider APIs with intelligent caching (6–24h TTL)

### Retry Strategies

| Provider | Retry Schedule | 429 (Rate Limit) Schedule |
|----------|---------------|--------------------------|
| Ollama | 0s, 2s, 5s | N/A (local) |
| Cloud | 2s, 5s, 10s, 30s | 5s, 10s, 20s, 40s, 60s |

Proactive throttle delays are applied per-provider to avoid hitting rate limits preemptively (e.g. 4.5s between Gemini requests for free-tier compliance).

---

## GPU Strategy

### Auto-Detection

The standard Docker image ships with CUDA-enabled PyTorch and FAISS-GPU. No separate configuration or GPU-specific compose file is needed.

```
On startup:
  1. Backend checks torch.cuda.is_available()
  2. If GPU found → logs device name, VRAM, CUDA version
  3. Embeddings + FAISS operations use GPU transparently
  4. If no GPU → all operations fall back to CPU silently
```

### Memory Lifecycle

| Phase | GPU Activity | Cleanup |
|-------|-------------|---------|
| Index build | PubMedBERT encodes chunks on GPU; FAISS index transferred to GPU | — |
| Between mini-batches | `gc.collect()` + `torch.cuda.empty_cache()` | Lightweight |
| After task completion | Full model unload + FAISS GPU resources released + cache cleared | Full |
| On error (finally block) | Same full cleanup as completion | Full |

The `release_gpu_memory()` utility in [backend/app/utils/gpu_cleanup.py](backend/app/utils/gpu_cleanup.py) centralises all cleanup logic, ensuring no code path leaks GPU memory.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_DEVICE` | `auto` | `auto`, `cuda`, `cuda:0`, `cuda:1`, `cpu` |
| `EMBEDDING_BATCH_SIZE` | `64` | Reduce to 32 or 16 if OOM occurs |
| `GPU_MEMORY_FRACTION` | `0.8` | Maximum fraction of GPU VRAM to use |

---

## Quality & Filtering System

Every generated Q&A pair passes through a multi-stage validation pipeline before acceptance.

### Scoring Components

| Check | Weight | Criteria |
|-------|--------|---------|
| **Length** | 0.25 | Question ≥ 30 chars, Answer ≥ 50 chars; ideal ranges scored proportionally |
| **Format** | 0.20 | Question ends with `?` or starts with question word; answer does not end with `?` |
| **Content** | 0.30 | No self-referential phrases ("I am a language model"); no citation artifacts ("Figure 1", "et al."); no forbidden source phrases |
| **Duplicate** | 0.25 | SequenceMatcher ratio against all accepted questions < 0.92 |

Composite score: $S = 0.25 \cdot S_{\text{length}} + 0.20 \cdot S_{\text{format}} + 0.30 \cdot S_{\text{content}} + 0.25 \cdot S_{\text{duplicate}}$

### Extended Validation

Beyond the four core checks, the quality validator adds:

- **Completeness check** — detects truncated answers (ending in "..."), incomplete sentences, and "NOT FOUND" responses
- **Semantic deduplication** — encodes accepted questions with PubMedBERT and rejects new questions with cosine similarity ≥ 0.92 to any existing pair
- **All individual check scores logged at DEBUG level** for full transparency

### Threshold Logic

- Pairs scoring below `MIN_QUALITY_SCORE` (default 0.4) are rejected
- The threshold is intentionally conservative — pairs are cheap to generate but expensive to manually review
- Semantic dedup uses a higher threshold (0.92) to catch near-duplicates without over-filtering paraphrases

### Adaptive Over-Generation

The system dynamically adjusts how many prompts it generates to compensate for quality filtering:

```
Initial multiplier: 2.0× (generate 2× the target to account for rejections)

Every 10 prompts:
  acceptance_rate = accepted / generated
  new_multiplier = 1.0 / acceptance_rate
  multiplier = clamp(new_multiplier, min=1.3, max=3.5)

Early stop: if acceptance_rate < 5% after 20+ prompts → abort batch
```

This converges quickly — after 2–3 adaptation cycles, LLM calls match the actual yield rate, minimising wasted API calls.

---

## Installation

### Prerequisites

- **Docker** with Docker Compose v2
- **NVIDIA GPU** (optional) — requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU acceleration
- **Ollama** (or a cloud LLM API key) running on the host machine

### Quick Start

```bash
git clone <repo-url>
cd medqa-forge
cp .env.example .env    # review and edit as needed
docker compose build
docker compose up -d
```

Or use the helper script:

```powershell
# Windows
.\scripts\start.ps1

# Linux / macOS
./scripts/start.sh
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## Running the System

### With Docker Compose (recommended)

```bash
# Build and start all services (backend, worker, frontend, redis)
docker compose build
docker compose up -d

# Follow logs
docker compose logs -f

# Stop everything
docker compose down

# Full reset (removes all data, volumes, containers)
docker compose down -v --remove-orphans
```

### Without Docker (development)

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install faiss-gpu-cu12      # or faiss-cpu if no GPU
pip install -r requirements.txt

# Start Redis (required for Celery)
docker run -d -p 6379:6379 redis:7-alpine

# Configure for local development
# Set OLLAMA_URL=http://localhost:11434 and REDIS_URL=redis://localhost:6379/0 in .env

# Run the API server
uvicorn app.main:app --reload --port 8000

# Run the Celery worker (separate terminal)
celery -A app.celery_app worker --loglevel=info --concurrency=2
```

```bash
# Frontend
cd frontend
npm install
npm run dev    # http://localhost:5173
```

---

## Configuration Reference

### Environment Variables (`.env`)

Infrastructure and connectivity settings — the only variables in the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama server address |
| `DATABASE_URL` | `sqlite:///./data/medqa.db` | Database connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis broker URL |
| `UPLOAD_DIR` | `/app/data/uploads` | Upload file storage path |
| `OUTPUT_DIR` | `/app/data/outputs` | Export output path |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum upload file size |
| `CELERY_CONCURRENCY` | `2` | Celery worker process count |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins (comma-separated) |

### Docker Compose Overrides

Set in the `environment:` section of `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_DEVICE` | `auto` | GPU selection: `auto`, `cuda`, `cuda:0`, `cpu` |
| `EMBEDDING_BATCH_SIZE` | `64` | Reduce if GPU OOM occurs |
| `GPU_MEMORY_FRACTION` | `0.8` | Max GPU memory fraction |
| `FAISS_INDEX_DIR` | `/app/data/faiss` | FAISS index storage |

### Application Defaults

All generation, quality, and RAG settings are defined in `backend/app/config.py` via Pydantic Settings. They are **not** in `.env` by default but can be overridden by setting the corresponding environment variable:

| Setting | Default | Purpose |
|---------|---------|---------|
| `DEFAULT_CHUNK_SIZE` | `500` | Document chunk size (words) |
| `DEFAULT_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `DEFAULT_TARGET_PAIRS` | `1000` | Target Q&A pairs per job |
| `DEFAULT_TEMPERATURE` | `0.7` | LLM sampling temperature |
| `RAG_TOP_K` | `5` | Chunks retrieved per prompt |
| `RAG_MIN_SCORE` | `0.25` | Minimum cosine similarity for retrieval |
| `MIN_QUALITY_SCORE` | `0.4` | Composite quality floor |
| `SEMANTIC_DUP_THRESHOLD` | `0.92` | Cosine similarity dedup cutoff |
| `OVER_GEN_INITIAL_MULTIPLIER` | `2.0` | Starting over-generation factor |
| `MINI_BATCH_SIZE` | `15` | Prompts per mini-batch |

To override:
```bash
# In .env or docker-compose.yml environment block
MIN_QUALITY_SCORE=0.5
RAG_TOP_K=10
```

---

## API Reference

All endpoints are under `/api/v1/`. Interactive documentation at `http://localhost:8000/docs`.

| Group | Endpoints | Description |
|-------|----------|-------------|
| **Projects** | `POST /projects`, `GET /projects`, `GET /projects/{id}`, `DELETE /projects/{id}` | Project CRUD with cascading delete |
| **Sources** | `POST /projects/{id}/sources/upload`, `POST /projects/{id}/sources/pubmed` | Document upload and PubMed search |
| **Generation** | `POST /projects/{id}/generate`, `GET /jobs/{id}/progress`, `DELETE /jobs/{id}` | Job lifecycle management |
| **Q&A Pairs** | `GET /projects/{id}/qa-pairs`, `PATCH /qa-pairs/{id}`, `GET /qa-pairs/stats` | Pair CRUD + analytics |
| **Export** | `POST /projects/{id}/export`, `GET /export/download/{filename}` | Multi-format dataset export |
| **LLM Review** | `POST /projects/{id}/llm-review`, `GET /review/status/{id}` | LLM-assisted quality review |
| **Providers** | `GET /generation/providers`, `POST /llm-providers/test`, `GET /llm-providers/models/{provider}` | Provider management |
| **Ollama** | `GET /ollama/status`, `GET /ollama/models` | Local Ollama connectivity |
| **WebSocket** | `WS /ws/generation/{job_id}` | Real-time progress streaming |
| **Health** | `GET /health`, `GET /gpu-status` | System health + GPU diagnostics |

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── api/v1/                  # FastAPI route handlers
│   │   │   ├── router.py           # Aggregate router (9 sub-routers)
│   │   │   ├── generation.py       # Job dispatch + progress
│   │   │   ├── projects.py         # Project CRUD
│   │   │   ├── sources.py          # File upload
│   │   │   ├── qa_pairs.py         # Q&A pair management + analytics
│   │   │   ├── export.py           # Dataset export
│   │   │   ├── llm_review.py       # LLM review sessions
│   │   │   ├── llm_providers.py    # API key management
│   │   │   ├── ollama.py           # Ollama status
│   │   │   └── websocket.py        # Redis → WebSocket bridge
│   │   ├── models/                  # SQLAlchemy ORM (10 models)
│   │   ├── schemas/                 # Pydantic request/response schemas
│   │   ├── services/               # Core business logic
│   │   │   ├── qa_generator.py     # Generation pipeline orchestrator
│   │   │   ├── batch_generator.py  # Adaptive batching + early stopping
│   │   │   ├── rag_service.py      # FAISS indexing + retrieval
│   │   │   ├── llm_generation_client.py  # Provider-agnostic LLM client
│   │   │   ├── llm_http.py         # Unified HTTP callers (5 providers)
│   │   │   ├── llm_review_service.py    # LLM review + fact-check
│   │   │   ├── quality_checker.py  # Individual quality checks
│   │   │   ├── quality_validator.py     # Composite scoring + semantic dedup
│   │   │   ├── prompt_builder.py   # Evidence formatting + prompt construction
│   │   │   ├── document_processor.py    # PDF/DOCX/XML text extraction
│   │   │   ├── pubmed_service.py   # PubMed/PMC article fetching
│   │   │   ├── export_service.py   # Multi-format export generation
│   │   │   ├── progress_tracker.py # Redis pub/sub + DB progress
│   │   │   ├── http_client_manager.py   # Pooled httpx clients
│   │   │   ├── api_key_service.py  # Key resolution + caching
│   │   │   ├── encryption_service.py    # Fernet encryption
│   │   │   ├── rate_limit_handler.py    # Backoff + concurrency limits
│   │   │   ├── model_fetcher.py    # Dynamic model list APIs
│   │   │   └── ollama_service.py   # Connection test + Q&A parser
│   │   ├── tasks/
│   │   │   └── generation.py       # Celery task (async→sync bridge)
│   │   ├── utils/
│   │   │   ├── gpu_cleanup.py      # GPU memory release
│   │   │   ├── startup.py          # Orphaned job cleanup (shared)
│   │   │   ├── prompts.py          # Prompt templates
│   │   │   ├── chunking.py         # 3 chunking strategies
│   │   │   ├── json_utils.py       # LLM JSON extraction
│   │   │   ├── text_cleaning.py    # PDF artifact removal
│   │   │   └── helpers.py          # Filename utilities
│   │   ├── celery_app.py           # Celery config + worker startup hooks
│   │   ├── config.py               # Pydantic Settings (60+ variables)
│   │   ├── database.py             # Engine, sessions, WAL mode, migrations
│   │   └── main.py                 # FastAPI app factory + lifespan
│   ├── Dockerfile                   # Python 3.11 + CUDA PyTorch + FAISS-GPU
│   ├── Dockerfile.gpu               # NVIDIA CUDA base image (optional)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                   # 5 pages (Dashboard, Project, Review, Analytics, Settings)
│   │   ├── components/              # shadcn/ui components (22 primitives + layout)
│   │   ├── hooks/                   # TanStack Query hooks, job watcher, toast
│   │   ├── store/                   # Zustand (theme, sidebar, active jobs)
│   │   ├── lib/                     # API client, WebSocket manager, utilities
│   │   └── types/                   # TypeScript interfaces
│   ├── Dockerfile                   # Multi-stage Node 20 → Nginx Alpine
│   ├── nginx.conf                   # SPA routing + API/WS proxy + gzip
│   └── package.json
├── scripts/                          # Helper scripts (.ps1 + .sh)
│   ├── start                        # Build + launch
│   ├── stop                         # Graceful shutdown
│   ├── logs                         # Follow container logs
│   ├── reset                        # Full data wipe
│   └── shell                        # Container shell access
├── docker-compose.yml                # 4-service orchestration (GPU-ready)
├── docker-compose.gpu.yml            # Optional NVIDIA base image override
└── .env.example                      # Environment template
```

---

## Performance Considerations

### GPU Usage

- The PubMedBERT embedding model requires ~1.5 GB VRAM
- FAISS GPU index memory scales linearly with chunk count (~4 bytes × 768 dimensions × chunk count)
- Memory is released between mini-batches and fully unloaded after task completion
- For machines with limited VRAM, reduce `EMBEDDING_BATCH_SIZE` to 16–32

### Scaling Workers

- `CELERY_CONCURRENCY=1` is safest for SQLite (avoids write contention)
- `CELERY_CONCURRENCY=2` works reliably with WAL mode enabled
- For higher concurrency, migrate to PostgreSQL

### Batch Tuning

- `MINI_BATCH_SIZE=15` balances memory usage against LLM API overhead
- `OVER_GEN_INITIAL_MULTIPLIER=2.0` is optimal for typical medical text; increase to 3.0 for niche topics with lower acceptance
- `RAG_TOP_K=5` provides broad context; reduce to 3 for highly focused generation

---

## Production Readiness

### Logging

- Structured format: `timestamp | level | module | message`
- Third-party HTTP loggers suppressed at WARNING to reduce noise
- Quality check score breakdowns available at DEBUG level
- Generation statistics tracked and reported at INFO level

### Error Handling

- Global exception handler returns structured JSON errors
- Celery task errors classified by type (auth, rate-limit, timeout, connection) with user-friendly messages
- Orphaned jobs automatically detected and marked as failed on system restart
- Stale broker tasks purged on worker startup to prevent ghost executions

### Monitoring

- `/health` endpoint for load balancer health checks
- `/gpu-status` endpoint for GPU diagnostics (device name, VRAM usage, CUDA version)
- WebSocket + polling dual-path for resilient progress delivery
- Job completion watcher with adaptive polling frequency

### Security

- API keys encrypted at rest (Fernet symmetric encryption)
- CORS restricted to configured origins (not wildcard)
- Non-root container user for backend and worker
- No authentication layer included — intended for single-user or trusted-network deployment

---


## Troubleshooting

### Ghost generation task on startup

A stale task from a previous unclean shutdown replayed automatically. Fixed:
- Celery ACKs tasks immediately (`task_acks_late=False`)
- Worker purges broker queue on startup
- Both API server and worker mark orphaned jobs as failed

Full reset if needed: `docker compose down -v && docker compose up --build -d`

### Ollama not connecting

```bash
# Verify Ollama is running
ollama serve

# Check .env
OLLAMA_URL=http://host.docker.internal:11434

# Test from inside the container
.\scripts\shell.ps1 backend
curl http://host.docker.internal:11434/api/tags
```

### GPU not detected

```bash
# Verify NVIDIA Container Toolkit
nvidia-ctk --version

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check backend logs
docker compose logs backend | Select-String -Pattern "gpu|cuda" -CaseSensitive:$false
```

### Out of memory

- Reduce `EMBEDDING_BATCH_SIZE` in `docker-compose.yml` (try 32 or 16)
- Set `CELERY_CONCURRENCY=1` in `.env`
- GPU memory is released automatically between batches

### Database locked

SQLite write contention under concurrent workers:
- Set `CELERY_CONCURRENCY=1`
- For production workloads, migrate to PostgreSQL

---

## License

This project is provided under the MIT License.
