# Installation Guide

## Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Multi-container orchestration |
| Ollama | Latest | Local LLM inference |

## Step 1: Install Ollama

Download and install from [ollama.ai](https://ollama.ai).

**Verify installation:**
```bash
ollama --version
```

**Pull a model:**
```bash
ollama pull llama3
# Or other medical-suitable models:
ollama pull meditron
ollama pull medllama2
```

**Start Ollama:**
```bash
ollama serve
```

> Ollama must be running before starting the application.

## Step 2: Clone the Repository

```bash
git clone <repository-url>
cd "dataset generator"
```

## Step 3: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` if needed. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama server URL |
| `CELERY_CONCURRENCY` | `2` | Number of parallel workers |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file upload size |
| `PUBMED_EMAIL` | (empty) | Required for PubMed API |

### Ollama URL Configuration

- **Docker on Mac/Windows:** `http://host.docker.internal:11434` (default, works out of the box)
- **Docker on Linux:** `http://host.docker.internal:11434` (works via `extra_hosts` in docker-compose.yml)
- **Without Docker:** `http://localhost:11434`

## Step 4: Start the Application

**Linux / macOS:**
```bash
chmod +x scripts/*.sh
./scripts/start.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\start.ps1
```

First run will:
1. Build Docker images (may take 3-5 minutes)
2. Start Redis, backend, Celery worker, and frontend
3. Create the SQLite database automatically
4. Print access URLs

## Step 5: Verify

1. Open **http://localhost:3000** in your browser
2. Check that the Ollama status badge in the top bar shows "Connected"
3. Create your first project and upload a document

## Troubleshooting

### Ollama shows "Disconnected"

- Ensure Ollama is running: `ollama list`
- Check the URL in `.env` matches your setup
- On Linux, verify `host.docker.internal` resolves: `docker compose exec backend ping host.docker.internal`

### Build fails

- Ensure Docker has sufficient resources (4GB+ RAM recommended)
- Try: `docker compose build --no-cache`

### Port conflicts

If ports 3000 or 8000 are in use, modify `docker-compose.yml`:
```yaml
ports:
  - "3001:3000"  # Change host port
```

### View logs

```bash
./scripts/logs.sh           # All services
docker compose logs backend # Specific service
```
