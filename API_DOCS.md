# API Documentation

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

## Authentication

No authentication required (single-user local tool).

## Endpoints

### Health Check

```
GET /health
```
Returns `{ "status": "healthy" }`.

---

### Projects

#### List Projects
```
GET /api/v1/projects
```
Returns array of projects with enriched counts.

#### Create Project
```
POST /api/v1/projects
Content-Type: application/json

{
  "name": "My Project",
  "domain": "cardiology",
  "description": "optional"
}
```

#### Get Project
```
GET /api/v1/projects/{project_id}
```

#### Update Project
```
PATCH /api/v1/projects/{project_id}
Content-Type: application/json

{
  "name": "Updated Name",
  "status": "archived"
}
```

#### Delete Project
```
DELETE /api/v1/projects/{project_id}
```

---

### Sources

#### Upload Files
```
POST /api/v1/projects/{project_id}/sources/upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.xml, ...]
```

#### List Sources
```
GET /api/v1/projects/{project_id}/sources
```

#### Delete Source
```
DELETE /api/v1/projects/{project_id}/sources/{source_id}
```

---

### Generation

#### Start Generation
```
POST /api/v1/projects/{project_id}/generate
Content-Type: application/json

{
  "model": "llama3",
  "num_pairs_per_chunk": 3,
  "question_types": ["factual", "reasoning"],
  "difficulty": "mixed",
  "chunking_strategy": "word_count",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "include_pubmed": true,
  "pubmed_query": "heart failure",
  "pubmed_max_results": 10,
  "min_quality_score": 0.6,
  "temperature": 0.7,
  "max_concurrent": 3
}
```

Returns `{ "job_id": "...", "celery_task_id": "..." }`.

#### Get Job Progress
```
GET /api/v1/projects/{project_id}/jobs/{job_id}
```

#### List Jobs
```
GET /api/v1/projects/{project_id}/jobs
```

#### Cancel Job
```
POST /api/v1/projects/{project_id}/jobs/{job_id}/cancel
```

---

### Q&A Pairs

#### List Pairs (paginated)
```
GET /api/v1/projects/{project_id}/qa-pairs?page=1&size=20&validation_status=pending&search=heart
```

Query parameters:
- `page` (int, default 1)
- `size` (int, default 20)
- `validation_status` (pending|approved|rejected|needs_review)
- `source_type` (pdf|xml|pubmed|docx)
- `min_quality` (float, 0-1)
- `search` (text search)
- `sort_by` (created_at|quality_score|validation_status)
- `sort_dir` (asc|desc)

#### Get Stats
```
GET /api/v1/projects/{project_id}/qa-pairs/stats
```

#### Update Pair
```
PATCH /api/v1/projects/{project_id}/qa-pairs/{pair_id}
Content-Type: application/json

{
  "question": "updated question",
  "answer": "updated answer",
  "validation_status": "approved"
}
```

#### Batch Update
```
PATCH /api/v1/projects/{project_id}/qa-pairs/batch
Content-Type: application/json

{
  "ids": ["id1", "id2"],
  "validation_status": "approved"
}
```

#### Delete Pair
```
DELETE /api/v1/projects/{project_id}/qa-pairs/{pair_id}
```

---

### Export

```
POST /api/v1/projects/{project_id}/export
Content-Type: application/json

{
  "format": "csv",
  "validation_filter": "approved",
  "min_quality_score": 0.6,
  "split_dataset": true,
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1
}
```

Returns file download. Supported formats: `csv`, `json`, `jsonl`, `alpaca`, `openai`, `parquet`.

---

### Ollama

#### Connection Status
```
GET /api/v1/ollama/status
```

Returns `{ "connected": true, "url": "...", "models": [...] }`.

#### List Models
```
GET /api/v1/ollama/models
```

---

### WebSocket

```
WS /api/v1/ws/{project_id}
```

Receives real-time generation progress as JSON:
```json
{
  "job_id": "...",
  "status": "running",
  "progress_pct": 45,
  "current_step": "Generating Q&A pairs from PDF chunks",
  "pairs_generated": 12,
  "pairs_accepted": 10,
  "error_message": null
}
```
