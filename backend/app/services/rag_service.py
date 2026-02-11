"""RAG (Retrieval-Augmented Generation) service.

Implements the full RAG pipeline for medical Q&A generation:
  1. Embed text chunks using a domain-specific medical embedding model
  2. Store embeddings in a FAISS vector index (persisted to disk)
  3. Retrieve top-k relevant chunks for a given query
  4. Format retrieved context with citations for LLM prompting

Design decisions:
  - Uses PubMedBERT (768-dim) trained on PubMed abstracts + full-text — accurate
    for medical terminology, drug names, clinical context.  Falls back to
    all-MiniLM-L6-v2 (384-dim) if the medical model cannot be loaded.
  - FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine similarity).
  - One FAISS index per project — stored at data/faiss/{project_id}/index.faiss.
  - Chunk-ID mapping persisted alongside the index so retrieval returns DB IDs.
  - Thread-safe: index operations are protected by a lock.

Why a medical model matters:
  - General models confuse similar medical terms (diabetes vs diabetic neuropathy).
  - PubMedBERT understands clinical context, abbreviations (MI, CHF, T2DM), and
    multi-word medical phrases.
  - Retrieval quality directly determines Q&A generation quality.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy-loaded globals (heavy imports deferred to first use) ──────────

_model = None
_model_lock = threading.Lock()
_gpu_available: bool | None = None
_device: str | None = None

# ── Medical embedding models (best → fallback) ───────────────────────
# PubMedBERT: trained on 30M+ PubMed abstracts and full-text articles.
# Dimension: 768.  Superior medical term understanding vs general models.
_MEDICAL_MODELS = [
    ("pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb", 768),
    ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", 768),
]
_FALLBACK_MODEL = ("sentence-transformers/all-MiniLM-L6-v2", 384)

# These are set at model-load time (may change if medical model unavailable)
EMBEDDING_MODEL_NAME: str = _MEDICAL_MODELS[0][0]
EMBEDDING_DIM: int = _MEDICAL_MODELS[0][1]


def _detect_device() -> str:
    """Detect the best available device (GPU with CUDA fallback to CPU).

    Respects the ``GPU_DEVICE`` setting from config:
    - ``"auto"`` (default): detect CUDA → CPU
    - ``"cpu"``: force CPU
    - ``"cuda"`` / ``"cuda:0"`` / ``"cuda:1"``: use specific GPU
    """
    global _gpu_available, _device
    if _device is not None:
        return _device

    from app.config import get_settings
    settings = get_settings()
    requested = settings.GPU_DEVICE.lower().strip()

    if requested == "cpu":
        _gpu_available = False
        _device = "cpu"
        logger.info("GPU disabled by config (GPU_DEVICE=cpu)")
        return _device

    try:
        import torch
        if requested == "auto":
            if torch.cuda.is_available():
                _gpu_available = True
                _device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")

                # Apply memory fraction limit
                if settings.GPU_MEMORY_FRACTION < 1.0:
                    torch.cuda.set_per_process_memory_fraction(settings.GPU_MEMORY_FRACTION)
                    logger.info(f"GPU memory fraction set to {settings.GPU_MEMORY_FRACTION}")
            else:
                _gpu_available = False
                _device = "cpu"
                logger.info("No CUDA GPU detected — falling back to CPU")
        else:
            # Explicit device like "cuda:0"
            import torch
            if torch.cuda.is_available():
                _gpu_available = True
                _device = requested
                logger.info(f"Using explicit GPU device: {_device}")
            else:
                _gpu_available = False
                _device = "cpu"
                logger.warning(f"Requested {requested} but no CUDA available — falling back to CPU")
    except ImportError:
        _gpu_available = False
        _device = "cpu"
        logger.info("PyTorch not available — using CPU for embeddings")

    return _device


def _get_embedding_model():
    """Lazy-load the best available medical embedding model (thread-safe singleton).

    Tries PubMedBERT first, falls back to the lightweight general model if the
    medical model download fails (e.g. no internet on first run).
    """
    global _model, EMBEDDING_MODEL_NAME, EMBEDDING_DIM
    if _model is None:
        with _model_lock:
            if _model is None:
                device = _detect_device()
                from sentence_transformers import SentenceTransformer

                # Try medical models in preference order
                candidates = list(_MEDICAL_MODELS) + [_FALLBACK_MODEL]

                # Check if user overrode via config / env
                from app.config import get_settings
                settings = get_settings()
                user_model = settings.EMBEDDING_MODEL_NAME
                if user_model and user_model not in (
                    m[0] for m in _MEDICAL_MODELS
                ) and user_model != _FALLBACK_MODEL[0]:
                    # Put user's custom choice first
                    candidates = [(user_model, 768)] + candidates

                for model_name, dim in candidates:
                    try:
                        logger.info(
                            "Loading embedding model: %s (dim=%d) on %s ...",
                            model_name, dim, device,
                        )
                        _model = SentenceTransformer(model_name, device=device)
                        EMBEDDING_MODEL_NAME = model_name
                        EMBEDDING_DIM = dim
                        logger.info(
                            "Embedding model loaded: %s (dim=%d, device=%s)",
                            model_name, dim, device,
                        )
                        break
                    except Exception as exc:
                        logger.warning(
                            "Could not load %s: %s — trying next candidate",
                            model_name, exc,
                        )
                        _model = None

                if _model is None:
                    raise RuntimeError(
                        "Failed to load ANY embedding model. "
                        "Ensure you have internet access on first run."
                    )
    return _model


def is_gpu_available() -> bool:
    """Check whether GPU acceleration is active for embeddings."""
    _detect_device()
    return bool(_gpu_available)


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk returned by the retrieval step, with citation metadata."""
    chunk_id: str          # DB primary key of the Chunk record
    chunk_index: int       # Position within the source document
    source_id: str         # DB primary key of the Source record
    source_filename: str   # Human-readable source filename
    content: str           # The actual text of the chunk
    score: float           # Cosine similarity score (0-1)
    word_count: int        # Number of words in the chunk


@dataclass
class RetrievalResult:
    """Complete result of a retrieval operation."""
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    total_indexed: int = 0  # How many chunks are in the FAISS index

    @property
    def has_evidence(self) -> bool:
        """True if at least one chunk was retrieved above the minimum threshold."""
        return len(self.chunks) > 0

    def citation_ids(self) -> list[str]:
        """Return the chunk IDs used as evidence."""
        return [c.chunk_id for c in self.chunks]

    def citation_metadata(self) -> list[dict]:
        """Return structured citation metadata for storage in QAPair.metadata_json."""
        return [
            {
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "source_filename": c.source_filename,
                "chunk_index": c.chunk_index,
                "score": round(c.score, 4),
                "word_count": c.word_count,
                "content_preview": c.content[:200],
            }
            for c in self.chunks
        ]

    def format_context(self) -> str:
        """Format retrieved chunks into a context block for the LLM prompt.

        Uses descriptive source separators instead of citation tags to
        prevent citation artifacts (e.g. "[Evidence 1]") from leaking
        into generated Q&A pairs.
        """
        if not self.chunks:
            return ""
        parts = []
        for chunk in self.chunks:
            source_name = chunk.source_filename or "Medical Source"
            parts.append(
                f"--- Medical Source ({source_name}) ---\n"
                f"{chunk.content}"
            )
        return "\n\n".join(parts)


# ── Embedding ──────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """Embed a list of texts into normalised vectors using the sentence-transformer.

    Returns an (N, 384) float32 array with L2-normalised rows so that
    inner product == cosine similarity.

    Uses GPU-aware batch sizing to prevent OOM errors.
    """
    from app.config import get_settings
    settings = get_settings()

    if batch_size is None:
        batch_size = settings.EMBEDDING_BATCH_SIZE

    model = _get_embedding_model()

    try:
        # show_progress_bar=False to avoid polluting Celery worker logs
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalise so IP == cosine
            convert_to_numpy=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and is_gpu_available():
            # GPU OOM — clear cache, halve batch size, retry on CPU fallback
            logger.warning(f"GPU OOM during embedding (batch_size={batch_size}). "
                           f"Retrying with smaller batch on CPU fallback.")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Retry with half batch size
            smaller_batch = max(8, batch_size // 2)
            embeddings = model.encode(
                texts,
                batch_size=smaller_batch,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device="cpu",
            )
        else:
            raise

    # Free GPU cache after large batch to prevent memory buildup
    if is_gpu_available() and len(texts) > batch_size:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return embeddings.astype(np.float32)


def embed_single(text: str) -> np.ndarray:
    """Embed a single text string. Returns a (384,) float32 vector."""
    return embed_texts([text])[0]


# ── GPU helpers for FAISS ──────────────────────────────────────────────

_gpu_res = None  # Singleton FAISS GPU resource


def _get_gpu_resource():
    """Get or create the FAISS GPU resource (singleton)."""
    global _gpu_res
    if _gpu_res is None:
        try:
            import faiss
            _gpu_res = faiss.StandardGpuResources()
            logger.info("FAISS GPU resources initialised")
        except Exception as e:
            logger.warning(f"Could not create FAISS GPU resources: {e}")
    return _gpu_res


def _maybe_gpu_index(cpu_index):
    """Promote a CPU FAISS index to GPU if hardware is available.

    Falls back gracefully to the original CPU index on any error.
    """
    if not is_gpu_available():
        return cpu_index
    try:
        import faiss
        res = _get_gpu_resource()
        if res is not None:
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logger.debug("FAISS index promoted to GPU")
            return gpu_index
    except Exception as e:
        logger.warning(f"Failed to move FAISS index to GPU, using CPU: {e}")
    return cpu_index


def _ensure_cpu_index(index):
    """If the index lives on GPU, copy it back to CPU (needed for persistence)."""
    try:
        import faiss
        if hasattr(faiss, "index_gpu_to_cpu"):
            return faiss.index_gpu_to_cpu(index)
    except Exception:
        pass
    return index


# ── FAISS Index Management ─────────────────────────────────────────────

@dataclass
class FAISSProjectIndex:
    """Manages a per-project FAISS index with chunk-ID mapping.

    The index is stored on disk at:
        {base_dir}/{project_id}/index.faiss   — the binary FAISS index
        {base_dir}/{project_id}/chunks.json   — chunk_id ↔ index position mapping
    """
    project_id: str
    base_dir: Path
    _index = None
    _chunk_ids: list[str] = field(default_factory=list)
    _chunk_map: dict[str, dict] = field(default_factory=dict)  # chunk_id → metadata
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self._chunk_ids = []
        self._chunk_map = {}
        self._lock = threading.Lock()
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._load_if_exists()

    @property
    def _index_dir(self) -> Path:
        return self.base_dir / self.project_id

    @property
    def _index_path(self) -> Path:
        return self._index_dir / "index.faiss"

    @property
    def _meta_path(self) -> Path:
        return self._index_dir / "chunks.json"

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def _load_if_exists(self):
        """Load a previously persisted FAISS index from disk (auto-promoted to GPU if available)."""
        import faiss
        if self._index_path.exists() and self._meta_path.exists():
            try:
                cpu_index = faiss.read_index(str(self._index_path))
                self._index = _maybe_gpu_index(cpu_index)
                with open(self._meta_path, "r") as f:
                    data = json.load(f)
                self._chunk_ids = data.get("chunk_ids", [])
                self._chunk_map = data.get("chunk_map", {})
                device = "GPU" if is_gpu_available() else "CPU"
                logger.info(
                    f"Loaded FAISS index for project {self.project_id}: "
                    f"{self._index.ntotal} vectors ({device})"
                )
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Will rebuild.")
                self._index = None
                self._chunk_ids = []
                self._chunk_map = {}

    def _save(self):
        """Persist the FAISS index and metadata to disk (GPU index downcast to CPU for I/O)."""
        import faiss
        if self._index is not None:
            # FAISS can only write CPU indices — downcast if on GPU
            cpu_index = _ensure_cpu_index(self._index)
            faiss.write_index(cpu_index, str(self._index_path))
            with open(self._meta_path, "w") as f:
                json.dump({
                    "chunk_ids": self._chunk_ids,
                    "chunk_map": self._chunk_map,
                }, f)

    def build_index(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        chunk_metadata: list[dict],
        batch_size: int = 64,
    ):
        """Build (or rebuild) the FAISS index from scratch.

        Skips the expensive rebuild if the cached index already contains
        exactly the same set of chunk IDs (same documents, same order).

        Parameters
        ----------
        chunk_ids : list of Chunk.id values (DB primary keys)
        chunk_texts : list of chunk content strings (same order)
        chunk_metadata : list of dicts with keys: source_id, source_filename,
                         chunk_index, word_count
        """
        import faiss

        if not chunk_ids:
            logger.warning(f"No chunks to index for project {self.project_id}")
            return

        with self._lock:
            # ── Cache check: skip rebuild if index matches current chunks ──
            if (
                self._index is not None
                and self._index.ntotal == len(chunk_ids)
                and self._chunk_ids == list(chunk_ids)
            ):
                logger.info(
                    f"FAISS index cache HIT for project {self.project_id}: "
                    f"{self._index.ntotal} vectors unchanged — skipping rebuild"
                )
                return

            logger.info(f"Building FAISS index for project {self.project_id}: {len(chunk_ids)} chunks")

            # Embed all chunks
            embeddings = embed_texts(chunk_texts, batch_size=batch_size)

            # Create a flat inner-product index (cosine similarity on L2-normed vectors)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            index.add(embeddings)

            # Promote to GPU if available for faster search
            self._index = _maybe_gpu_index(index)
            self._chunk_ids = list(chunk_ids)
            self._chunk_map = {
                cid: meta for cid, meta in zip(chunk_ids, chunk_metadata)
            }

            self._save()
            logger.info(
                f"FAISS index built and saved: {index.ntotal} vectors, "
                f"dim={EMBEDDING_DIM}"
            )

    def add_chunks(
        self,
        chunk_ids: list[str],
        chunk_texts: list[str],
        chunk_metadata: list[dict],
    ):
        """Incrementally add new chunks to an existing index."""
        import faiss

        with self._lock:
            if self._index is None:
                # No existing index — build from scratch
                self.build_index(chunk_ids, chunk_texts, chunk_metadata)
                return

            embeddings = embed_texts(chunk_texts)
            self._index.add(embeddings)

            for cid, meta in zip(chunk_ids, chunk_metadata):
                self._chunk_ids.append(cid)
                self._chunk_map[cid] = meta

            self._save()
            logger.info(f"Added {len(chunk_ids)} chunks to FAISS index (total: {self._index.ntotal})")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.25,
        chunk_texts: dict[str, str] | None = None,
        diverse: bool = True,
        max_per_source: int = 3,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a query with source diversity.

        Parameters
        ----------
        query : the search query (e.g. a medical topic or term)
        top_k : maximum number of chunks to return
        min_score : minimum cosine similarity threshold (0-1)
        chunk_texts : optional dict of chunk_id → full text content.
                      If not provided, content_preview from metadata is used.
        diverse : if True, enforce source diversity (max_per_source per document)
        max_per_source : maximum chunks from a single source document when diverse=True

        Returns
        -------
        List of RetrievedChunk, sorted by descending score with source diversity.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        with self._lock:
            query_vec = embed_single(query).reshape(1, -1)
            # Fetch a wide candidate pool for diversity selection
            candidate_k = min(top_k * 10, self._index.ntotal) if diverse else min(top_k * 2, self._index.ntotal)
            scores, indices = self._index.search(query_vec, candidate_k)

        # Build candidate list
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunk_ids):
                continue
            if score < min_score:
                continue

            chunk_id = self._chunk_ids[idx]
            meta = self._chunk_map.get(chunk_id, {})

            content = ""
            if chunk_texts and chunk_id in chunk_texts:
                content = chunk_texts[chunk_id]
            else:
                content = meta.get("content_preview", "")

            candidates.append(RetrievedChunk(
                chunk_id=chunk_id,
                chunk_index=meta.get("chunk_index", 0),
                source_id=meta.get("source_id", ""),
                source_filename=meta.get("source_filename", "unknown"),
                content=content,
                score=float(score),
                word_count=meta.get("word_count", 0),
            ))

        if not diverse or not candidates:
            return candidates[:top_k]

        # ── Source-diverse selection ──
        # Group candidates by source, pick top max_per_source per source,
        # then interleave to get a diverse final set.
        from collections import defaultdict
        source_groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for c in candidates:
            source_groups[c.source_filename].append(c)

        # Round-robin across sources, taking the best remaining from each
        results: list[RetrievedChunk] = []
        source_counts: dict[str, int] = defaultdict(int)

        # Sort sources by their best chunk score (descending) for priority ordering
        sorted_sources = sorted(
            source_groups.keys(),
            key=lambda s: source_groups[s][0].score if source_groups[s] else 0,
            reverse=True,
        )

        round_num = 0
        while len(results) < top_k:
            added_this_round = False
            for src in sorted_sources:
                if len(results) >= top_k:
                    break
                chunks = source_groups[src]
                if source_counts[src] < max_per_source and source_counts[src] < len(chunks):
                    results.append(chunks[source_counts[src]])
                    source_counts[src] += 1
                    added_this_round = True
            if not added_this_round:
                break
            round_num += 1

        # Sort final results by score (descending)
        results.sort(key=lambda c: c.score, reverse=True)

        if len(source_groups) > 1:
            logger.debug(
                "Diverse retrieval: %d results from %d sources (max %d/source) out of %d candidates",
                len(results), len(set(r.source_filename for r in results)),
                max_per_source, len(candidates),
            )

        return results

    def clear(self):
        """Remove the index from memory and disk."""
        import shutil
        with self._lock:
            self._index = None
            self._chunk_ids = []
            self._chunk_map = {}
            if self._index_dir.exists():
                shutil.rmtree(self._index_dir, ignore_errors=True)


# ── Project Index Cache ────────────────────────────────────────────────

_index_cache: dict[str, FAISSProjectIndex] = {}
_cache_lock = threading.Lock()


def get_project_index(project_id: str) -> FAISSProjectIndex:
    """Get or create the FAISS index for a project (cached in memory)."""
    from app.config import get_settings
    settings = get_settings()
    base_dir = Path(settings.OUTPUT_DIR).parent / "faiss"

    with _cache_lock:
        if project_id not in _index_cache:
            _index_cache[project_id] = FAISSProjectIndex(
                project_id=project_id,
                base_dir=base_dir,
            )
        return _index_cache[project_id]


def clear_project_index(project_id: str):
    """Remove and clear the FAISS index for a project."""
    with _cache_lock:
        if project_id in _index_cache:
            _index_cache[project_id].clear()
            del _index_cache[project_id]


def unload_gpu_resources() -> None:
    """Fully unload the embedding model and FAISS GPU resources from VRAM.

    Called by ``gpu_cleanup.release_gpu_memory()`` after generation tasks
    to ensure the Celery worker doesn't hold GPU memory between jobs.

    Releases:
      1. SentenceTransformer model (PubMedBERT/MiniLM)
      2. FAISS StandardGpuResources singleton
      3. All cached per-project FAISS GPU indices
    """
    global _model, _gpu_res, _gpu_available, _device

    # 1. Unload embedding model
    if _model is not None:
        try:
            del _model
        except Exception:
            pass
        _model = None
        logger.info("Embedding model unloaded from memory")

    # 2. Release FAISS GPU resources
    if _gpu_res is not None:
        try:
            del _gpu_res
        except Exception:
            pass
        _gpu_res = None
        logger.info("FAISS GPU resources released")

    # 3. Clear all cached project indices (they hold GPU-promoted indices)
    with _cache_lock:
        for pid, idx in list(_index_cache.items()):
            try:
                idx._index = None
                idx._chunk_ids = []
                idx._chunk_map = {}
            except Exception:
                pass
        _index_cache.clear()
        logger.info("FAISS index cache cleared")

    # Reset device detection so next load re-detects
    _gpu_available = None
    _device = None


# ── High-level retrieval function ──────────────────────────────────────

def retrieve_for_topic(
    project_id: str,
    topic: str,
    top_k: int = 5,
    min_score: float = 0.25,
    db=None,
) -> RetrievalResult:
    """Retrieve the most relevant chunks for a given medical topic.

    This is the main entry point for the RAG retrieval step.
    Called before each Q&A generation to select evidence chunks.

    Parameters
    ----------
    project_id : the project whose chunks to search
    topic : the medical topic or query (e.g. "heart failure treatment")
    top_k : max chunks to retrieve
    min_score : cosine similarity cutoff
    db : optional SQLAlchemy session to load full chunk texts

    Returns
    -------
    RetrievalResult with retrieved chunks, citations, and formatted context.
    """
    index = get_project_index(project_id)

    # Optionally load full chunk texts from DB
    chunk_texts = None
    if db is not None:
        from app.models import Chunk
        chunks = db.query(Chunk).filter(Chunk.project_id == project_id).all()
        chunk_texts = {c.id: c.content for c in chunks}

    retrieved = index.search(
        query=topic,
        top_k=top_k,
        min_score=min_score,
        chunk_texts=chunk_texts,
    )

    return RetrievalResult(
        query=topic,
        chunks=retrieved,
        total_indexed=index.size,
    )
