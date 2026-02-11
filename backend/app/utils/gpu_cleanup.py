"""Centralized GPU memory cleanup utility.

Provides ``release_gpu_memory()`` for comprehensive GPU resource release:
  1. Unload the SentenceTransformer embedding model from VRAM
  2. Release FAISS GPU resources and clear the per-project index cache
  3. Run Python garbage collection to break reference cycles
  4. Call ``torch.cuda.empty_cache()`` to return memory to the OS

Call this from Celery task ``finally`` blocks, post-generation hooks,
or anywhere lingering GPU memory usage is observed.
"""
from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def release_gpu_memory(log_label: str = "", full_unload: bool = True) -> None:
    """Release GPU memory comprehensively.

    Parameters
    ----------
    log_label : optional context string for log messages.
    full_unload : if True (default), unload the embedding model and FAISS
        GPU resources completely.  Set to False for lightweight cleanup
        between mini-batches (only clears PyTorch cache).
    """
    prefix = f"[{log_label}] " if log_label else ""

    # 1. Unload embedding model + FAISS GPU resources (heavyweight)
    if full_unload:
        try:
            from app.services.rag_service import unload_gpu_resources
            unload_gpu_resources()
            logger.info("%sEmbedding model and FAISS GPU resources unloaded", prefix)
        except Exception as e:
            logger.debug("%sFailed to unload RAG resources (non-critical): %s", prefix, e)

    # 2. Python GC — break reference cycles so tensors can be freed
    collected = gc.collect()

    # 3. PyTorch CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            before_free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            torch.cuda.empty_cache()
            after_free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            freed = after_free - before_free
            if freed > 1.0:
                logger.info(
                    "%sGPU memory released: %.0f MB freed (%.0f MB now free)",
                    prefix, freed, after_free,
                )
            else:
                logger.debug(
                    "%sGPU cache cleared (%.0f MB free, %d GC objects collected)",
                    prefix, after_free, collected,
                )
        else:
            logger.debug("%sNo CUDA GPU available — skipped GPU cleanup", prefix)
    except ImportError:
        logger.debug("%sPyTorch not available — skipped GPU cleanup", prefix)
    except Exception as e:
        logger.debug("%sGPU cleanup error (non-critical): %s", prefix, e)
