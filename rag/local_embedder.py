"""
Lightweight local embedding inference using FastEmbed (ONNX Runtime) for
`BAAI/bge-small-en-v1.5` (384 dimensions).

Designed to fit comfortably inside Render Free Tier's 512 MB RAM budget without
PyTorch or Sentence-Transformers overhead, while enforcing consistent text
preprocessing and L2 unit-norm normalization across documents and queries.
"""

import logging
import os
import threading
from typing import Iterable

import numpy as np

import config

logger = logging.getLogger(__name__)

# Official BGE v1.5 retrieval query prefix for asymmetric passage search
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model_lock = threading.Lock()
_model_cache: dict[tuple[str, int, str], object] = {}


class LocalEmbeddingError(Exception):
    """Raised when local ONNX embedding initialization or inference fails."""

    def __init__(self, error_code: str, user_message: str):
        super().__init__(user_message)
        self.error_code = error_code
        self.user_message = user_message


def l2_normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D float32 array row-wise."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def preprocess_text(text: str, is_query: bool = False) -> str:
    """
    Apply consistent whitespace normalization across documents and queries,
    adding the standard BGE retrieval instruction prefix for queries.
    """
    cleaned = " ".join(str(text).strip().split())
    if is_query and cleaned:
        return f"{BGE_QUERY_PREFIX}{cleaned}"
    return cleaned


def get_local_embedder():
    """
    Lazily load and cache the singleton FastEmbed TextEmbedding model.
    Uses threads=1 by default to minimize ONNX Runtime memory arena growth.
    """
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", config.LOCAL_EMBEDDING_MODEL).strip()
    threads = max(1, int(os.getenv("LOCAL_EMBEDDING_THREADS", config.LOCAL_EMBEDDING_THREADS)))
    cache_dir = os.getenv("LOCAL_EMBEDDING_CACHE_DIR", config.LOCAL_EMBEDDING_CACHE_DIR).strip()
    key = (model_name, threads, cache_dir)

    if key in _model_cache:
        return _model_cache[key]

    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]
        try:
            from fastembed import TextEmbedding

            kwargs = {"model_name": model_name, "threads": threads}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            logger.info("Loading local ONNX embedding model '%s' (threads=%d)...", model_name, threads)
            model = TextEmbedding(**kwargs)
            _model_cache[key] = model
            return model
        except ImportError as exc:
            raise LocalEmbeddingError(
                "local_embedder_missing",
                "Local embedding runtime (fastembed) is not installed on the server.",
            ) from exc
        except Exception as exc:
            logger.error("Failed to load local embedding model '%s': %s", model_name, exc)
            raise LocalEmbeddingError(
                "local_embedder_failed",
                "Could not initialize the local embedding model.",
            ) from exc


def local_embed_texts(
    texts: Iterable[str],
    is_query: bool = False,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Embed texts using `BAAI/bge-small-en-v1.5`, returning a float32 L2-normalized
    numpy array of shape (N, 384).
    """
    items = [preprocess_text(t, is_query=is_query) for t in texts]
    if not items:
        expected_dim = int(os.getenv("LOCAL_EMBEDDING_DIM", config.LOCAL_EMBEDDING_DIM))
        return np.empty((0, expected_dim), dtype=np.float32)

    bs = batch_size or max(1, int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", config.LOCAL_EMBEDDING_BATCH_SIZE)))
    model = get_local_embedder()

    try:
        if is_query:
            raw = list(model.query_embed(items, batch_size=bs))
        else:
            raw = list(model.passage_embed(items, batch_size=bs))
    except LocalEmbeddingError:
        raise
    except Exception as exc:
        logger.error("Local embedding inference failed: %s", exc)
        raise LocalEmbeddingError(
            "embedding_failed",
            "Failed to generate local text embeddings.",
        ) from exc

    matrix = l2_normalize_matrix(np.vstack(raw))
    expected_dim = int(os.getenv("LOCAL_EMBEDDING_DIM", config.LOCAL_EMBEDDING_DIM))
    if matrix.shape[1] != expected_dim:
        raise LocalEmbeddingError(
            "embedding_dim_mismatch",
            f"Local embedding model returned dimension {matrix.shape[1]}, expected {expected_dim}.",
        )
    return matrix
