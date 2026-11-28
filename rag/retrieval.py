"""
Embedding generation and semantic retrieval.
Supports local BGE-small (`local`), local Ollama (`ollama`), and cloud Google Gemini (`gemini`)
via EMBEDDING_PROVIDER while preserving cosine similarity retrieval.
"""

import logging
import threading
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

import config
from rag.gemini_client import GeminiServiceError, gemini_embed_texts
from rag.local_embedder import LocalEmbeddingError, local_embed_texts

logger = logging.getLogger(__name__)

_thread_state = threading.local()


def _set_last_embed_error(error_code: str | None, user_message: str | None = None) -> None:
    _thread_state.error_code = error_code
    _thread_state.user_message = user_message


def clear_last_embed_error() -> None:
    """Reset the thread-local embedding error state."""
    _set_last_embed_error(None, None)


def get_last_embed_error() -> tuple[str | None, str | None]:
    """Return (error_code, user_message) from the most recent embed_query call in this thread."""
    return getattr(_thread_state, "error_code", None), getattr(_thread_state, "user_message", None)


def embed_query(text: str) -> list[float] | None:
    """
    Convert a text string into an embedding vector using the active EMBEDDING_PROVIDER.
    Returns a list of floats, or None on failure (check get_last_embed_error()).
    """
    _set_last_embed_error(None, None)

    if not text or not text.strip():
        logger.warning("embed_query called with empty text")
        _set_last_embed_error("empty_query", "Please enter a question.")
        return None

    provider = config.get_embedding_provider()
    if provider == "local":
        return _embed_query_local(text.strip())
    if provider == "gemini":
        return _embed_query_gemini(text.strip())
    return _embed_query_ollama(text.strip())


def _embed_query_local(text: str) -> list[float] | None:
    try:
        vectors = local_embed_texts([text], is_query=True)
        return vectors[0].tolist() if len(vectors) else None
    except LocalEmbeddingError as exc:
        _set_last_embed_error(exc.error_code, exc.user_message)
        return None
    except Exception as exc:
        logger.error("Unexpected local embedding error: %s", type(exc).__name__)
        _set_last_embed_error(
            "embedding_failed",
            "Could not generate query embedding locally. Please try again.",
        )
        return None


def _embed_query_gemini(text: str) -> list[float] | None:
    try:
        vectors = gemini_embed_texts([text], task_type="RETRIEVAL_QUERY")
        return vectors[0] if vectors else None
    except GeminiServiceError as exc:
        _set_last_embed_error(exc.error_code, exc.user_message)
        return None
    except Exception as exc:
        logger.error("Unexpected Gemini embedding error: %s", type(exc).__name__)
        _set_last_embed_error(
            "embedding_failed",
            "Could not generate query embedding with Gemini. Please try again.",
        )
        return None


def _embed_query_ollama(text: str) -> list[float] | None:
    try:
        r = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": [text]},
            timeout=config.OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        embeddings = r.json().get("embeddings")
        if not embeddings:
            logger.error("Ollama returned empty embeddings list")
            _set_last_embed_error("embedding_failed", "Ollama returned an empty embedding.")
            return None
        return embeddings[0]
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama at %s — is it running?", config.OLLAMA_BASE_URL)
        _set_last_embed_error(
            "embedding_failed",
            "I couldn't connect to Ollama right now. Please make sure Ollama is running and try again.",
        )
        return None
    except requests.exceptions.Timeout:
        logger.error("Ollama embedding request timed out after %ss", config.OLLAMA_TIMEOUT)
        _set_last_embed_error(
            "ai_timeout",
            "The embedding request timed out. Please try again.",
        )
        return None
    except Exception as exc:
        logger.error("Unexpected error during Ollama embedding: %s", exc)
        _set_last_embed_error(
            "embedding_failed",
            "An error occurred while generating the query embedding.",
        )
        return None


def find_top_k(
    df,
    query_embedding: list[float],
    top_k: int | None = None,
    embeddings_matrix: np.ndarray | None = None,
) -> tuple:
    """
    Given a preloaded embeddings DataFrame (and optional pre-stacked float32 matrix)
    and a query embedding, return (result_df, similarity_scores) for the top-k chunks.

    Returns (None, None) if the query embedding is invalid or dimensions mismatch.
    Applies RELEVANCE_THRESHOLD: if the best score is below the threshold,
    returns an empty DataFrame to signal "insufficient context".
    """
    if query_embedding is None:
        return None, None

    k = config.TOP_K if top_k is None else top_k
    threshold = config.get_relevance_threshold()

    try:
        all_embeddings = (
            embeddings_matrix
            if embeddings_matrix is not None
            else np.vstack(df["embedding"].values).astype(np.float32, copy=False)
        )
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        if all_embeddings.shape[1] != query_vec.shape[1]:
            logger.error(
                "Embedding dimension mismatch: index=%d vs query=%d (embedding_provider=%s)",
                all_embeddings.shape[1],
                query_vec.shape[1],
                config.get_embedding_provider(),
            )
            _set_last_embed_error(
                "embedding_dim_mismatch",
                "The loaded embedding index does not match the active embedding model dimensions.",
            )
            return None, None

        similarities = cosine_similarity(all_embeddings, query_vec).flatten()

        best_score = float(similarities.max())
        if best_score < threshold:
            logger.info(
                "Best similarity %.3f is below threshold %.3f — returning empty results",
                best_score,
                threshold,
            )
            return df.iloc[0:0], similarities

        top_indices = similarities.argsort()[::-1][:k]
        result_df = df.loc[top_indices].copy()
        result_df["similarity"] = similarities[top_indices]
        return result_df, similarities

    except Exception as exc:
        logger.error("Error during retrieval: %s", exc)
        return None, None


def format_sources(result_df, include_full_text: bool = False) -> list[dict]:
    """
    Convert retrieved DataFrame rows into a JSON-serialisable list of source dicts.
    """
    if result_df is None or result_df.empty:
        return []

    sources = []
    for _, row in result_df.iterrows():
        raw_text = str(row["text"]).strip()
        item = {
            "video_number": str(row["number"]),
            "title": str(row["title"]).strip(),
            "start": round(float(row["start"]), 1),
            "end": round(float(row["end"]), 1),
            "start_fmt": _fmt_time(row["start"]),
            "end_fmt": _fmt_time(row["end"]),
            "similarity": round(float(row.get("similarity", 0)), 3),
            "excerpt": raw_text[:220] + ("…" if len(raw_text) > 220 else ""),
        }
        if include_full_text:
            item["text"] = raw_text
        sources.append(item)
    return sources


def _fmt_time(seconds: float) -> str:
    """Convert seconds (float) to MM:SS string."""
    s = int(round(seconds))
    m, sec = divmod(s, 60)
    return f"{m:02d}:{sec:02d}"
