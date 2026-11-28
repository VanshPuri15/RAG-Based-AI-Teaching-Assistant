"""
Top-level RAG pipeline.
Loads precomputed embeddings once at startup (caching a single float32 matrix to avoid
per-request array duplication) and orchestrates retrieval + generation across configured
EMBEDDING_PROVIDER ('local' | 'ollama' | 'gemini') and LLM_PROVIDER ('gemini' | 'ollama').
"""

import importlib.util
import logging
import joblib
import numpy as np
import pandas as pd
import requests

import config
from rag.retrieval import (
    embed_query,
    find_top_k,
    format_sources,
    clear_last_embed_error,
    get_last_embed_error,
)
from rag.generation import (
    build_prompt,
    call_llm,
    clear_last_llm_error,
    get_last_llm_error,
)
from rag.gemini_client import is_gemini_configured

logger = logging.getLogger(__name__)

# Global in-memory store — loaded once at startup without duplicating arrays per request
_df: pd.DataFrame | None = None
_embedding_matrix: np.ndarray | None = None
_embedding_matrix_df_id: int | None = None


def load_embeddings() -> bool:
    """
    Load the embeddings DataFrame for the active EMBEDDING_PROVIDER into memory
    and cache a contiguous float32 matrix for zero-copy retrieval.
    Returns True on success, False on failure.
    """
    global _df, _embedding_matrix, _embedding_matrix_df_id
    path = config.get_embeddings_path()
    try:
        loaded = joblib.load(path)
        matrix = np.vstack(loaded["embedding"].values).astype(np.float32, copy=False)
        _df = loaded
        _embedding_matrix = matrix
        _embedding_matrix_df_id = id(_df)
        logger.info(
            "Loaded embeddings (embedding_provider=%s, llm_provider=%s) from %s: %d chunks (%d dims) across %d videos",
            config.get_embedding_provider(),
            config.get_llm_provider(),
            path,
            len(_df),
            matrix.shape[1],
            _df["number"].nunique(),
        )
        return True
    except FileNotFoundError:
        logger.error("Embeddings file not found at %s", path)
        _df = None
        _embedding_matrix = None
        _embedding_matrix_df_id = None
        return False
    except Exception as exc:
        logger.error("Failed to load embeddings from %s: %s", path, exc)
        _df = None
        _embedding_matrix = None
        _embedding_matrix_df_id = None
        return False


def get_df() -> pd.DataFrame | None:
    """Return the loaded DataFrame (or None if not yet loaded)."""
    return _df


def get_embedding_matrix() -> np.ndarray | None:
    """Return cached float32 embedding matrix only if it still corresponds to current _df."""
    if _df is None or _embedding_matrix is None:
        return None
    if id(_df) != _embedding_matrix_df_id or len(_df) != _embedding_matrix.shape[0]:
        return None
    return _embedding_matrix


def answer_question(query: str, mode: str = "default", history: list = None) -> dict:
    """
    Full RAG pipeline: embed → retrieve → generate → return structured result.
    """
    result = {"query": query, "mode": mode, "answer": "", "sources": [], "error": None}

    if not query or not query.strip():
        result["error"] = "empty_query"
        result["answer"] = "Please enter a question."
        return result

    if _df is None:
        result["error"] = "embeddings_not_loaded"
        result["answer"] = "The knowledge base is not ready yet. Please try again shortly."
        return result

    # Expand referential follow-up queries with the previous student question for retrieval
    retrieval_query = query
    if history:
        last_user_turns = [t["content"] for t in history if t.get("role") == "user" and t.get("content")]
        words = [w.strip("?.!,:;'\"") for w in query.lower().split()]
        referential = {"it", "this", "that", "they", "them", "those", "these", "its", "their"}
        followup_phrases = ("what about", "how about", "more about", "tell me more", "give an example", "step by step", "summarize")
        q_lower = query.lower().strip()
        is_referential = any(w in referential for w in words) or any(q_lower.startswith(p) for p in followup_phrases)
        if last_user_turns and is_referential:
            retrieval_query = f"{last_user_turns[-1]} — {query}"

    # Step 1: Embed the query using the same model family as the loaded index
    clear_last_embed_error()
    query_embedding = embed_query(retrieval_query)
    if query_embedding is None:
        err_code, err_msg = get_last_embed_error()
        result["error"] = err_code or "embedding_failed"
        result["answer"] = err_msg or (
            "I couldn't connect to the AI embedding service right now. Please try again shortly."
        )
        return result

    # Step 2: Retrieve top-k chunks using cosine similarity (reusing cached matrix)
    top_df, _ = find_top_k(_df, query_embedding, embeddings_matrix=get_embedding_matrix())
    if top_df is None:
        err_code, err_msg = get_last_embed_error()
        result["error"] = err_code or "retrieval_failed"
        result["answer"] = err_msg or "An error occurred during retrieval. Please try again."
        return result

    context_chunks = format_sources(top_df, include_full_text=True)
    sources = [{k: v for k, v in c.items() if k != "text"} for c in context_chunks]

    if not sources:
        result["error"] = "no_relevant_content"
        result["answer"] = (
            f"I could not find relevant coverage for that question in the indexed {config.COURSE_NAME} material. "
            "I specialize in the topics taught in this course (such as HTML tags, forms, semantic elements, "
            "CSS selectors, and the CSS box model)."
        )
        return result

    # Step 3: Build prompt (inject conversation history if present)
    prompt = build_prompt(query, context_chunks, mode=mode)

    if history:
        history_text = ""
        for turn in history[-4:]:
            role_label = "Student" if turn["role"] == "user" else "Assistant"
            history_text += f"{role_label}: {turn['content']}\n"
        if history_text:
            prompt = f"Previous conversation context:\n{history_text}\n---\n{prompt}"

    # Step 4: Generate response
    clear_last_llm_error()
    answer = call_llm(prompt)
    if answer is None:
        err_code, err_msg = get_last_llm_error()
        result["error"] = err_code or "llm_failed"
        result["answer"] = err_msg or (
            "I couldn't generate a response right now. Please try again shortly."
        )
        return result

    result["answer"] = answer
    result["sources"] = sources
    return result


def check_ollama_health(timeout: float = 1.5) -> dict:
    """
    Probe local Ollama at OLLAMA_BASE_URL/api/tags and report availability
    and whether the configured embedding and/or LLM models are present.
    """
    emb_prov = config.get_embedding_provider()
    llm_prov = config.get_llm_provider()
    embed_model, llm_model = config.get_active_model_names()
    status = {
        "provider": "ollama",
        "embedding_provider": emb_prov,
        "llm_provider": llm_prov,
        "ollama_connected": False,
        "ai_ready": False,
        "embed_model": embed_model,
        "llm_model": llm_model,
        "embed_model_ready": False,
        "llm_model_ready": False,
    }
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        status["ollama_connected"] = True
        if emb_prov == "ollama":
            status["embed_model_ready"] = any(
                m == config.EMBED_MODEL or m.startswith(f"{config.EMBED_MODEL}:") for m in models
            )
        elif emb_prov == "local":
            status["embed_model_ready"] = importlib.util.find_spec("fastembed") is not None
        else:
            status["embed_model_ready"] = is_gemini_configured()

        if llm_prov == "ollama":
            status["llm_model_ready"] = any(
                m == config.LLM_MODEL or m.startswith(f"{config.LLM_MODEL}:") for m in models
            )
        else:
            status["llm_model_ready"] = is_gemini_configured()

        status["ai_ready"] = status["embed_model_ready"] and status["llm_model_ready"]
    except Exception as exc:
        logger.debug("Ollama health probe failed: %s", exc)
    return status


def check_ai_health(timeout: float = 1.5) -> dict:
    """
    Return provider-aware AI readiness status without requiring Ollama unless
    either EMBEDDING_PROVIDER or LLM_PROVIDER is set to 'ollama'.
    """
    emb_prov = config.get_embedding_provider()
    llm_prov = config.get_llm_provider()
    if emb_prov == "ollama" or llm_prov == "ollama":
        return check_ollama_health(timeout=timeout)

    embed_model, llm_model = config.get_active_model_names()
    gemini_ready = is_gemini_configured()
    local_ready = importlib.util.find_spec("fastembed") is not None

    embed_ready = local_ready if emb_prov == "local" else gemini_ready
    llm_ready = gemini_ready
    provider_label = "hybrid" if emb_prov != llm_prov else llm_prov

    return {
        "provider": provider_label,
        "embedding_provider": emb_prov,
        "llm_provider": llm_prov,
        "ai_ready": embed_ready and llm_ready,
        "gemini_configured": gemini_ready,
        "ollama_connected": False,
        "embed_model": embed_model,
        "llm_model": llm_model,
        "embed_model_ready": embed_ready,
        "llm_model_ready": llm_ready,
    }


def get_all_videos() -> list[dict]:
    """Return a list of unique videos in the knowledge base (for display)."""
    if _df is None:
        return []
    unique = (
        _df[["number", "title"]]
        .drop_duplicates()
        .sort_values("number", key=lambda s: s.astype(int) if s.str.isnumeric().all() else s)
    )
    return [
        {"number": str(row["number"]), "title": str(row["title"]).strip()}
        for _, row in unique.iterrows()
    ]
