"""
Central configuration — all tuneable values come from environment variables.
Supports independent selection of EMBEDDING_PROVIDER ('local', 'ollama', 'gemini')
and LLM_PROVIDER ('gemini', 'ollama'), while remaining backward-compatible with AI_PROVIDER.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── AI Provider Selection ──────────────────────────────────────────────────
# AI_PROVIDER acts as a convenient shorthand when both providers share a backend
# ('ollama' or 'gemini'). EMBEDDING_PROVIDER and LLM_PROVIDER override independently.
_RAW_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()
AI_PROVIDER = _RAW_PROVIDER if _RAW_PROVIDER in {"ollama", "gemini", "hybrid"} else "ollama"

VALID_EMBEDDING_PROVIDERS = {"local", "ollama", "gemini"}
VALID_LLM_PROVIDERS = {"gemini", "ollama"}


def get_ai_provider() -> str:
    """Return shorthand AI_PROVIDER ('ollama', 'gemini', or 'hybrid') from config/env."""
    val = os.getenv("AI_PROVIDER", AI_PROVIDER).strip().lower()
    return val if val in {"ollama", "gemini", "hybrid"} else AI_PROVIDER


def get_embedding_provider() -> str:
    """Return active embedding provider: 'local' (BGE-small), 'ollama', or 'gemini'."""
    explicit = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if explicit in VALID_EMBEDDING_PROVIDERS:
        return explicit
    ai_prov = get_ai_provider()
    if ai_prov == "hybrid":
        return "local"
    return ai_prov if ai_prov in VALID_EMBEDDING_PROVIDERS else "ollama"


def get_llm_provider() -> str:
    """Return active generation provider: 'gemini' or 'ollama'."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in VALID_LLM_PROVIDERS:
        return explicit
    ai_prov = get_ai_provider()
    if ai_prov == "hybrid":
        return "gemini"
    return ai_prov if ai_prov in VALID_LLM_PROVIDERS else "ollama"


def get_embeddings_path() -> str:
    """Resolve embeddings file path based on explicit override or active embedding provider."""
    explicit = os.getenv("EMBEDDINGS_PATH", "").strip()
    if explicit:
        return explicit
    emb_provider = get_embedding_provider()
    if emb_provider == "local":
        filename = "embeddings_local.joblib"
    elif emb_provider == "gemini":
        filename = "embeddings_gemini.joblib"
    else:
        filename = "embeddings.joblib"
    return str(BASE_DIR / filename)


EMBEDDINGS_PATH = get_embeddings_path()

# ── Local BGE Embeddings (FastEmbed / ONNX Runtime) ────────────────────────
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM", "384"))
LOCAL_EMBEDDING_THREADS = int(os.getenv("LOCAL_EMBEDDING_THREADS", "1"))
LOCAL_EMBEDDING_BATCH_SIZE = int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "32"))
LOCAL_EMBEDDING_CACHE_DIR = os.getenv("LOCAL_EMBEDDING_CACHE_DIR", "").strip()

# ── Ollama (Local Development Provider) ────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # seconds

# ── Google Gemini API (Cloud Generation / Optional Embeddings) ─────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
GEMINI_EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "60"))  # seconds
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
_DEFAULT_THRESHOLDS = {
    "local": 0.68,   # Calibrated empirically for BAAI/bge-small-en-v1.5 (covered: 0.716-0.884, off-topic: 0.504-0.664)
    "gemini": 0.50,
    "ollama": 0.55,
}
_ENV_THRESHOLD = os.getenv("RELEVANCE_THRESHOLD", "").strip()
RELEVANCE_THRESHOLD = (
    float(_ENV_THRESHOLD)
    if _ENV_THRESHOLD
    else _DEFAULT_THRESHOLDS[get_embedding_provider()]
)


def get_relevance_threshold() -> float:
    """Return active cosine similarity threshold calibrated for the active embedding provider."""
    raw = os.getenv("RELEVANCE_THRESHOLD", "").strip()
    if raw:
        return float(raw)
    return _DEFAULT_THRESHOLDS.get(get_embedding_provider(), 0.55)


def get_active_model_names() -> tuple[str, str]:
    """Return (embed_model_name, llm_model_name) for the active providers."""
    emb_prov = get_embedding_provider()
    llm_prov = get_llm_provider()

    if emb_prov == "local":
        embed_name = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL).strip()
    elif emb_prov == "gemini":
        embed_name = os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL).strip()
    else:
        embed_name = os.getenv("EMBED_MODEL", EMBED_MODEL).strip()

    if llm_prov == "gemini":
        llm_name = os.getenv("GEMINI_GENERATION_MODEL", GEMINI_GENERATION_MODEL).strip()
    else:
        llm_name = os.getenv("LLM_MODEL", LLM_MODEL).strip()

    return embed_name, llm_name


# ── Flask ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))

# ── Course metadata (for display) ──────────────────────────────────────────
COURSE_NAME = os.getenv("COURSE_NAME", "Sigma Web Development Course")
