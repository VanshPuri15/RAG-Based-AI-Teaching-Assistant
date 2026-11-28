"""
Google Gemini API client wrapper using the official `google-genai` SDK.
Handles lazy client initialization, exponential-backoff retries for rate limits
and transient failures, L2 vector normalization, and safe error classification.
Never logs or exposes GEMINI_API_KEY.
"""

import logging
import os
import random
import time
from typing import Callable, TypeVar

import numpy as np
from google import genai
from google.genai import errors, types

import config

logger = logging.getLogger(__name__)

T = TypeVar("T")

_client_cache: dict[tuple[str, int], genai.Client] = {}


class GeminiServiceError(Exception):
    """Sanitized exception carrying a stable internal error_code and user message."""

    def __init__(
        self,
        error_code: str,
        user_message: str,
        retry_after: float | None = None,
        is_daily_quota: bool = False,
    ):
        super().__init__(user_message)
        self.error_code = error_code
        self.user_message = user_message
        self.retry_after = retry_after
        self.is_daily_quota = is_daily_quota


def _get_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY", "") or config.GEMINI_API_KEY).strip()


def is_gemini_configured() -> bool:
    """Return True if a non-empty GEMINI_API_KEY is available."""
    return bool(_get_api_key())


def get_gemini_client() -> genai.Client:
    """Return a cached Gemini API client configured with HTTP timeout."""
    api_key = _get_api_key()
    if not api_key:
        raise GeminiServiceError(
            "gemini_not_configured",
            "Gemini API key is not configured on the server. Set GEMINI_API_KEY to enable cloud AI.",
        )

    timeout_sec = max(5, int(os.getenv("GEMINI_TIMEOUT", config.GEMINI_TIMEOUT)))
    cache_key = (api_key, timeout_sec)
    if cache_key not in _client_cache:
        _client_cache[cache_key] = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_sec * 1000),
        )
    return _client_cache[cache_key]


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize an embedding vector (required for truncated Gemini dimensions)."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


def _classify_api_error(exc: errors.APIError) -> tuple[str, str, bool]:
    """
    Map google.genai.errors.APIError into (error_code, user_message, is_retryable).
    Never includes API keys or raw secrets in messages.
    """
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()

    if code == 429 or "RESOURCE_EXHAUSTED" in status:
        return (
            "rate_limited",
            "The AI service is temporarily rate-limited. Please wait a few seconds and try again.",
            True,
        )
    if code in (401, 403) or "PERMISSION_DENIED" in status or "UNAUTHENTICATED" in status:
        return (
            "gemini_auth_error",
            "Gemini API authentication failed. Please check the server's GEMINI_API_KEY configuration.",
            False,
        )
    if code == 404 or "NOT_FOUND" in status:
        return (
            "gemini_model_error",
            "Configured Gemini model was not found. Please check GEMINI_GENERATION_MODEL / GEMINI_EMBEDDING_MODEL.",
            False,
        )
    if code == 400 or "INVALID_ARGUMENT" in status:
        return (
            "gemini_bad_request",
            "The AI service could not process this request format.",
            False,
        )
    if code in (500, 502, 503, 504) or "UNAVAILABLE" in status or "DEADLINE_EXCEEDED" in status:
        return (
            "gemini_unavailable",
            "The Gemini AI service is temporarily unavailable. Please try again shortly.",
            True,
        )
    return (
        "gemini_unavailable",
        "An error occurred while contacting the Gemini AI service. Please try again.",
        False,
    )


def _extract_quota_metadata(exc: errors.APIError) -> tuple[float | None, bool]:
    """Extract RetryInfo delay (seconds) and whether quota is daily from APIError.details."""
    details = getattr(exc, "details", None)
    retry_after: float | None = None
    is_daily = False

    if isinstance(details, dict):
        err_block = details.get("error", {})
        detail_list = err_block.get("details", []) if isinstance(err_block, dict) else []
        for item in detail_list:
            if not isinstance(item, dict):
                continue
            if "retryDelay" in item:
                raw_delay = str(item["retryDelay"]).rstrip("s")
                try:
                    retry_after = float(raw_delay)
                except ValueError:
                    pass
            for v in item.get("violations", []) if isinstance(item.get("violations"), list) else []:
                if isinstance(v, dict):
                    qid = str(v.get("quotaId", "")) + " " + str(v.get("quotaMetric", ""))
                    if "PerDay" in qid:
                        is_daily = True
    return retry_after, is_daily


def _execute_with_retry(fn: Callable[[], T], operation_name: str) -> T:
    """
    Execute a Gemini SDK operation with exponential backoff + jitter for 429 and transient 5xx.
    """
    max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", config.GEMINI_MAX_RETRIES)))
    attempt = 0

    while True:
        try:
            return fn()
        except GeminiServiceError:
            raise
        except errors.APIError as exc:
            error_code, user_msg, retryable = _classify_api_error(exc)
            retry_after, is_daily = _extract_quota_metadata(exc)
            logger.warning(
                "Gemini APIError during %s (code=%s, status=%s, attempt=%d/%d, retry_after=%s, daily=%s)",
                operation_name,
                getattr(exc, "code", "?"),
                getattr(exc, "status", "?"),
                attempt + 1,
                max_retries + 1,
                retry_after,
                is_daily,
            )
            if is_daily:
                raise GeminiServiceError(
                    error_code,
                    user_msg,
                    retry_after=retry_after,
                    is_daily_quota=True,
                ) from exc
            if retryable and attempt < max_retries:
                delay = (2 ** attempt) + random.uniform(0.1, 0.6)
                time.sleep(delay)
                attempt += 1
                continue
            raise GeminiServiceError(
                error_code,
                user_msg,
                retry_after=retry_after,
                is_daily_quota=is_daily,
            ) from exc
        except TimeoutError as exc:
            logger.warning("Gemini timeout during %s (attempt=%d/%d)", operation_name, attempt + 1, max_retries + 1)
            if attempt < max_retries:
                delay = (2 ** attempt) + random.uniform(0.1, 0.5)
                time.sleep(delay)
                attempt += 1
                continue
            raise GeminiServiceError(
                "ai_timeout",
                "The AI service timed out while responding. Please try again.",
            ) from exc
        except Exception as exc:
            # Handle httpx/network timeouts cleanly without leaking internals
            exc_name = type(exc).__name__.lower()
            is_timeout = "timeout" in exc_name
            is_network = any(k in exc_name for k in ("connect", "network", "transport", "protocol"))
            logger.error("Unexpected Gemini error during %s: %s", operation_name, type(exc).__name__)
            if (is_timeout or is_network) and attempt < max_retries:
                delay = (2 ** attempt) + random.uniform(0.1, 0.5)
                time.sleep(delay)
                attempt += 1
                continue
            if is_timeout:
                raise GeminiServiceError(
                    "ai_timeout",
                    "The AI service timed out while responding. Please try again.",
                ) from exc
            raise GeminiServiceError(
                "gemini_unavailable",
                "Could not reach the Gemini AI service right now. Please try again shortly.",
            ) from exc


def gemini_embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_QUERY",
    titles: list[str] | None = None,
) -> list[list[float]]:
    """
    Generate normalized embeddings for a list of strings via Gemini API.
    """
    if not texts:
        return []

    model = os.getenv("GEMINI_EMBEDDING_MODEL", config.GEMINI_EMBEDDING_MODEL).strip()
    dim = int(os.getenv("GEMINI_EMBEDDING_DIM", config.GEMINI_EMBEDDING_DIM))
    client = get_gemini_client()

    cfg_kwargs: dict = {
        "task_type": task_type,
        "output_dimensionality": dim,
    }
    # title applies only when embedding a single document with RETRIEVAL_DOCUMENT
    if titles and len(texts) == 1 and task_type == "RETRIEVAL_DOCUMENT" and titles[0]:
        cfg_kwargs["title"] = titles[0]

    embed_config = types.EmbedContentConfig(**cfg_kwargs)

    def _call():
        return client.models.embed_content(
            model=model,
            contents=texts,
            config=embed_config,
        )

    response = _execute_with_retry(_call, "embed_content")
    raw_embeddings = getattr(response, "embeddings", None) or []
    if len(raw_embeddings) != len(texts):
        raise GeminiServiceError(
            "embedding_failed",
            "The embedding service returned an unexpected number of vectors.",
        )

    return [l2_normalize(list(item.values)) for item in raw_embeddings]


def gemini_generate_text(
    prompt: str,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    response_json_schema: dict | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """
    Generate text (or structured JSON) via Gemini API using GenerateContentConfig.
    """
    model = os.getenv("GEMINI_GENERATION_MODEL", config.GEMINI_GENERATION_MODEL).strip()
    client = get_gemini_client()

    cfg_kwargs: dict = {}
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        cfg_kwargs["response_mime_type"] = response_mime_type
    if response_json_schema:
        cfg_kwargs["response_json_schema"] = response_json_schema
    if temperature is not None:
        cfg_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        cfg_kwargs["max_output_tokens"] = max_output_tokens

    gen_config = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

    def _call():
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=gen_config,
        )

    response = _execute_with_retry(_call, "generate_content")
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise GeminiServiceError(
            "llm_failed",
            "The AI model returned an empty response. Please try rephrasing your question.",
        )
    return text
