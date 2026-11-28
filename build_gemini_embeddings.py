"""
Generate course transcript embeddings using Google Gemini API (`gemini-embedding-001`)
and save them to `embeddings_gemini.joblib` without touching the original `embeddings.joblib`.

Includes incremental checkpointing and automatic free-tier rate-limit window waiting
so batch indexing across all 1,406 chunks can resume cleanly without wasting quota.

Usage:
    1. Set GEMINI_API_KEY in `.env` (or as an environment variable).
    2. Run: python build_gemini_embeddings.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import config
from rag.gemini_client import GeminiServiceError, gemini_embed_texts, is_gemini_configured

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _natural_sort_key(filename: str) -> tuple:
    prefix = filename.split("_", 1)[0]
    return (0, int(prefix), filename) if prefix.isdigit() else (1, 0, filename)


def load_chunks(input_dir: Path) -> list[dict]:
    """Read all merged transcript JSON files deterministically from `input_dir`."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {input_dir}")

    json_files = sorted(
        (f for f in os.listdir(input_dir) if f.endswith(".json")),
        key=_natural_sort_key,
    )
    if not json_files:
        raise ValueError(f"No .json files found in {input_dir}")

    records: list[dict] = []
    chunk_id = 0
    for json_file in json_files:
        path = input_dir / json_file
        with open(path, encoding="utf-8") as f:
            content = json.load(f)

        for chunk in content.get("chunks", []):
            records.append(
                {
                    "number": str(chunk["number"]),
                    "title": str(chunk["title"]),
                    "start": float(chunk["start"]),
                    "end": float(chunk["end"]),
                    "text": str(chunk["text"]),
                    "chunk_id": chunk_id,
                }
            )
            chunk_id += 1

    return records


def _embed_batch_with_quota_wait(texts: list[str], max_rate_waits: int = 8) -> list[list[float]]:
    """
    Call `gemini_embed_texts` and, if free-tier RPM/TPM quota is reached, wait for the
    server-reported retry window (or ~62s) before retrying. Stops immediately on daily quota.
    """
    waits = 0
    while True:
        try:
            return gemini_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
        except GeminiServiceError as exc:
            if exc.error_code == "rate_limited" and not exc.is_daily_quota and waits < max_rate_waits:
                wait_sec = max(float(exc.retry_after or 0.0) + 3.0, 15.0)
                waits += 1
                logger.info(
                    "Free-tier per-minute limit reached; sleeping %.1fs before retry (%d/%d)...",
                    wait_sec,
                    waits,
                    max_rate_waits,
                )
                time.sleep(wait_sec)
                continue
            raise


def build_embeddings(
    input_dir: Path,
    output_path: Path,
    batch_size: int = 50,
    delay_seconds: float = 0.5,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Embed all transcript chunks with Gemini and persist the validated DataFrame."""
    if not is_gemini_configured():
        raise GeminiServiceError(
            "gemini_not_configured",
            "GEMINI_API_KEY is not set. Add GEMINI_API_KEY to .env or your environment first.",
        )

    # Guard against accidentally overwriting the Ollama BGE-M3 index
    default_ollama_path = (config.BASE_DIR / "embeddings.joblib").resolve()
    if output_path.resolve() == default_ollama_path:
        raise ValueError("Refusing to overwrite original embeddings.joblib. Use a separate output path.")

    records = load_chunks(input_dir)
    total = len(records)
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", config.GEMINI_EMBEDDING_MODEL)
    dim = int(os.getenv("GEMINI_EMBEDDING_DIM", config.GEMINI_EMBEDDING_DIM))

    ckpt = checkpoint_path or output_path.with_suffix(".checkpoint.joblib")
    embedded_by_id: dict[int, np.ndarray] = {}
    if ckpt.exists():
        try:
            saved = joblib.load(ckpt)
            if isinstance(saved, dict):
                embedded_by_id = saved
                logger.info("Resuming from checkpoint %s (%d/%d chunks already embedded)", ckpt.name, len(embedded_by_id), total)
        except Exception as exc:
            logger.warning("Could not read checkpoint %s (%s); starting fresh", ckpt.name, exc)

    logger.info("Loaded %d chunks from %s. Embedding with %s (dim=%d)...", total, input_dir, model_name, dim)

    pending = [r for r in records if r["chunk_id"] not in embedded_by_id]
    api_batches = 0
    for start_idx in range(0, len(pending), batch_size):
        batch = pending[start_idx : start_idx + batch_size]
        texts = [
            f"Video {r['number']}: {r['title'].strip()} | {r['text'].strip()}"
            for r in batch
        ]
        vectors = _embed_batch_with_quota_wait(texts)
        api_batches += 1
        for rec, vec in zip(batch, vectors):
            embedded_by_id[rec["chunk_id"]] = np.asarray(vec, dtype=np.float32)

        joblib.dump(embedded_by_id, ckpt)
        logger.info("Embedded %d / %d chunks (API batches this run: %d)", len(embedded_by_id), total, api_batches)
        if start_idx + batch_size < len(pending) and delay_seconds > 0:
            time.sleep(delay_seconds)

    for rec in records:
        rec["embedding"] = embedded_by_id[rec["chunk_id"]]

    df = pd.DataFrame.from_records(records)
    validate_embeddings_df(df, expected_dim=dim)

    joblib.dump(df, output_path, compress=3)
    if ckpt.exists():
        ckpt.unlink()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %d Gemini embeddings to %s (%.2f MB)", len(df), output_path, size_mb)
    return df


def validate_embeddings_df(df: pd.DataFrame, expected_dim: int) -> None:
    """Validate schema, dimensions, finite values, and unit L2 norms."""
    required_cols = {"number", "title", "start", "end", "text", "chunk_id", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in embeddings DataFrame: {missing}")
    if df.empty:
        raise ValueError("Embeddings DataFrame is empty")

    matrix = np.vstack(df["embedding"].values)
    if matrix.shape[1] != expected_dim:
        raise ValueError(f"Expected embedding dim {expected_dim}, got {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or Inf values")

    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"Embeddings are not unit-normalized (min={norms.min():.4f}, max={norms.max():.4f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Gemini embeddings for course transcripts.")
    parser.add_argument("--input-dir", default=str(config.BASE_DIR / "newjsons"))
    parser.add_argument("--output", default=str(config.BASE_DIR / "embeddings_gemini.joblib"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    try:
        build_embeddings(
            input_dir=Path(args.input_dir),
            output_path=Path(args.output),
            batch_size=args.batch_size,
            delay_seconds=args.delay,
        )
        return 0
    except GeminiServiceError as exc:
        logger.error("Gemini error (%s, daily=%s): %s", exc.error_code, exc.is_daily_quota, exc.user_message)
        return 1
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
