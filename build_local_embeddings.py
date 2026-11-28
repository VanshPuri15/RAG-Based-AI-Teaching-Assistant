"""
Generate course transcript embeddings locally using `BAAI/bge-small-en-v1.5` (384 dims)
via FastEmbed (ONNX Runtime) and save them to `embeddings_local.joblib`.

Strictly preserves `embeddings.joblib`, `embeddings_gemini.joblib`, and
`embeddings_gemini.checkpoint.joblib` without modification.

Usage:
    python build_local_embeddings.py
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
from rag.local_embedder import LocalEmbeddingError, local_embed_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROTECTED_FILES = {
    "embeddings.joblib",
    "embeddings_gemini.joblib",
    "embeddings_gemini.checkpoint.joblib",
}


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


def format_document_text(record: dict) -> str:
    """Format a transcript chunk consistently for passage embedding."""
    return f"Video {record['number']}: {record['title'].strip()} | {record['text'].strip()}"


def validate_embeddings_df(df: pd.DataFrame, expected_dim: int = 384) -> None:
    """Validate schema, dimensions, float32 dtype, finiteness, and unit L2 norms."""
    required_cols = {"number", "title", "start", "end", "text", "chunk_id", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in embeddings DataFrame: {missing}")
    if df.empty:
        raise ValueError("Embeddings DataFrame is empty")

    matrix = np.vstack(df["embedding"].values)
    if matrix.shape[1] != expected_dim:
        raise ValueError(f"Expected embedding dim {expected_dim}, got {matrix.shape[1]}")
    if matrix.dtype != np.float32:
        raise ValueError(f"Expected float32 embeddings, got {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or Inf values")

    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(f"Embeddings are not unit-normalized (min={norms.min():.6f}, max={norms.max():.6f})")


def build_local_embeddings(
    input_dir: Path,
    output_path: Path,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Embed all transcript chunks with BAAI/bge-small-en-v1.5 and save `embeddings_local.joblib`."""
    if output_path.name in PROTECTED_FILES:
        raise ValueError(f"Refusing to overwrite protected file: {output_path.name}")

    records = load_chunks(input_dir)
    total = len(records)
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", config.LOCAL_EMBEDDING_MODEL)
    dim = int(os.getenv("LOCAL_EMBEDDING_DIM", config.LOCAL_EMBEDDING_DIM))
    logger.info("Loaded %d chunks from %s. Embedding locally with %s (dim=%d)...", total, input_dir, model_name, dim)

    t0 = time.time()
    texts = [format_document_text(r) for r in records]
    matrix = local_embed_texts(texts, is_query=False, batch_size=batch_size)

    for rec, vec in zip(records, matrix):
        rec["embedding"] = vec

    df = pd.DataFrame.from_records(records)
    validate_embeddings_df(df, expected_dim=dim)

    joblib.dump(df, output_path, compress=3)
    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %d local embeddings to %s (%.2f MB) in %.2fs", len(df), output_path, size_mb, elapsed)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local BGE-small-en-v1.5 embeddings (384 dims).")
    parser.add_argument("--input-dir", default=str(config.BASE_DIR / "newjsons"))
    parser.add_argument("--output", default=str(config.BASE_DIR / "embeddings_local.joblib"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        build_local_embeddings(
            input_dir=Path(args.input_dir),
            output_path=Path(args.output),
            batch_size=args.batch_size,
        )
        return 0
    except LocalEmbeddingError as exc:
        logger.error("Local embedding error (%s): %s", exc.error_code, exc.user_message)
        return 1
    except Exception as exc:
        logger.error("Local embedding build failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
