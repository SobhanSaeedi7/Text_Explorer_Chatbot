"""Minimal configuration (upload stage).

Reads optional environment variables so later stages (embeddings,
vector store, LLM) have a single place for settings. No network
calls or model loading happens here.
"""

import os

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "data/chroma")


def _read_max_upload_mb(default: int = 2) -> int:
    """Read MAX_UPLOAD_MB from the environment, falling back to default."""
    try:
        return max(1, int(os.getenv("MAX_UPLOAD_MB", str(default))))
    except (TypeError, ValueError):
        return default


MAX_UPLOAD_MB: int = _read_max_upload_mb()
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
