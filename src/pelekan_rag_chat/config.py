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


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    """Read an int setting from the environment, falling back to default."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


MAX_UPLOAD_MB: int = _read_int("MAX_UPLOAD_MB", 2)
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

# Text chunking (character-based) and ChromaDB collection settings.
CHUNK_SIZE: int = _read_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP: int = _read_int("CHUNK_OVERLAP", 200, minimum=0)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "pelekan_docs")
