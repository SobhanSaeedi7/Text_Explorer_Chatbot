"""Split plain text into overlapping chunks (no extra framework).

Character-based splitting works the same for Persian and English and
needs no tokenizer. The original text is preserved: chunks are plain
slices of the document (only the whole-document edges are stripped).
"""

from __future__ import annotations

import hashlib


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """Split `text` into overlapping chunks.

    - `chunk_size`: max characters per chunk (must be > 0).
    - `chunk_overlap`: characters shared with the previous chunk.
      Values >= `chunk_size` are clamped to `chunk_size - 1`, so the
      window always moves forward (no infinite loop, no empty chunks).
    - Returns [] for empty or whitespace-only input.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative.")

    text = text.strip()
    if not text:
        return []

    overlap = min(chunk_overlap, chunk_size - 1)
    step = chunk_size - overlap  # always >= 1, so the loop terminates

    chunks: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        chunk = text[start : start + chunk_size]
        if chunk.strip():  # skip whitespace-only windows
            chunks.append(chunk)
        if start + chunk_size >= total:
            break
        start += step
    return chunks


def make_doc_id(filename: str, text: str) -> str:
    """Stable id for one document version (filename + content hash).

    Re-uploading the exact same file yields the same id, so the app can
    skip re-indexing. Any content or filename change yields a new id.
    """
    digest = hashlib.sha256(f"{filename}\n{text}".encode("utf-8")).hexdigest()
    return digest
