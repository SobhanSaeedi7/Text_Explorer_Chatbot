from __future__ import annotations

import hashlib


#ُSplit text into chunks 
def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative.")

    text = text.strip()
    if not text:
        return []
    
    # always >= 1, so the loop terminates
    overlap = min(chunk_overlap, chunk_size - 1)
    step = chunk_size - overlap  

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

#Make an id for file to stop reindex same files 
def make_doc_id(filename: str, text: str) -> str:
    digest = hashlib.sha256(f"{filename}\n{text}".encode("utf-8")).hexdigest()
    return digest