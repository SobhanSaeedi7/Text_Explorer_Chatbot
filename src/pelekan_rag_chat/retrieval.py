"""Retrieve relevant chunks for a question (no re-indexing, no LLM here).

Queries the existing persistent ChromaDB collection with the
collection's own embedding function. Only the short question text is
embedded per query; document embeddings are reused, never recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass

from pelekan_rag_chat import config


@dataclass
class Source:
    """One retrieved excerpt, tied to the answer that used it."""

    chunk_id: str
    text: str
    filename: str
    chunk_index: int


def format_source_label(source: Source) -> str:
    """Short caption for a source (only metadata the TXT index stores)."""
    return f"{source.filename} — بخش {source.chunk_index}"


def retrieve(
    collection,
    question: str,
    doc_id: str | None,
    top_k: int | None = None,
) -> list[Source]:
    """Return up to `top_k` chunks of the active document only.

    - Empty/whitespace questions return [] (no model call needed).
    - A missing `doc_id` raises ValueError (the UI reports it in Persian).
    - Results are restricted with `where={"doc_id": doc_id}`, so chunks
      from any other document can never be returned.
    - An unknown/empty document yields [] (caller shows "nothing found").
    """
    if not question or not question.strip():
        return []
    if not doc_id:
        raise ValueError("No active document.")
    k = top_k if top_k is not None and top_k > 0 else config.TOP_K

    result = collection.query(
        query_texts=[question.strip()],
        n_results=k,
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]

    sources: list[Source] = []
    for chunk_id, text, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        sources.append(
            Source(
                chunk_id=chunk_id,
                text=text or "",
                filename=str(meta.get("filename", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
            )
        )
    return sources
