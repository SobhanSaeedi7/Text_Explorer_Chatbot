"""Persistent ChromaDB index for the active document's chunks.

- Uses ChromaDB's own DefaultEmbeddingFunction: ChromaDB downloads the
  model once (to the local Chroma cache) and generates embeddings
  automatically when chunks are added. We never embed text manually.
- Storage is persistent under CHROMA_DIR: ChromaDB keeps collection
  metadata in a `chroma.sqlite3` file plus local vector-index segment
  files next to it (not a single SQLite file).
- Only one document is active at a time; every chunk carries `doc_id`,
  `chunk_index`, and `filename` metadata.
"""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from pelekan_rag_chat import config
from pelekan_rag_chat.chunking import split_text

_embedding_function: DefaultEmbeddingFunction | None = None


def get_embedding_function() -> DefaultEmbeddingFunction:
    """Single shared default embedding function (create and reopen agree)."""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = DefaultEmbeddingFunction()
    return _embedding_function


def get_client(persist_dir: str | None = None) -> chromadb.api.ClientAPI:
    """Open the persistent local ChromaDB database (reused across restarts)."""
    return chromadb.PersistentClient(path=persist_dir or config.CHROMA_DIR)


def get_collection(client, name: str | None = None, embedding_function=None):
    """Open (or create) the document-chunks collection.

    Production passes no `embedding_function`, so the collection always
    uses ChromaDB's own default embedding function (the same one on
    create and reopen, keeping stored embeddings valid). Tests may inject
    a lightweight deterministic function to avoid downloading or
    initializing the real model.
    """
    return client.get_or_create_collection(
        name or config.COLLECTION_NAME,
        embedding_function=embedding_function or get_embedding_function(),
    )


def is_indexed(collection, doc_id: str) -> bool:
    """True if this document version already has chunks in the index."""
    result = collection.get(where={"doc_id": doc_id}, limit=1)
    return len(result["ids"]) > 0


def index_document(
    collection,
    text: str,
    filename: str,
    doc_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Split, embed (automatically), and store one document. Idempotent:
    if this `doc_id` is already indexed, nothing is recomputed and the
    stored chunk count is returned. Returns the number of chunks."""
    chunks = split_text(
        text,
        chunk_size if chunk_size is not None else config.CHUNK_SIZE,
        chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
    )
    if not chunks:
        raise ValueError("Nothing to index: the document produced no chunks.")
    if is_indexed(collection, doc_id):
        stored = collection.get(where={"doc_id": doc_id})
        return len(stored["ids"])
    ids = [f"{doc_id}#{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "chunk_index": i,
            "filename": filename,
            "chunk_count": len(chunks),
        }
        for i in range(len(chunks))
    ]
    # No embeddings passed: the collection's default embedding function
    # generates them automatically here.
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def remove_other_documents(collection, doc_id: str) -> None:
    """Delete chunks from any previous/replaced document so retrieval for
    the new file can never see stale chunks (also cleans leftovers from
    earlier app runs sharing the same persistent database)."""
    collection.delete(where={"doc_id": {"$ne": doc_id}})


def clear_document(collection, doc_id: str | None) -> None:
    """Delete one document's chunks (used when its file is removed)."""
    if not doc_id:
        return
    collection.delete(where={"doc_id": doc_id})
