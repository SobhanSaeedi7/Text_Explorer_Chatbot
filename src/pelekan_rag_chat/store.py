from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from pelekan_rag_chat import config
from pelekan_rag_chat.chunking import split_text

_embedding_function: DefaultEmbeddingFunction | None = None


def get_embedding_function() -> DefaultEmbeddingFunction:
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = DefaultEmbeddingFunction()
    return _embedding_function

#Open ChromaDB database
def get_client(persist_dir: str | None = None) -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir or config.CHROMA_DIR)

#Document chunks collection
def get_collection(client, name: str | None = None, embedding_function=None):
    return client.get_or_create_collection(
        name or config.COLLECTION_NAME,
        embedding_function=embedding_function or get_embedding_function(),
    )


def is_indexed(collection, doc_id: str) -> bool:
    result = collection.get(where={"doc_id": doc_id}, limit=1)
    return len(result["ids"]) > 0


#Split, embed and store the document

def index_document(
    collection,
    text: str,
    filename: str,
    doc_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    if is_indexed(collection, doc_id):
        stored = collection.get(where={"doc_id": doc_id})
        return len(stored["ids"])

    chunks = split_text(
        text,
        chunk_size if chunk_size is not None else config.CHUNK_SIZE,
        chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError("Nothing to index: the document produced no chunks.")

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

    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


#Delete chunks of any other document
def remove_other_documents(collection, doc_id: str) -> None:
    collection.delete(where={"doc_id": {"$ne": doc_id}})

#To remove the document file
def clear_document(collection, doc_id: str | None) -> None:
    if not doc_id:
        return
    collection.delete(where={"doc_id": doc_id})
