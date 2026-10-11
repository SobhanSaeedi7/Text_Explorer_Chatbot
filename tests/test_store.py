"""Unit tests for ChromaDB indexing (real persistent client, fake embeddings).

These are ordinary unit tests: they use a tiny deterministic
hash-based embedding function, so they never download or initialize
ChromaDB's real embedding model. They run offline and fast.
The real `DefaultEmbeddingFunction` is covered separately by the manual
smoke test in `tests/test_default_embedding_smoke.py`.

A temporary directory is used, so tests never touch `data/chroma`.
"""

import hashlib

import chromadb
import pytest

from pelekan_rag_chat import store
from pelekan_rag_chat.chunking import make_doc_id


class HashEmbeddingFunction:
    """Lightweight deterministic stand-in (8-dim sha256 vectors).

    Implements the same protocol the installed ChromaDB version expects
    (`__call__` for documents, `embed_query` for queries, `name`).
    """

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def embed_query(self, input):
        return [self._embed(text) for text in input]

    def name(self):
        return "test_hash_embedding"

    @staticmethod
    def _embed(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[:8]]


@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return store.get_collection(
        client, "test_docs", embedding_function=HashEmbeddingFunction()
    )


def test_fake_embedding_is_deterministic():
    ef = HashEmbeddingFunction()
    assert ef(["hello"]) == ef(["hello"])
    assert ef(["hello"]) != ef(["world"])
    assert ef.embed_query(["hello"]) == ef(["hello"])


def test_index_once_and_reuse_without_recompute(collection):
    text = "Hello world. " * 200
    doc_id = make_doc_id("notes.txt", text)

    first = store.index_document(collection, text, "notes.txt", doc_id,
                                 chunk_size=200, chunk_overlap=20)
    assert first > 1
    assert store.is_indexed(collection, doc_id)

    # Second call must not recompute: same count, no duplicate ids.
    second = store.index_document(collection, text, "notes.txt", doc_id,
                                  chunk_size=200, chunk_overlap=20)
    assert second == first
    assert collection.count() == first


def test_chunk_metadata_present(collection):
    doc_id = make_doc_id("farsi.txt", "سلام دنیا")
    store.index_document(collection, "سلام دنیا", "farsi.txt", doc_id,
                         chunk_size=1000, chunk_overlap=0)
    got = collection.get(where={"doc_id": doc_id})
    assert got["ids"] == [f"{doc_id}#0"]
    meta = got["metadatas"][0]
    assert meta["doc_id"] == doc_id
    assert meta["chunk_index"] == 0
    assert meta["filename"] == "farsi.txt"


def test_replacement_removes_previous_chunks(collection):
    text_a = "Apples are red. " * 100
    text_b = "Persian cats are calm. " * 100
    doc_a = make_doc_id("a.txt", text_a)
    doc_b = make_doc_id("b.txt", text_b)

    store.index_document(collection, text_a, "a.txt", doc_a,
                         chunk_size=200, chunk_overlap=20)
    assert store.is_indexed(collection, doc_a)

    store.index_document(collection, text_b, "b.txt", doc_b,
                         chunk_size=200, chunk_overlap=20)
    store.remove_other_documents(collection, doc_b)

    assert not store.is_indexed(collection, doc_a)
    assert store.is_indexed(collection, doc_b)
    leftovers = collection.get(where={"doc_id": doc_a})
    assert leftovers["ids"] == []


def test_clear_document_on_removal(collection):
    doc_id = make_doc_id("gone.txt", "temporary content here")
    store.index_document(collection, "temporary content here", "gone.txt",
                         doc_id, chunk_size=1000, chunk_overlap=0)
    store.clear_document(collection, doc_id)
    assert not store.is_indexed(collection, doc_id)


def test_empty_text_raises(collection):
    with pytest.raises(ValueError):
        store.index_document(collection, "   ", "empty.txt", "doc-empty",
                             chunk_size=100, chunk_overlap=10)


def test_embedding_function_consistent_on_reopen(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col1 = store.get_collection(
        client, "reopen-docs", embedding_function=HashEmbeddingFunction()
    )
    col2 = store.get_collection(
        client, "reopen-docs", embedding_function=HashEmbeddingFunction()
    )
    assert type(col1._embedding_function) is type(col2._embedding_function)


def test_persistence_across_client_reopen(tmp_path):
    path = str(tmp_path / "chroma")
    doc_id = make_doc_id("persist.txt", "persistent content here")
    col = store.get_collection(
        chromadb.PersistentClient(path=path),
        "persist-docs",
        embedding_function=HashEmbeddingFunction(),
    )
    store.index_document(col, "persistent content here", "persist.txt",
                         doc_id, chunk_size=1000, chunk_overlap=0)

    reopened = store.get_collection(
        chromadb.PersistentClient(path=path),
        "persist-docs",
        embedding_function=HashEmbeddingFunction(),
    )
    assert store.is_indexed(reopened, doc_id)
    assert reopened.count() == 1
