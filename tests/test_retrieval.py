"""Unit tests for retrieval (fake embeddings, no network, no real model)."""

import chromadb
import pytest

from fake_embeddings import HashEmbeddingFunction
from pelekan_rag_chat import config, store
from pelekan_rag_chat.chunking import make_doc_id
from pelekan_rag_chat.retrieval import format_source_label, retrieve


@pytest.fixture
def two_doc_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = store.get_collection(
        client, "rag_docs", embedding_function=HashEmbeddingFunction()
    )
    text_a = "The Pelekan bird flies over the lake at dawn. " * 60
    text_b = "Quantum computers use qubits for calculation. " * 60
    doc_a = make_doc_id("birds.txt", text_a)
    doc_b = make_doc_id("physics.txt", text_b)
    store.index_document(collection, text_a, "birds.txt", doc_a,
                         chunk_size=200, chunk_overlap=20)
    store.index_document(collection, text_b, "physics.txt", doc_b,
                         chunk_size=200, chunk_overlap=20)
    return collection, doc_a, doc_b


def test_retrieves_chunks_with_source_metadata(two_doc_collection):
    collection, doc_a, _ = two_doc_collection
    sources = retrieve(collection, "bird over the lake", doc_a, top_k=4)
    assert 0 < len(sources) <= 4
    for source in sources:
        assert source.text.strip()
        assert source.filename == "birds.txt"
        assert source.chunk_id.startswith(doc_a)


def test_results_restricted_to_active_document(two_doc_collection):
    collection, doc_a, doc_b = two_doc_collection
    for source in retrieve(collection, "qubits calculation", doc_a, top_k=4):
        assert source.filename == "birds.txt"
        assert not source.chunk_id.startswith(doc_b)
    for source in retrieve(collection, "bird lake dawn", doc_b, top_k=4):
        assert source.filename == "physics.txt"


def test_top_k_default_and_limit(two_doc_collection):
    collection, doc_a, _ = two_doc_collection
    assert config.TOP_K == 4
    assert len(retrieve(collection, "bird", doc_a, top_k=1)) == 1
    assert len(retrieve(collection, "bird", doc_a)) <= config.TOP_K


def test_empty_results_for_unknown_document(two_doc_collection):
    collection, _, _ = two_doc_collection
    assert retrieve(collection, "anything", "no-such-doc", top_k=4) == []


def test_empty_question_returns_nothing(two_doc_collection):
    collection, doc_a, _ = two_doc_collection
    before = collection.count()
    assert retrieve(collection, "   ", doc_a) == []
    assert collection.count() == before  # no re-indexing on questions


def test_missing_doc_id_raises(two_doc_collection):
    collection, _, _ = two_doc_collection
    with pytest.raises(ValueError):
        retrieve(collection, "bird", None)


def test_source_label_has_only_real_metadata(two_doc_collection):
    collection, doc_a, _ = two_doc_collection
    (source,) = retrieve(collection, "bird", doc_a, top_k=1)
    label = format_source_label(source)
    assert "birds.txt" in label
    assert str(source.chunk_index) in label
    assert "page" not in label.lower()
