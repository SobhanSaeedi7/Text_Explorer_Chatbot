"""Manual smoke test: real ChromaDB `DefaultEmbeddingFunction` end to end.

NOT an ordinary unit test: it initializes ChromaDB's own default
embedding model (downloaded once to the local Chroma cache, ~80 MB) and
needs network access on the very first run. It is therefore skipped by
default and never runs during `uv run pytest` / `make test`.

Run it explicitly when you want to verify the production embedding path:
    PELEKAN_RUN_CHROMA_SMOKE=1 uv run pytest tests/test_default_embedding_smoke.py -v

It uses a temporary directory, so it never touches `data/chroma`.
"""

import os

import chromadb
import pytest

from pelekan_rag_chat import store
from pelekan_rag_chat.chunking import make_doc_id

pytestmark = pytest.mark.skipif(
    os.getenv("PELEKAN_RUN_CHROMA_SMOKE") != "1",
    reason="Manual smoke test: set PELEKAN_RUN_CHROMA_SMOKE=1 to run "
    "(downloads/initializes the real embedding model).",
)


def test_default_embedding_end_to_end(tmp_path):
    # Production path: no injected function, so the collection uses
    # ChromaDB's own DefaultEmbeddingFunction.
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = store.get_collection(client, "smoke-docs")
    assert type(collection._embedding_function).__name__ == (
        "DefaultEmbeddingFunction"
    )

    text_en = "The Pelekan bird flies over the lake. " * 30
    text_fa = "پلیکان بر فراز دریاچه پرواز می‌کند. " * 30
    doc_en = make_doc_id("english.txt", text_en)
    doc_fa = make_doc_id("farsi.txt", text_fa)

    n_en = store.index_document(collection, text_en, "english.txt", doc_en,
                                chunk_size=200, chunk_overlap=20)
    n_fa = store.index_document(collection, text_fa, "farsi.txt", doc_fa,
                                chunk_size=200, chunk_overlap=20)
    assert n_en > 0 and n_fa > 0

    # Embeddings were generated automatically: a query resolves back to
    # the matching document's chunks.
    hits = collection.query(query_texts=["bird over the lake"], n_results=1)
    assert hits["ids"][0][0].startswith(doc_en)
