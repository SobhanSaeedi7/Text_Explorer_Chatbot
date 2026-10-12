"""Unit tests for temporary conversation memory and follow-up rewriting.

Mocks/fakes only: no live AVAL AI calls, no real embedding model.
"""

from types import SimpleNamespace

import chromadb
import pytest

from fake_embeddings import HashEmbeddingFunction
from pelekan_rag_chat import llm, store
from pelekan_rag_chat.chunking import make_doc_id
from pelekan_rag_chat.conversation import (
    Message,
    append_exchange,
    build_rewrite_messages,
    reset_conversation_state,
    reset_document_state,
    rewrite_followup,
)
from pelekan_rag_chat.retrieval import retrieve


class FakeRewriteModel:
    """Canned rewriter: returns `reply` or raises `error` when invoked."""

    def __init__(self, reply="rewritten query", error=None):
        self.reply = reply
        self.error = error
        self.calls = 0
        self.seen_messages = None

    def invoke(self, messages):
        self.calls += 1
        self.seen_messages = messages
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.reply)


def _history():
    return [
        Message(role="user", content="بازی اخیر تیم ملی چه زمانی بود؟"),
        Message(role="assistant", content="تیم ملی جمعه گذشته با برزیل بازی کرد."),
    ]


def test_empty_history_uses_original_without_model_call():
    model = FakeRewriteModel(reply="SHOULD NOT BE USED")
    assert rewrite_followup("نتیجه چه شد؟", [], model=model) == "نتیجه چه شد؟"
    assert model.calls == 0


def test_followup_rewritten_with_recent_context():
    model = FakeRewriteModel(reply="بازی تیم ملی با برزیل چه نتیجه‌ای داشت؟")
    result = rewrite_followup("نتیجه چه شد؟", _history(), model=model)
    assert result == "بازی تیم ملی با برزیل چه نتیجه‌ای داشت؟"
    assert model.calls == 1
    # Recent context was actually sent for rewriting.
    sent = " ".join(m.content for m in model.seen_messages)
    assert "برزیل" in sent
    assert "نتیجه چه شد؟" in sent


def test_original_question_kept_for_display_and_answering():
    history = _history()
    snapshot = [Message(m.role, m.content, list(m.sources)) for m in history]
    original = "نتیجه چه شد؟  "
    rewrite_followup(original, history, model=FakeRewriteModel())
    assert original == "نتیجه چه شد؟  "  # untouched, still displayable
    assert history == snapshot  # history not mutated by rewriting
    updated = append_exchange(history, original.strip(), "۲ بر ۱ برد.")
    assert updated[-2] == Message(role="user", content="نتیجه چه شد؟")
    assert updated[-1].content == "۲ بر ۱ برد."


def test_rewrite_failure_falls_back_to_original():
    model = FakeRewriteModel(error=ConnectionError("offline"))
    assert rewrite_followup("نتیجه چه شد؟", _history(), model=model) == "نتیجه چه شد؟"
    empty = FakeRewriteModel(reply="   ")
    assert rewrite_followup("نتیجه چه شد؟", _history(), model=empty) == "نتیجه چه شد؟"


def test_rewrite_failure_without_api_key_falls_back(monkeypatch):
    from pelekan_rag_chat import config

    monkeypatch.setattr(config, "AVALAI_API_KEY", "")
    # No model injected and no key configured: must not raise.
    assert rewrite_followup("نتیجه چه شد؟", _history()) == "نتیجه چه شد؟"


def test_retrieval_stays_restricted_to_active_doc(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = store.get_collection(
        client, "conv_docs", embedding_function=HashEmbeddingFunction()
    )
    text_a = "The national team played Brazil on Friday. " * 60
    text_b = "Quantum computers use qubits for calculation. " * 60
    doc_a = make_doc_id("match.txt", text_a)
    doc_b = make_doc_id("physics.txt", text_b)
    store.index_document(collection, text_a, "match.txt", doc_a,
                         chunk_size=200, chunk_overlap=20)
    store.index_document(collection, text_b, "physics.txt", doc_b,
                         chunk_size=200, chunk_overlap=20)

    rewritten = "What was the result of the national team match against Brazil?"
    sources = retrieve(collection, rewritten, doc_a, top_k=4)
    assert sources
    assert all(s.filename == "match.txt" for s in sources)


def test_history_kept_in_chronological_order():
    history = []
    history = append_exchange(history, "سؤال اول", "پاسخ اول")
    history = append_exchange(history, "سؤال دوم", "پاسخ دوم")
    assert [(m.role, m.content) for m in history] == [
        ("user", "سؤال اول"),
        ("assistant", "پاسخ اول"),
        ("user", "سؤال دوم"),
        ("assistant", "پاسخ دوم"),
    ]


def test_sources_stay_with_their_answer():
    history = append_exchange([], "س؟", "ج.", sources=["excerpt-1"])
    assert history[-1].sources == ["excerpt-1"]
    assert history[-2].sources == []


def test_clearing_conversation_keeps_document_and_index(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = store.get_collection(
        client, "keep_docs", embedding_function=HashEmbeddingFunction()
    )
    doc_id = make_doc_id("notes.txt", "Some persistent content here. " * 40)
    store.index_document(collection, "Some persistent content here. " * 40,
                         "notes.txt", doc_id, chunk_size=200, chunk_overlap=20)

    state = {
        "loaded_filename": "notes.txt",
        "loaded_text": "Some persistent content here.",
        "loaded_doc_id": doc_id,
        "indexed_doc_id": doc_id,
        "chunk_count": 3,
        "messages": [Message("user", "س؟"), Message("assistant", "ج.")],
    }
    reset_conversation_state(state)
    assert "messages" not in state  # history gone…
    assert state["loaded_doc_id"] == doc_id  # …document state kept…
    assert state["indexed_doc_id"] == doc_id
    assert store.is_indexed(collection, doc_id)  # …and the index untouched.
    assert collection.count() > 0


def test_replacing_document_clears_history_answer_and_sources():
    state = {
        "loaded_filename": "old.txt",
        "loaded_text": "old content",
        "loaded_doc_id": "old-id",
        "indexed_doc_id": "old-id",
        "chunk_count": 5,
        "messages": [Message("user", "س؟"), Message("assistant", "ج.")],
    }
    reset_document_state(state)
    assert state == {}


def test_insufficient_evidence_fallback_preserved():
    assert llm.FALLBACK_ANSWER == "اطلاعات کافی در سند موجود نیست."
    assert llm.FALLBACK_ANSWER in llm.SYSTEM_PROMPT
    model = FakeRewriteModel()
    with pytest.raises(ValueError):
        llm.generate_answer("سؤال؟", [], model=model)
    assert model.calls == 0  # no model call without retrieved evidence
