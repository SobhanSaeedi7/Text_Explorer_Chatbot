"""Temporary in-session conversation memory for follow-up questions.

- History lives only in Streamlit `st.session_state` (a plain list of
  `Message`). It is never written to disk, ChromaDB, or any external
  database, and it disappears when the session ends.
- Follow-ups are rewritten into standalone retrieval queries using the
  recent history. The ORIGINAL user question is always kept for display
  and answer generation; only the rewritten query is used for retrieval.
- Answers stay grounded: history only resolves references, while factual
  claims must come from retrieved excerpts of the active document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import MutableMapping

from langchain_core.messages import HumanMessage, SystemMessage

from pelekan_rag_chat.llm import get_chat_model

HISTORY_KEY = "messages"

# Session-state keys that belong to the active document (chat included).
DOCUMENT_KEYS = (
    "loaded_filename",
    "loaded_text",
    "loaded_doc_id",
    "indexed_doc_id",
    "chunk_count",
    HISTORY_KEY,
)

# How many recent messages are sent to the model for query rewriting.
MAX_CONTEXT_MESSAGES = 6

REWRITE_SYSTEM_PROMPT = """آخرین سؤال کاربر در یک گفت‌وگو درباره‌ی یک سند را به یک سؤال مستقل و کامل برای جست‌وجو در سند بازنویسی کن.
مرجع‌های آن (مانند «این»، «آن»، «نتیجه»، "it" یا "the result") را با کمک تاریخچه‌ی گفت‌وگو مشخص کن.
اگر سؤال از قبل مستقل و کامل است، همان را بدون هیچ تغییری برگردان.
فقط سؤال بازنویسی‌شده را بنویس، بدون هیچ توضیح اضافه‌ای."""


@dataclass
class Message:
    """One chat turn kept in session state (never persisted)."""

    role: str  # "user" or "assistant"
    content: str
    # Retrieval excerpts the answer was built from (assistant only).
    # Stored so each answer stays displayed with the sources that
    # produced it. Kept in memory, never persisted.
    sources: list = field(default_factory=list)


def append_exchange(
    history: list[Message],
    question: str,
    answer: str,
    sources: list | None = None,
) -> list[Message]:
    """Return history with one user/assistant exchange appended, in order."""
    return history + [
        Message(role="user", content=question),
        Message(role="assistant", content=answer, sources=list(sources or [])),
    ]


def build_rewrite_messages(question: str, history: list[Message]) -> list:
    """Prompt for standalone-query rewriting from recent context."""
    lines = []
    for message in history[-MAX_CONTEXT_MESSAGES:]:
        speaker = "کاربر" if message.role == "user" else "دستیار"
        lines.append(f"{speaker}: {message.content}")
    lines.append(f"سؤال جدید: {question.strip()}")
    context = "\n".join(lines)
    return [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=f"تاریخچه‌ی گفت‌وگو:\n{context}"),
    ]


def rewrite_followup(
    question: str, history: list[Message], model=None
) -> str:
    """Rewrite a follow-up into a standalone retrieval query.

    - Empty history (or blank question): returns the question unchanged
      WITHOUT building a model or making any LLM call.
    - Otherwise one small rewrite call; the rewrite prompt itself returns
      already-standalone questions unchanged, so no separate
      standalone-detection heuristic is needed.
    - Never raises: any failure (missing key, network, empty output)
      falls back to the original question.
    """
    if not question or not question.strip():
        return question
    if not history:
        return question
    try:
        chat = model if model is not None else get_chat_model()
        response = chat.invoke(build_rewrite_messages(question, history))
        rewritten = (getattr(response, "content", "") or "").strip()
        return rewritten or question
    except Exception:
        return question


def reset_conversation_state(state: MutableMapping) -> None:
    """Clear chat history only. Document and ChromaDB index are untouched."""
    state.pop(HISTORY_KEY, None)


def reset_document_state(state: MutableMapping) -> None:
    """Clear the active document AND its conversation history.

    Used on replace/remove/indexing failure so follow-ups can never use
    the previous document's history or chunks. ChromaDB deletion itself
    stays in the app layer (`store.clear_document`); this only drops
    in-memory session keys.
    """
    for key in DOCUMENT_KEYS:
        state.pop(key, None)
