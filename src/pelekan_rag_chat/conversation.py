from __future__ import annotations

from dataclasses import dataclass, field
from typing import MutableMapping

from langchain_core.messages import HumanMessage, SystemMessage

from pelekan_rag_chat.llm import get_chat_model

#Save chat history in session-state key
HISTORY_KEY = "messages"

#Session-state keys that belong to the active document (chat included).
DOCUMENT_KEYS = (
    "loaded_filename",
    "loaded_text",
    "loaded_doc_id",
    "indexed_doc_id",
    "chunk_count",
    HISTORY_KEY,
)

# How many recent messages are sent to the model for query rewriting(for related questions)
MAX_CONTEXT_MESSAGES = 6

REWRITE_SYSTEM_PROMPT = """آخرین سؤال کاربر در یک گفت‌وگو درباره‌ی یک سند را به یک سؤال مستقل و کامل برای جست‌وجو در سند بازنویسی کن.
مرجع‌های آن (مانند «این»، «آن»، «نتیجه»، "it" یا "the result") را با کمک تاریخچه‌ی گفت‌وگو مشخص کن.
اگر سؤال از قبل مستقل و کامل است، همان را بدون هیچ تغییری برگردان.
فقط سؤال بازنویسی‌شده را بنویس، بدون هیچ توضیح اضافه‌ای."""


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str 
    # Retrieval excerpts the answer was built from (assistant only).
    # Stored so each answer stays displayed with the sources produced it
    sources: list = field(default_factory=list)


#Append the last messages to chat history
def append_exchange(
    history: list[Message],
    question: str,
    answer: str,
    sources: list | None = None,
) -> list[Message]:
    return history + [
        Message(role="user", content=question),
        Message(role="assistant", content=answer, sources=list(sources or [])),
    ]

#Make prompt for rewriting the question
def build_rewrite_messages(question: str, history: list[Message]) -> list:
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

#Specify the meaning of follow-up question and rewrite it into a standalone one
def rewrite_followup(
    question: str, history: list[Message], model=None
) -> str:
    #return the question if it be empty
    if not question or not question.strip():
        return question
    #return the question if there is no history
    if not history:
        return question
    try:
        #use the input model or make one if there isn`t any
        chat = model if model is not None else get_chat_model()
        response = chat.invoke(build_rewrite_messages(question, history))
        rewritten = (getattr(response, "content", "") or "").strip()
        return rewritten or question
    except Exception:
        return question

#Clear chat history
def reset_conversation_state(state: MutableMapping) -> None:
    state.pop(HISTORY_KEY, None)

#Clear the active document and whole history
def reset_document_state(state: MutableMapping) -> None:
    for key in DOCUMENT_KEYS:
        state.pop(key, None)
