""" Scope of this stage:
- Upload exactly one .txt file, validate, split, and index it once.
- Retrieve relevant chunks per question (active document only).
- Generate grounded Persian answers with AVAL AI + show sources.
- Temporary in-session conversation history with follow-up support
  (rewritten into standalone retrieval queries). History is never
  persisted to disk or any database.
"""

from __future__ import annotations

import streamlit as st

from pelekan_rag_chat import config
from pelekan_rag_chat.chunking import make_doc_id
from pelekan_rag_chat.conversation import (
    HISTORY_KEY,
    append_exchange,
    reset_conversation_state,
    reset_document_state,
    rewrite_followup,
)
from pelekan_rag_chat.document import process_upload
from pelekan_rag_chat import store
from pelekan_rag_chat.llm import generate_answer
from pelekan_rag_chat.retrieval import format_source_label, retrieve


@st.cache_resource #Don`t remake database with UI rerun
def _get_collection():
    return store.get_collection(store.get_client())


def _reset_loaded_file() -> None:
    reset_document_state(st.session_state)
    st.session_state.pop("last_result", None)

#Clear previous document chunks
def _invalidate_previous_doc(collection, new_doc_id: str | None) -> None:
    old_doc_id = st.session_state.get("loaded_doc_id")
    if collection is not None and old_doc_id and old_doc_id != new_doc_id:
        try:
            store.clear_document(collection, old_doc_id)
        except Exception:
            pass


def main() -> None:
    st.title("Text Document Files Chatbot")
    st.write("Upload one `.txt` file, then ask a question about it.")
    st.caption(f"Maximum file size: {config.MAX_UPLOAD_MB} MB. Only `.txt` files.")
    # st.caption(
    #     "First indexing downloads ChromaDB's default embedding model once "
    #     "(~80 MB, cached locally afterwards)."
    # )

    try:
        collection = _get_collection()
    except Exception as exc:
        collection = None
        st.error(f"Could not open the local index: {exc}")

    uploaded = st.file_uploader(
        "Choose a .txt file",
        type=["txt"],
        accept_multiple_files=False,
        help=f"Single .txt file up to {config.MAX_UPLOAD_MB} MB.",
    )

    if uploaded is None:
        st.warning("DEBUG: uploaded is None; resetting document state")
        _invalidate_previous_doc(collection, None)
        _reset_loaded_file()
        st.info("No file selected yet.")
    elif collection is None:
        st.error("Index is unavailable, so the file cannot be processed.")
    else:
        try:
            raw_bytes = uploaded.getvalue()
        except Exception:
            _reset_loaded_file()
            st.error(f"Selected file: {uploaded.name} — could not read the file.")
            raw_bytes = None

        if raw_bytes is not None:
            #Validate file and decode the raw bytes
            text, error = process_upload(
                uploaded.name, raw_bytes, config.MAX_UPLOAD_BYTES
            )
            if error is not None:
                _invalidate_previous_doc(collection, None)
                _reset_loaded_file()
                st.error(f"Selected file: {uploaded.name} — {error}")
            else:
                assert text is not None
                doc_id = make_doc_id(uploaded.name, text)
                st.write("Current doc_id:", doc_id)
                st.write("Previous indexed_doc_id:", st.session_state.get("indexed_doc_id"))
                #If same document has already indexed: do not recompute.
                if st.session_state.get("indexed_doc_id") == doc_id:
                    st.success(f"Loaded: {uploaded.name} ({len(raw_bytes):,} bytes)")
                    st.success(
                        f"Indexed: {st.session_state.get('chunk_count', 0)} chunks "
                        "(already indexed — nothing recomputed)"
                    )
                else:
                    # Invalidate the previous document's index BEFORE the new
                    _invalidate_previous_doc(collection, doc_id)
                    try:
                        with st.spinner("Splitting and indexing the document…"):
                            chunk_count = store.index_document(
                                collection, text, uploaded.name, doc_id
                            )
                            store.remove_other_documents(collection, doc_id)
                    except Exception as exc:
                        try:
                            store.clear_document(collection, doc_id)
                        except Exception:
                            pass
                        _reset_loaded_file()
                        st.error(
                            f"Selected file: {uploaded.name} — "
                            f"indexing failed: {exc}"
                        )
                    else:
                        st.session_state["loaded_filename"] = uploaded.name
                        st.session_state["loaded_text"] = text
                        st.session_state["loaded_doc_id"] = doc_id
                        st.session_state["indexed_doc_id"] = doc_id
                        st.session_state["chunk_count"] = chunk_count
                        st.success(
                            f"Loaded: {uploaded.name} ({len(raw_bytes):,} bytes)"
                        )
                        st.success(f"Indexed: {chunk_count} chunks")

    #Check to be ready by the uploaded file to start conversation
    ready = (
        bool(st.session_state.get("loaded_text"))
        and st.session_state.get("indexed_doc_id")
        == st.session_state.get("loaded_doc_id")
        and st.session_state.get("loaded_doc_id") is not None
    )

    st.divider()
    st.subheader("گفت‌وگو")
    messages = st.session_state.get(HISTORY_KEY, [])

    for message in messages:
        with st.chat_message(message.role):
            st.write(message.content)
            #Write answer sources
            if message.role == "assistant" and message.sources:
                with st.expander("منابع"):
                    st.caption(
                        "گزیده‌هایی از سند که این پاسخ بر اساس آن‌ها نوشته شده است."
                    )
                    for source in message.sources:
                        st.caption(format_source_label(source))
                        st.write(source.text)

    if st.button("پاک کردن مکالمه", disabled=not messages):
        #Reset the chat history only: the document and its ChromaDB index stay.
        reset_conversation_state(st.session_state)

    prompt = st.chat_input(
        "سؤال خود را درباره‌ی فایل بنویسید…",
        disabled=not ready,
    )

    if prompt:
        if not ready:
            st.warning("Upload a valid .txt file before asking a question.")
        elif not prompt.strip():
            st.warning("Type a question first.")
        elif collection is None:
            st.error("Index is unavailable, so the question cannot be answered.")
        else:
            active_doc_id = st.session_state.get("loaded_doc_id")
            history = list(st.session_state.get(HISTORY_KEY, []))
            retrieval_query = rewrite_followup(prompt, history)
            try:
                with st.spinner("در حال جست‌وجو در سند…"):
                    sources = retrieve(
                        collection, retrieval_query, active_doc_id,
                        top_k=config.TOP_K,
                    )
            except ValueError:
                st.error("سند فعال معتبر نیست. لطفاً فایل را دوباره بارگذاری کنید.")
            except Exception: 
                st.error("خطا در بازیابی متن از حافظه محلی. لطفاً دوباره تلاش کنید.")
            else:
                if not sources:
                    st.warning("بخش مرتبطی در سند برای این سؤال پیدا نشد.")
                else:
                    try:
                        with st.spinner("در حال تولید پاسخ…"):
                            answer = generate_answer(prompt, sources)
                    except ValueError as exc:
                        if "AVALAI_API_KEY" in str(exc):
                            st.error(
                                "کلید AVALAI_API_KEY تنظیم نشده است. "
                                "آن را در فایل .env قرار دهید."
                            )
                        else:
                            st.error("سؤال یا متن بازیابی‌شده معتبر نیست.")
                    except RuntimeError:
                        st.error(
                            "پاسخ‌گیری از مدل ناموفق بود. اتصال اینترنت و "
                            "تنظیمات AVAL AI را بررسی کنید و دوباره تلاش کنید."
                        )
                    else:
                        st.session_state[HISTORY_KEY] = append_exchange(
                            history, prompt.strip(), answer, sources
                        )
                        st.rerun()


if __name__ == "__main__":
    main()
