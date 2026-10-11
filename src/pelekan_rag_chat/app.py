"""Streamlit entry point: TXT upload + chunking + ChromaDB indexing.

Scope of this stage:
- Upload exactly one .txt file (max size shown in the UI).
- Validate extension, size, decoding, and empty content.
- Split the text into overlapping chunks and index them once in a
  persistent local ChromaDB collection (default embedding function).
- Enable the question box only after the file was loaded and indexed.

RAG retrieval, chat history, and LLM calls come later.
"""

from __future__ import annotations

import streamlit as st

from pelekan_rag_chat import config
from pelekan_rag_chat.chunking import make_doc_id
from pelekan_rag_chat.document import process_upload
from pelekan_rag_chat import store


@st.cache_resource
def _get_collection():
    """Persistent ChromaDB collection, opened once per app run.

    Cached so Streamlit reruns (e.g. each question submit) reuse the
    same client instead of reopening the database every time.
    """
    return store.get_collection(store.get_client())


def _reset_loaded_file() -> None:
    """Discard any previously loaded file content and index state."""
    st.session_state.pop("loaded_filename", None)
    st.session_state.pop("loaded_text", None)
    st.session_state.pop("loaded_doc_id", None)
    st.session_state.pop("indexed_doc_id", None)
    st.session_state.pop("chunk_count", None)


def _invalidate_previous_doc(collection, new_doc_id: str | None) -> None:
    """Drop the previous document's chunks when it is replaced/removed."""
    old_doc_id = st.session_state.get("loaded_doc_id")
    if collection is not None and old_doc_id and old_doc_id != new_doc_id:
        try:
            store.clear_document(collection, old_doc_id)
        except Exception:  # noqa: BLE001 - index cleanup must not crash the UI
            pass


def main() -> None:
    st.title("Pelekan RAG Chat")
    st.write("Upload one `.txt` file, then ask a question about it.")
    st.caption(f"Maximum file size: {config.MAX_UPLOAD_MB} MB. Only `.txt` files.")
    st.caption(
        "First indexing downloads ChromaDB's default embedding model once "
        "(~80 MB, cached locally afterwards)."
    )

    try:
        collection = _get_collection()
    except Exception as exc:  # noqa: BLE001 - show DB problems in the UI
        collection = None
        st.error(f"Could not open the local index: {exc}")

    uploaded = st.file_uploader(
        "Choose a .txt file",
        type=["txt"],
        accept_multiple_files=False,
        help=f"Single .txt file up to {config.MAX_UPLOAD_MB} MB.",
    )

    if uploaded is None:
        # No file selected: invalidate the previous document's content
        # and index before showing the empty state.
        _invalidate_previous_doc(collection, None)
        _reset_loaded_file()
        st.info("No file selected yet.")
    elif collection is None:
        st.error("Index is unavailable, so the file cannot be processed.")
    else:
        try:
            raw_bytes = uploaded.getvalue()
        except Exception:  # noqa: BLE001 - show any read failure in the UI
            _reset_loaded_file()
            st.error(f"Selected file: {uploaded.name} — could not read the file.")
            raw_bytes = None

        if raw_bytes is not None:
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
                if st.session_state.get("indexed_doc_id") == doc_id:
                    # Same document already indexed: do not recompute.
                    st.success(f"Loaded: {uploaded.name} ({len(raw_bytes):,} bytes)")
                    st.success(
                        f"Indexed: {st.session_state.get('chunk_count', 0)} chunks "
                        "(already indexed — nothing recomputed)"
                    )
                else:
                    # Invalidate the previous document's index BEFORE the new
                    # document becomes active, so old chunks can never leak
                    # into the replacement's results.
                    _invalidate_previous_doc(collection, doc_id)
                    try:
                        with st.spinner("Splitting and indexing the document…"):
                            chunk_count = store.index_document(
                                collection, text, uploaded.name, doc_id
                            )
                            store.remove_other_documents(collection, doc_id)
                    except Exception as exc:
                        # noqa: BLE001 - show indexing failures in the UI
                        # The replacement failed: remove anything the failed
                        # attempt may have stored, and forget the old
                        # document too (it was invalidated above). Neither
                        # the old nor the failed new doc stays active.
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

    ready = (
        bool(st.session_state.get("loaded_text"))
        and st.session_state.get("indexed_doc_id")
        == st.session_state.get("loaded_doc_id")
        and st.session_state.get("loaded_doc_id") is not None
    )

    question = st.text_input(
        "Ask a question about the file",
        placeholder="Upload a valid .txt file first…",
        disabled=not ready,
    )
    submitted = st.button("Submit question", disabled=not ready)

    if submitted:
        if not ready:
            st.warning("Upload a valid .txt file before asking a question.")
        elif not question.strip():
            st.warning("Type a question first.")
        else:
            # RAG answering arrives in a later stage.
            st.info("Question answering is not implemented yet.")


if __name__ == "__main__":
    main()
