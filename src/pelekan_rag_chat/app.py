"""Streamlit entry point: TXT upload stage.

Scope of this stage:
- Upload exactly one .txt file (max size shown in the UI).
- Validate extension, size, decoding, and empty content.
- Enable the question box only after a valid file was read.

Embeddings, ChromaDB, RAG, chat history, and LLM calls come later.
"""

from __future__ import annotations

import streamlit as st

from pelekan_rag_chat import config
from pelekan_rag_chat.document import process_upload


def _reset_loaded_file() -> None:
    """Discard any previously loaded file content."""
    st.session_state.pop("loaded_filename", None)
    st.session_state.pop("loaded_text", None)


def main() -> None:
    st.title("Pelekan RAG Chat")
    st.write("Upload one `.txt` file, then ask a question about it.")
    st.caption(f"Maximum file size: {config.MAX_UPLOAD_MB} MB. Only `.txt` files.")

    uploaded = st.file_uploader(
        "Choose a .txt file",
        type=["txt"],
        accept_multiple_files=False,
        help=f"Single .txt file up to {config.MAX_UPLOAD_MB} MB.",
    )

    if uploaded is None:
        # No file selected: make sure old content is discarded.
        _reset_loaded_file()
        st.info("No file selected yet.")
    else:
        try:
            raw_bytes = uploaded.getvalue()
        except Exception:  # noqa: BLE001 - show any read failure in the UI
            _reset_loaded_file()
            st.error(f"Selected file: {uploaded.name} — could not read the file.")
            raw_bytes = None

        if raw_bytes is not None:
            # Replacing the file discards the previous in-memory content:
            # only the latest upload is kept in session state.
            if st.session_state.get("loaded_filename") != uploaded.name:
                _reset_loaded_file()
            text, error = process_upload(
                uploaded.name, raw_bytes, config.MAX_UPLOAD_BYTES
            )
            if error is not None:
                _reset_loaded_file()
                st.error(f"Selected file: {uploaded.name} — {error}")
            else:
                st.session_state["loaded_filename"] = uploaded.name
                st.session_state["loaded_text"] = text
                st.success(f"Loaded: {uploaded.name} ({len(raw_bytes):,} bytes)")

    ready = bool(st.session_state.get("loaded_text"))

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
