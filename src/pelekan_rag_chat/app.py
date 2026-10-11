"""Starter Streamlit entry point (skeleton only).

No file processing, embeddings, ChromaDB, RAG, or LLM calls yet.
This page only verifies that Python, uv, and Streamlit are wired up.
"""

import streamlit as st


def main() -> None:
    st.title("Pelekan RAG Chat")
    st.write(
        "Project skeleton is set up. "
        "File upload and RAG Q&A will be added in a later stage."
    )


if __name__ == "__main__":
    main()
