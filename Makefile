.PHONY: install run test

install:
	uv sync

run:
	uv run streamlit run src/pelekan_rag_chat/app.py

test:
	uv run pytest
