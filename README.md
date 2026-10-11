# Pelekan RAG Chat

A small local educational project. The eventual goal: upload one `.txt`
file and ask questions about its contents using RAG (retrieval-augmented
generation), running comfortably on a laptop with ~6 GB RAM.

Current stage: **project skeleton and dev setup only**. No file
processing, embeddings, ChromaDB logic, RAG, chat memory, or LLM calls
are implemented yet.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for project and dependency management
- Git (no commits will be made for you)

## Setup

```bash
# 1. Install dependencies (creates .venv automatically)
make install
# or directly:
uv sync

# 2. (Optional) copy placeholder config for later stages
cp .env.example .env

# 3. Run the starter app
make run
# or directly:
uv run streamlit run src/pelekan_rag_chat/app.py

# 4. Run tests
make test
# or directly:
uv run pytest
```

## Project layout

```text
.
├── src/pelekan_rag_chat/   # application package
│   ├── __init__.py
│   ├── app.py              # minimal Streamlit entry point (skeleton)
│   └── config.py           # env-var placeholders for later stages
├── tests/                  # pytest tests
├── data/                   # runtime data (gitignored, local only)
├── pyproject.toml          # project + dependencies (uv)
├── Makefile                # install / run / test shortcuts
├── .env.example            # placeholder config names only, no real keys
└── README.md
```

## Notes

- Dependencies are intentionally minimal: `streamlit`, `python-dotenv`,
  `pytest`. Vector-store / LLM libraries come in later stages.
- Runtime data and local ChromaDB storage live under `data/` and are
  excluded from Git via `.gitignore`.
- Never put real API keys in the repo. Use a local `.env` (untracked).
