# Pelekan RAG Chat

A small local educational project. The eventual goal: upload one `.txt`
file and ask questions about its contents using RAG (retrieval-augmented
generation), running comfortably on a laptop with ~6 GB RAM.

Current stage: **TXT upload + chunking + ChromaDB indexing**. Users can
upload one `.txt` file (max 2 MB, configurable via `MAX_UPLOAD_MB`),
with validation for extension, size, decoding, and empty content. The
text is split into overlapping chunks (configurable `CHUNK_SIZE` /
`CHUNK_OVERLAP`, defaults 1000/200) and indexed once in a persistent
local ChromaDB collection using ChromaDB's own default embedding
function. No RAG retrieval, chat memory, or LLM calls are implemented
yet.

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
│   ├── app.py              # Streamlit entry point (upload + indexing)
│   ├── document.py         # .txt validation + UTF-8 reading helpers
│   ├── chunking.py         # overlapping text splitter + document ids
│   ├── store.py            # persistent ChromaDB index (default embeddings)
│   └── config.py           # env-var placeholders for later stages
├── tests/                  # pytest tests
├── data/                   # runtime data (gitignored, local only)
├── pyproject.toml          # project + dependencies (uv)
├── Makefile                # install / run / test shortcuts
├── .env.example            # placeholder config names only, no real keys
└── README.md
```

## Notes

- Dependencies: `streamlit`, `python-dotenv`, `chromadb`, `pytest`.
  LLM libraries come in later stages.
- Runtime data and local ChromaDB storage live under `data/` and are
  excluded from Git via `.gitignore`. ChromaDB keeps collection metadata
  in a `chroma.sqlite3` file plus local vector-index segment files next
  to it (not a single SQLite file).
- The first indexing run downloads ChromaDB's default embedding model
  once (~80 MB, `all-MiniLM-L6-V2` ONNX) into the local Chroma cache;
  later runs reuse it with no download.
- Never put real API keys in the repo. Use a local `.env` (untracked).
