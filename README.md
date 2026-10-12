# Pelekan RAG Chat

A small local educational project. The eventual goal: upload one `.txt`
file and ask questions about its contents using RAG (retrieval-augmented
generation), running comfortably on a laptop with ~6 GB RAM.

Current stage: **RAG chat with temporary memory**. Upload one `.txt`
file (max 2 MB, `MAX_UPLOAD_MB`), which is split into overlapping chunks
(`CHUNK_SIZE` / `CHUNK_OVERLAP`, defaults 1000/200) and indexed once in
a persistent local ChromaDB collection using ChromaDB's own default
embedding function. Each question retrieves the top matching chunks
(`TOP_K`, default 4) from the active document only, then AVAL AI
generates a grounded Persian answer shown together with its source
excerpts. Follow-up questions are rewritten into standalone retrieval
queries using the recent in-session history; the conversation lives
only in Streamlit session state and is never persisted.

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
│   ├── retrieval.py        # top-k chunk search, active document only
│   ├── llm.py              # grounded Persian answers via AVAL AI
│   ├── conversation.py     # temporary chat memory + follow-up rewriting
│   └── config.py           # environment-based settings (see below)
├── tests/                  # pytest tests
├── data/                   # runtime data (gitignored, local only)
├── pyproject.toml          # project + dependencies (uv)
├── Makefile                # install / run / test shortcuts
├── .env.example            # placeholder config names only, no real keys
└── README.md
```

## Configuration

Copy `.env.example` to `.env` and set at least `AVALAI_API_KEY`:

```text
AVALAI_API_KEY=            # required for answering questions
AVALAI_BASE_URL=https://api.avalai.ir/v1
AVALAI_MODEL=gpt-6-luna
TOP_K=4                    # retrieved chunks per question
```

Uploading and indexing work fully offline. Answering questions additionally
requires the API key and a network connection to AVAL AI.

## Conversation memory (temporary)

- Questions and answers are kept in Streamlit session state and shown as
  a chat. History is never written to disk, ChromaDB, or any database,
  and it disappears when the session ends.
- A follow-up such as «نتیجه چه شد؟» is rewritten into a standalone
  retrieval query using the last few exchanges; the original question is
  kept for display and answering, and the rewritten query is used only
  for retrieval. With no history, the question is used as-is with no
  extra model call. If rewriting fails, the original question is used.
- «پاک کردن مکالمه» clears the chat only; the uploaded document and its
  index are kept. Uploading a different file (or removing it) clears the
  chat too, so a new document never sees old history or chunks.

## Privacy

- Local: the document, its chunks, embeddings, and the conversation
  history stay on your machine (ChromaDB default embedding model, no
  external service; history in session memory only).
- External: when you ask a question, the retrieved excerpts *and* your
  question are sent to AVAL AI to generate the answer. When a follow-up
  needs rewriting, the recent conversation context is also sent to
  AVAL AI for that rewrite. Do not upload sensitive documents unless you
  accept this.

## Prompt-safety note

The answer prompt instructs the model to use only the retrieved excerpts,
to say «اطلاعات کافی در سند موجود نیست.» when unsupported, and to treat
excerpts as untrusted data (ignoring instructions hidden inside the
document). This reduces but does not eliminate prompt-injection risk:
a sufficiently insistent document can still influence the model, so
review answers against the shown sources.

## Notes

- Dependencies: `streamlit`, `python-dotenv`, `chromadb`,
  `langchain-openai` (AVAL AI chat), `pytest`.
- Runtime data and local ChromaDB storage live under `data/` and are
  excluded from Git via `.gitignore`. ChromaDB keeps collection metadata
  in a `chroma.sqlite3` file plus local vector-index segment files next
  to it (not a single SQLite file).
- The first indexing run downloads ChromaDB's default embedding model
  once (~80 MB, `all-MiniLM-L6-V2` ONNX) into the local Chroma cache;
  later runs reuse it with no download.
- Never put real API keys in the repo. Use a local `.env` (untracked).
