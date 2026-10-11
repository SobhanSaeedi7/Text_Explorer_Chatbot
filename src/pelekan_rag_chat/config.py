"""Minimal configuration placeholder (skeleton only).

Reads optional environment variables so later stages (embeddings,
vector store, LLM) have a single place for settings. No network
calls or model loading happens here.
"""

import os

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "data/chroma")
