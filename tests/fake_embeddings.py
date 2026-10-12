"""Shared lightweight embedding stand-in for ordinary unit tests.

Hash-based deterministic vectors: no downloads, no model, no network.
Implements the protocol the installed ChromaDB version expects
(`__call__` for documents, `embed_query` for queries, `name`).
"""

import hashlib


class HashEmbeddingFunction:
    """Deterministic 8-dim sha256 vectors (tests only, never production)."""

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def embed_query(self, input):
        return [self._embed(text) for text in input]

    def name(self):
        return "test_hash_embedding"

    @staticmethod
    def _embed(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[:8]]
