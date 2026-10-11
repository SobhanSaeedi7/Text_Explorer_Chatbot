"""Focused tests for text chunking and document ids."""

from pelekan_rag_chat import config
from pelekan_rag_chat.chunking import make_doc_id, split_text


def test_short_text_is_single_chunk():
    assert split_text("Hello world", chunk_size=1000, chunk_overlap=200) == [
        "Hello world"
    ]


def test_chunks_overlap_and_cover_full_text():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = split_text(text, chunk_size=10, chunk_overlap=3)
    assert len(chunks) > 1
    assert all(len(c) <= 10 for c in chunks)
    # Overlap: each chunk starts with the previous chunk's tail.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.startswith(prev[-3:])
    # Full coverage: walking the windows reproduces the text.
    assert chunks[0] + "".join(c[3:] for c in chunks[1:]) == text


def test_no_overlap_concatenates_to_original():
    text = "one two three four five six"
    chunks = split_text(text, chunk_size=7, chunk_overlap=0)
    assert "".join(chunks) == text


def test_large_overlap_is_clamped_and_terminates():
    text = "abcdefghij" * 50
    chunks = split_text(text, chunk_size=10, chunk_overlap=999)
    assert chunks  # terminates instead of looping forever
    assert all(chunks)


def test_empty_and_whitespace_input_gives_no_chunks():
    assert split_text("", chunk_size=10, chunk_overlap=2) == []
    assert split_text("   \n  ", chunk_size=10, chunk_overlap=2) == []


def test_invalid_parameters_raise():
    import pytest

    with pytest.raises(ValueError):
        split_text("hi", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        split_text("hi", chunk_size=10, chunk_overlap=-1)


def test_persian_text_is_preserved():
    original = "سلام دنیا! این یک متن آزمایشی برای بررسی تکه‌تکه کردن است. " * 20
    chunks = split_text(original, chunk_size=100, chunk_overlap=0)
    assert "".join(chunks) == original.strip()
    assert any("سلام" in c for c in chunks)


def test_config_defaults_are_sane():
    assert config.CHUNK_SIZE > config.CHUNK_OVERLAP >= 0
    assert config.CHUNK_SIZE > 0


def test_doc_id_stable_and_content_sensitive():
    assert make_doc_id("a.txt", "hello") == make_doc_id("a.txt", "hello")
    assert make_doc_id("a.txt", "hello") != make_doc_id("a.txt", "world")
    assert make_doc_id("a.txt", "hello") != make_doc_id("b.txt", "hello")
