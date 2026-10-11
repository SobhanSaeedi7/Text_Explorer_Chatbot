"""Focused tests for TXT upload validation and reading."""

from pelekan_rag_chat import config
from pelekan_rag_chat.document import (
    decode_text,
    process_upload,
    validate_filename,
    validate_file_size,
)


def test_accepts_txt_extension_case_insensitive():
    assert validate_filename("notes.txt") is None
    assert validate_filename("NOTES.TXT") is None


def test_rejects_non_txt_extension():
    assert validate_filename("notes.pdf") == "Only .txt files are accepted."
    assert validate_filename("notes") == "Only .txt files are accepted."
    assert validate_filename("") == "No file selected."
    assert validate_filename(None) == "No file selected."


def test_rejects_oversize_file():
    assert (
        validate_file_size(config.MAX_UPLOAD_BYTES + 1, config.MAX_UPLOAD_BYTES)
        is not None
    )
    assert (
        validate_file_size(config.MAX_UPLOAD_BYTES, config.MAX_UPLOAD_BYTES) is None
    )


def test_reads_utf8_english_text():
    text, error = decode_text("Hello world\nSecond line".encode("utf-8"))
    assert error is None
    assert text == "Hello world\nSecond line"


def test_reads_utf8_persian_text():
    original = "سلام دنیا\nاین یک فایل آزمایشی است."
    text, error = decode_text(original.encode("utf-8"))
    assert error is None
    assert text == original


def test_rejects_empty_and_whitespace_only_content():
    _, error = decode_text(b"")
    assert error == "The file is empty."
    _, error = decode_text("   \n  ".encode("utf-8"))
    assert error == "The file is empty."


def test_rejects_non_utf8_bytes():
    _, error = decode_text(b"\xff\xfe\x00 invalid utf-8 \x80")
    assert error == "Could not decode the file as UTF-8 text."


def test_process_upload_success_and_replacement():
    # First file loads fine.
    text, error = process_upload("first.txt", "hello".encode("utf-8"), 1024)
    assert error is None and text == "hello"
    # A different valid file simply produces its own text; the caller
    # (Streamlit session state) keeps only the latest one.
    text2, error2 = process_upload("second.txt", "world".encode("utf-8"), 1024)
    assert error2 is None and text2 == "world"


def test_process_upload_surfaces_each_failure():
    _, error = process_upload("notes.pdf", b"hello", 1024)
    assert error == "Only .txt files are accepted."
    _, error = process_upload("big.txt", b"x" * 1025, 1024)
    assert "Maximum size" in error
    _, error = process_upload("empty.txt", b"  ", 1024)
    assert error == "The file is empty."
    _, error = process_upload("broken.txt", b"\xff\xfe\x80", 1024)
    assert error == "Could not decode the file as UTF-8 text."
    _, error = process_upload("missing.txt", None, 1024)
    assert error == "Could not read the file content."
