"""TXT upload validation and reading (no embeddings/RAG/LLM here).

Kept as small pure functions so they are easy to test without Streamlit.
"""

from __future__ import annotations


def validate_filename(filename: str | None) -> str | None:
    """Return an error message if the filename is not an accepted .txt name."""
    if not filename:
        return "No file selected."
    if not filename.lower().endswith(".txt"):
        return "Only .txt files are accepted."
    return None


def validate_file_size(num_bytes: int, max_bytes: int) -> str | None:
    """Return an error message if the file exceeds the size limit."""
    if num_bytes > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return f"File is too large. Maximum size is {limit_mb:g} MB."
    return None


def decode_text(data: bytes) -> tuple[str | None, str | None]:
    """Decode raw bytes as UTF-8. Returns (text, error)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, "Could not decode the file as UTF-8 text."
    except Exception:  # noqa: BLE001 - surface any read failure as a message
        return None, "Could not read the file content."
    if not text.strip():
        return None, "The file is empty."
    return text, None


def process_upload(
    filename: str | None, data: bytes | None, max_bytes: int
) -> tuple[str | None, str | None]:
    """Validate one uploaded file and return (text, error).

    `data` is the full in-memory content of the uploaded file.
    Only call-site I/O is `uploaded_file.getvalue()` in the UI;
    everything else here is pure validation.
    """
    error = validate_filename(filename)
    if error is not None:
        return None, error
    if data is None:
        return None, "Could not read the file content."
    error = validate_file_size(len(data), max_bytes)
    if error is not None:
        return None, error
    return decode_text(data)
