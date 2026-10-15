#
from __future__ import annotations


#Just validate file to be a .txt
def validate_filename(filename: str | None) -> str | None:
    if not filename:
        return "No file selected."
    if not filename.lower().endswith(".txt"):
        return "Only .txt files are accepted."
    return None

#Validate file size to be in size limitation
def validate_file_size(num_bytes: int, max_bytes: int) -> str | None:
    if num_bytes > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return f"File is too large. Maximum size is {limit_mb:g} MB."
    return None

#Decode raw bytes as UTF-8. Returns (text, error)
def decode_text(data: bytes) -> tuple[str | None, str | None]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, "Could not decode the file as UTF-8 text."
    except Exception:
        return None, "Could not read the file content."
    if not text.strip():
        return None, "The file is empty."
    return text, None

#Do all of the file operations. Returns (text, error)
def process_upload(
    filename: str | None, data: bytes | None, max_bytes: int
) -> tuple[str | None, str | None]:
    error = validate_filename(filename)
    if error is not None:
        return None, error
    if data is None:
        return None, "Could not read the file content."
    error = validate_file_size(len(data), max_bytes)
    if error is not None:
        return None, error
    return decode_text(data)
