import os

from dotenv import load_dotenv

load_dotenv()


CHROMA_DIR: str = os.getenv("CHROMA_DIR", "data/chroma")


#Read environment setting as int
def _read_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


MAX_UPLOAD_MB: int = _read_int("MAX_UPLOAD_MB", 2)
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024


CHUNK_SIZE: int = _read_int("CHUNK_SIZE", 100)
CHUNK_OVERLAP: int = _read_int("CHUNK_OVERLAP", 20, minimum=0)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "pelekan_docs")

# Retrieval: how many chunks to include as answer context.
TOP_K: int = _read_int("TOP_K", 2)


AVALAI_API_KEY: str = os.getenv("AVALAI_API_KEY", "")
AVALAI_BASE_URL: str = os.getenv("AVALAI_BASE_URL", "https://api.avalai.ir/v1")
AVALAI_MODEL: str = os.getenv("AVALAI_MODEL", "gpt-6-luna")