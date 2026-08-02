"""Centralized configuration for the RAG module.

All settings can be overridden via environment variables.
Designed for local-first development with easy migration to cloud/OpenAI later.
"""
import os


# ── Qdrant Configuration ──────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "course_materials")

# ── Embedding Models (FastEmbed — local, no API key needed) ───────────
DENSE_EMBEDDING_MODEL = os.getenv(
    "DENSE_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-en"
)
SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"
)

# ── Chunking Configuration ────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── LLM Configuration (for query transformation & evaluation) ────────
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
# Optional fallback model served when the primary is unavailable.
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "").strip() or None
LLM_FALLBACK_BASE_URL = os.getenv("LLM_FALLBACK_BASE_URL", LLM_BASE_URL).strip()

# ── Retrieval Configuration ───────────────────────────────────────────
DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "5"))
PREFETCH_LIMIT = int(os.getenv("PREFETCH_LIMIT", "20"))

# ── Supported File Formats ────────────────────────────────────────────
SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".html", ".htm", ".md", ".markdown"}
