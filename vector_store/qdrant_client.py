"""Shared Qdrant client and embedding model singletons.

Avoids re-initializing heavy resources on every function call.
"""
from functools import lru_cache

from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding

from config import QDRANT_URL, DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client."""
    return QdrantClient(url=QDRANT_URL)


@lru_cache(maxsize=1)
def get_dense_embedder() -> TextEmbedding:
    """Return a singleton dense embedding model."""
    return TextEmbedding(model_name=DENSE_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_sparse_embedder() -> SparseTextEmbedding:
    """Return a singleton sparse embedding model (BM25)."""
    return SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)
