"""Vector store module — Qdrant client, collection management, and indexing."""

from vector_store.qdrant_client import get_qdrant_client, get_dense_embedder, get_sparse_embedder
from vector_store.collection_manager import ensure_collection, list_user_documents, delete_document
from vector_store.indexing import index_chunks

__all__ = [
    "get_qdrant_client", "get_dense_embedder", "get_sparse_embedder",
    "ensure_collection", "list_user_documents", "delete_document",
    "index_chunks",
]
