"""Embedding and indexing pipeline — process chunks into Qdrant points."""
import uuid
from datetime import datetime, timezone

from qdrant_client import models

from config import COLLECTION_NAME
from vector_store.qdrant_client import (
    get_qdrant_client,
    get_dense_embedder,
    get_sparse_embedder,
)
from vector_store.collection_manager import ensure_collection


def index_chunks(
    chunks: list,
    user_id: str,
    document_id: str | None = None,
    source_filename: str = "unknown",
    document_type: str = "unknown",
    collection_name: str | None = None,
) -> dict:
    """Embed chunks and upload them to Qdrant with rich metadata.

    Args:
        chunks: List of LangChain Document objects (from chunking).
        user_id: The student/user who owns this document.
        document_id: Unique ID for the document. Auto-generated if None.
        source_filename: Original filename.
        document_type: File type (pdf, docx, etc.).
        collection_name: Qdrant collection to use.

    Returns:
        dict with document_id, collection_name, and chunks_indexed count.
    """
    name = collection_name or COLLECTION_NAME
    ensure_collection(name)

    doc_id = document_id or str(uuid.uuid4())
    upload_date = datetime.now(timezone.utc).isoformat()

    client = get_qdrant_client()
    dense_embedder = get_dense_embedder()
    sparse_embedder = get_sparse_embedder()

    # Extract text from chunks
    texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings in batch
    dense_vectors = list(dense_embedder.embed(texts))
    sparse_vectors = list(sparse_embedder.embed(texts))

    # Build Qdrant points
    points = []
    for idx, (chunk, dense_vec, sparse_vec) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        sparse_indices = sparse_vec.indices.tolist()
        sparse_values = sparse_vec.values.tolist()

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": (
                    dense_vec.tolist()
                    if hasattr(dense_vec, "tolist")
                    else list(dense_vec)
                ),
                "sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            },
            payload={
                # Content
                "page_content": chunk.page_content,
                # User isolation
                "user_id": user_id,
                # Document identity
                "document_id": doc_id,
                "source_filename": source_filename,
                "document_type": document_type,
                "upload_date": upload_date,
                # Positional info
                "chunk_index": chunk.metadata.get("chunk_index", idx),
                "total_chunks": chunk.metadata.get("total_chunks", len(chunks)),
                # Original metadata from loader
                "page_number": chunk.metadata.get("page", chunk.metadata.get("page_number")),
                "original_metadata": {
                    k: str(v) for k, v in chunk.metadata.items()
                    if k not in ("chunk_index", "total_chunks")
                },
            },
        )
        points.append(point)

    # Upload in batch
    client.upload_points(
        collection_name=name,
        points=points,
        batch_size=64,
        parallel=1,
        max_retries=3,
        wait=True,
    )

    return {
        "document_id": doc_id,
        "collection_name": name,
        "chunks_indexed": len(points),
        "user_id": user_id,
    }
