"""Embedding and indexing pipeline — process chunks into Qdrant points."""
import uuid
from datetime import datetime, timezone

from qdrant_client import models

from config import COLLECTION_NAME, EMBEDDING_BATCH_SIZE, QDRANT_UPLOAD_BATCH_SIZE
from vector_store.qdrant_client import (
    get_qdrant_client,
    get_dense_embedder,
    get_sparse_embedder,
)
from vector_store.collection_manager import ensure_collection


def _as_float_list(vector) -> list[float]:
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


def index_chunks(
    chunks: list,
    user_id: str,
    document_id: str | None = None,
    source_filename: str = "unknown",
    document_type: str = "unknown",
    collection_name: str | None = None,
    *,
    content_hash: str | None = None,
    artifact_key: str | None = None,
    embeddings: list[tuple[list[float] | None, list[int] | None, list[float] | None]] | None = None,
) -> dict:
    """Embed chunks and upload them to Qdrant with rich metadata.

    Args:
        chunks: List of LangChain Document objects (from chunking).
        user_id: The student/user who owns this document.
        document_id: Unique ID for the document. Auto-generated if None.
        source_filename: Original filename.
        document_type: File type (pdf, docx, etc.).
        collection_name: Qdrant collection to use.
        content_hash: Server-computed SHA-256 of the source bytes, stored for
            provenance when the chunk came from a cached artifact.
        artifact_key: The content-addressed artifact key the chunk belongs to,
            stored for provenance when the chunk came from a cached artifact.
        embeddings: Optional per-chunk ``(dense, sparse_indices, sparse_values)``
            triples, so a cached artifact's vectors are reused instead of being
            re-embedded.

    Returns:
        dict with document_id, collection_name, and chunks_indexed count.
    """
    name = collection_name or COLLECTION_NAME
    ensure_collection(name)

    doc_id = document_id or str(uuid.uuid4())
    indexing_run_id = str(uuid.uuid4())
    upload_date = datetime.now(timezone.utc).isoformat()

    client = get_qdrant_client()

    cached = embeddings if embeddings is not None and len(embeddings) == len(chunks) else None

    # Embed and upload bounded slices.  Besides controlling peak ONNX memory,
    # the slice boundary lets the OS fairly schedule another tenant's worker.
    try:
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_chunks = chunks[start : start + EMBEDDING_BATCH_SIZE]
            if cached is not None:
                batch_embeddings = cached[start : start + len(batch_chunks)]
                dense_vectors = [entry[0] for entry in batch_embeddings]
                sparse_parts = [(entry[1], entry[2]) for entry in batch_embeddings]
            else:
                texts = [chunk.page_content for chunk in batch_chunks]
                dense_vectors = [
                    _as_float_list(vector) for vector in get_dense_embedder().embed(texts)
                ]
                sparse_parts = [
                    (sparse.indices.tolist(), sparse.values.tolist())
                    for sparse in get_sparse_embedder().embed(texts)
                ]

            points = []
            for offset, (chunk, dense_vec, (sparse_indices, sparse_values)) in enumerate(
                zip(batch_chunks, dense_vectors, sparse_parts)
            ):
                idx = start + offset
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vec if dense_vec is not None else _as_float_list(
                                list(get_dense_embedder().embed([chunk.page_content]))[0]
                            ),
                            "sparse": models.SparseVector(
                                indices=sparse_indices or [],
                                values=sparse_values or [],
                            ),
                        },
                        payload={
                            "page_content": chunk.page_content,
                            "user_id": user_id,
                            "document_id": doc_id,
                            "indexing_run_id": indexing_run_id,
                            "source_filename": source_filename,
                            "document_type": document_type,
                            "upload_date": upload_date,
                            "content_hash": content_hash,
                            "artifact_key": artifact_key,
                            "chunk_index": chunk.metadata.get("chunk_index", idx),
                            "total_chunks": chunk.metadata.get("total_chunks", len(chunks)),
                            "page_number": chunk.metadata.get(
                                "page", chunk.metadata.get("page_number")
                            ),
                            "original_metadata": {
                                k: str(v)
                                for k, v in chunk.metadata.items()
                                if k not in ("chunk_index", "total_chunks")
                            },
                        },
                    )
                )

            client.upload_points(
                collection_name=name,
                points=points,
                batch_size=min(QDRANT_UPLOAD_BATCH_SIZE, len(points)),
                parallel=1,
                max_retries=3,
                wait=True,
            )
    except Exception:
        # Only remove points written by this invocation. A stable document ID
        # may already have an older known-good generation that must survive.
        client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="indexing_run_id",
                            match=models.MatchValue(value=indexing_run_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        raise

    return {
        "document_id": doc_id,
        "collection_name": name,
        "chunks_indexed": len(chunks),
        "user_id": user_id,
    }
