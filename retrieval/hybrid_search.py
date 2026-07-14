"""Hybrid search using dense + sparse vectors with RRF fusion.

Cleaned-up version of the original hyprid_search.py with:
- Shared singleton clients (no re-initialization)
- Parameterized collection name
- Per-user metadata filtering
"""
from qdrant_client import models

from config import COLLECTION_NAME, PREFETCH_LIMIT
from vector_store.qdrant_client import (
    get_qdrant_client,
    get_dense_embedder,
    get_sparse_embedder,
)


def hybrid_search_rrf(
    query_text: str,
    user_id: str | None = None,
    limit: int = 5,
    collection_name: str | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Perform hybrid search (dense + sparse) with Reciprocal Rank Fusion.

    Args:
        query_text: Natural-language search query.
        user_id: If provided, restrict results to this user's documents.
        limit: Max number of results to return.
        collection_name: Qdrant collection to search.
        filters: Optional extra filters, e.g. {"document_type": "pdf"}.

    Returns:
        List of dicts with keys: content, metadata, score, id, source_filename,
        document_id, page_number, chunk_index.
    """
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()
    dense_embedder = get_dense_embedder()
    sparse_embedder = get_sparse_embedder()

    # Generate query embeddings
    dense_query = list(dense_embedder.embed([query_text]))[0]
    sparse_query = list(sparse_embedder.embed([query_text]))[0]

    sparse_indices = sparse_query.indices.tolist()
    sparse_values = sparse_query.values.tolist()

    # Build metadata filter
    query_filter = _build_filter(user_id, filters)

    # Hybrid search with RRF fusion using prefetch
    results = client.query_points(
        collection_name=name,
        prefetch=[
            # Sparse vector prefetch (keyword/BM25 search)
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse",
                limit=PREFETCH_LIMIT,
                filter=query_filter,
            ),
            # Dense vector prefetch (semantic search)
            models.Prefetch(
                query=(
                    dense_query.tolist()
                    if hasattr(dense_query, "tolist")
                    else list(dense_query)
                ),
                using="dense",
                limit=PREFETCH_LIMIT,
                filter=query_filter,
            ),
        ],
        # RRF fusion to combine results
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit,
    )

    # Format results
    documents = []
    for point in results.points:
        doc = {
            "content": point.payload.get("page_content", ""),
            "score": point.score,
            "id": point.id,
            # Citation-relevant fields
            "source_filename": point.payload.get("source_filename", "unknown"),
            "document_id": point.payload.get("document_id", ""),
            "page_number": point.payload.get("page_number"),
            "chunk_index": point.payload.get("chunk_index"),
            "total_chunks": point.payload.get("total_chunks"),
            "document_type": point.payload.get("document_type", ""),
            "metadata": point.payload.get("original_metadata", {}),
        }
        documents.append(doc)

    return documents


def _build_filter(
    user_id: str | None, extra_filters: dict | None
) -> models.Filter | None:
    """Build a Qdrant filter from user_id and extra filter conditions."""
    conditions = []

    if user_id:
        conditions.append(
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        )

    if extra_filters:
        for key, value in extra_filters.items():
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

    if not conditions:
        return None

    return models.Filter(must=conditions)
