"""Qdrant collection management — create, list, delete."""
from qdrant_client import models

from config import COLLECTION_NAME
from vector_store.qdrant_client import get_qdrant_client, get_dense_embedder


def ensure_collection(collection_name: str | None = None) -> str:
    """Create the collection if it doesn't already exist.

    Uses dual vector config: dense (cosine) + sparse (BM25).

    Returns:
        The collection name.
    """
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()

    if client.collection_exists(collection_name=name):
        return name

    # Determine vector size from the dense model
    sample = list(get_dense_embedder().embed(["sample"]))[0]
    vector_size = len(sample)

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    return name


def list_user_documents(user_id: str, collection_name: str | None = None) -> list[dict]:
    """List unique documents uploaded by a specific user.

    Returns a list of dicts with document_id, source_filename, document_type, upload_date.
    """
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()

    if not client.collection_exists(collection_name=name):
        return []

    results = client.scroll(
        collection_name=name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            ]
        ),
        limit=1000,
        with_payload=["document_id", "source_filename", "document_type", "upload_date"],
        with_vectors=False,
    )

    # Deduplicate by document_id
    seen = {}
    for point in results[0]:
        doc_id = point.payload.get("document_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "document_id": doc_id,
                "source_filename": point.payload.get("source_filename", "unknown"),
                "document_type": point.payload.get("document_type", "unknown"),
                "upload_date": point.payload.get("upload_date", "unknown"),
            }

    return list(seen.values())


def delete_document(
    user_id: str,
    document_id: str,
    collection_name: str | None = None,
) -> int:
    """Delete all chunks belonging to a specific document for a user.

    Returns the number of points deleted.
    """
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()

    # Count before delete for reporting
    before = client.count(
        collection_name=name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ]
        ),
    ).count

    client.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                ]
            )
        ),
    )

    return before


def get_collection_stats(collection_name: str | None = None) -> dict:
    """Return basic stats about the collection."""
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()

    if not client.collection_exists(collection_name=name):
        return {"exists": False, "collection_name": name}

    info = client.get_collection(collection_name=name)
    return {
        "exists": True,
        "collection_name": name,
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
        "status": str(info.status),
    }
