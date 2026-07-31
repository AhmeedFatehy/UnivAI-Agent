"""Qdrant collection management — create, list, delete."""
import logging

from qdrant_client import models

from config import COLLECTION_NAME
from document_processing.metadata import (
    INDEXED_PAYLOAD_KEYS,
    PAYLOAD_COLLECTION_ID,
)
from vector_store.qdrant_client import get_qdrant_client, get_dense_embedder

logger = logging.getLogger(__name__)


def ensure_collection(collection_name: str | None = None) -> str:
    """Create the collection if it doesn't already exist.

    Uses dual vector config: dense (cosine) + sparse (BM25).

    Returns:
        The collection name.
    """
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()

    if client.collection_exists(collection_name=name):
        ensure_payload_indexes(name)
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
    ensure_payload_indexes(name)
    return name


def ensure_payload_indexes(collection_name: str | None = None) -> list[str]:
    """Index the payload keys we filter on: user, collection, document, book.

    Without these, multi-book filtering degrades to a full scan. Creating an
    index that already exists is a no-op on the Qdrant side, so this is safe to
    call on every ``ensure_collection``.
    """
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()
    created = []

    for key in INDEXED_PAYLOAD_KEYS:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=key,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            created.append(key)
        except Exception as error:  # noqa: BLE001 — already-indexed is the common case
            logger.debug("payload index for '%s' not created: %s", key, error)

    return created


def list_collection_documents(
    user_id: str,
    collection_id: str,
    collection_name: str | None = None,
) -> list[dict]:
    """List the books a user has indexed under one logical collection."""
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()

    if not client.collection_exists(collection_name=name):
        return []

    points, _ = client.scroll(
        collection_name=name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(
                    key=PAYLOAD_COLLECTION_ID,
                    match=models.MatchValue(value=collection_id),
                ),
            ]
        ),
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    seen: dict[str, dict] = {}
    for point in points:
        payload = point.payload or {}
        nested = payload.get("original_metadata") or {}
        document_id = payload.get("document_id")
        if not document_id:
            continue
        entry = seen.setdefault(
            document_id,
            {
                "document_id": document_id,
                "collection_id": collection_id,
                "book_title": nested.get("book_title", payload.get("source_filename", "unknown")),
                "source_filename": payload.get("source_filename", "unknown"),
                "document_type": payload.get("document_type", "unknown"),
                "upload_date": payload.get("upload_date", "unknown"),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1

    return sorted(seen.values(), key=lambda item: item["book_title"])


def fetch_document_chunks(
    user_id: str,
    document_id: str,
    chunk_index: int | None = None,
    collection_name: str | None = None,
    limit: int = 25,
) -> list[dict]:
    """Fetch indexed chunks for one document, so a citation can be verified.

    Returns flattened dicts in the same shape ``retrieval.hybrid_search``
    produces, which is what ``document_processing.metadata.citation_from_payload``
    expects.
    """
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()

    if not client.collection_exists(collection_name=name):
        return []

    conditions = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
        models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
    ]
    if chunk_index is not None:
        conditions.append(
            models.FieldCondition(
                key="chunk_index", match=models.MatchValue(value=chunk_index)
            )
        )

    points, _ = client.scroll(
        collection_name=name,
        scroll_filter=models.Filter(must=conditions),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    rows = []
    for point in points:
        payload = point.payload or {}
        rows.append(
            {
                "id": point.id,
                "content": payload.get("page_content", ""),
                "document_id": payload.get("document_id", ""),
                "source_filename": payload.get("source_filename", ""),
                "document_type": payload.get("document_type", ""),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "total_chunks": payload.get("total_chunks"),
                "metadata": payload.get("original_metadata", {}),
            }
        )

    return sorted(rows, key=lambda row: (row["chunk_index"] is None, row["chunk_index"]))


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


def delete_document_versions(
    user_id: str,
    document_id: str,
    preserve_ingestion_id: str,
    collection_name: str | None = None,
) -> int:
    """Delete older generations while preserving a newly indexed replacement."""
    name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()
    superseded = models.Filter(
        must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(
                key="document_id", match=models.MatchValue(value=document_id)
            ),
        ],
        must_not=[
            models.FieldCondition(
                key="original_metadata.ingestion_id",
                match=models.MatchValue(value=preserve_ingestion_id),
            )
        ],
    )
    before = client.count(collection_name=name, count_filter=superseded).count
    if before:
        client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(filter=superseded),
            wait=True,
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
        # CollectionInfo has no `vectors_count`; the count of vectors the index
        # has actually built is `indexed_vectors_count`, and it is None while a
        # freshly written collection is still indexing.
        "indexed_vectors_count": info.indexed_vectors_count,
        "segments_count": info.segments_count,
        "status": str(info.status),
    }
