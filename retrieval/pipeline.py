"""Full retrieval pipeline — orchestrates hybrid search, query transformation, and reranking.

This is the main entry point for retrieval. It chains:
  1. (Optional) Query transformation → multiple sub-queries
  2. Hybrid search (dense + sparse + RRF) per sub-query
  3. Merge & deduplicate results
  4. (Optional) Cross-encoder reranking
  5. Format results with citations
"""
import logging

from config import DEFAULT_SEARCH_LIMIT
from document_processing.metadata import (
    PAYLOAD_BOOK_TITLE,
    PAYLOAD_COLLECTION_ID,
    PAYLOAD_DOCUMENT_ID,
)
from retrieval.hybrid_search import hybrid_search_rrf
from retrieval.reranker import rerank
from retrieval.query_transform import decompose_query

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    user_id: str | None = None,
    limit: int | None = None,
    use_reranking: bool = True,
    use_query_transform: bool = False,
    collection_name: str | None = None,
    filters: dict | None = None,
    collection_id: str | None = None,
    document_ids: list[str] | None = None,
    book_titles: list[str] | None = None,
) -> list[dict]:
    """Full retrieval pipeline with all advanced features.

    Args:
        query: Natural-language search query.
        user_id: Restrict results to this user's documents.
        limit: Max results to return. Defaults to DEFAULT_SEARCH_LIMIT.
        use_reranking: Whether to apply cross-encoder reranking.
        use_query_transform: Whether to decompose the query into sub-queries.
        collection_name: Qdrant collection to search.
        filters: Extra metadata filters (e.g., {"document_type": "pdf"}).
        collection_id: Restrict to one logical multi-book collection.
        document_ids: Restrict to specific documents (source isolation).
        book_titles: Restrict to specific books by title.

    Returns:
        List of document dicts with content, scores, and citation info.
    """
    limit = limit or DEFAULT_SEARCH_LIMIT
    filter_variants = build_source_filters(
        filters,
        collection_id=collection_id,
        document_ids=document_ids,
        book_titles=book_titles,
    )

    # Step 1: Query transformation
    if use_query_transform:
        queries = decompose_query(query)
        logger.info("Query decomposed into %d sub-queries: %s", len(queries), queries)
    else:
        queries = [query]

    # Step 2: Hybrid search for each sub-query, once per source filter variant.
    # Fetch more results than needed so reranking has room to work
    fetch_limit = limit * 3 if use_reranking else limit
    all_results = []

    for q in queries:
        for variant in filter_variants:
            results = hybrid_search_rrf(
                query_text=q,
                user_id=user_id,
                limit=fetch_limit,
                collection_name=collection_name,
                filters=variant,
            )
            all_results.extend(results)

    # Step 3: Deduplicate by chunk ID
    unique_results = _deduplicate(all_results)
    logger.info("Retrieved %d unique chunks from %d total", len(unique_results), len(all_results))

    # Step 4: Reranking
    if use_reranking and len(unique_results) > 1:
        unique_results = rerank(query, unique_results, top_k=limit)
        logger.info("Reranked to top %d results", len(unique_results))
    else:
        # Sort by score and trim
        unique_results = sorted(
            unique_results, key=lambda d: d.get("score", 0), reverse=True
        )[:limit]

    # Step 5: Promote citation identity and format the citation line
    for i, doc in enumerate(unique_results):
        _promote_citation_fields(doc)
        doc["citation"] = _format_citation(doc, rank=i + 1)

    return unique_results


def build_source_filters(
    filters: dict | None,
    *,
    collection_id: str | None = None,
    document_ids: list[str] | None = None,
    book_titles: list[str] | None = None,
) -> list[dict | None]:
    """Expand multi-book scoping into concrete filter variants.

    ``hybrid_search_rrf`` matches one value per payload key, so restricting to
    several documents means one search per document. Returns the variants to
    search; results are merged and deduplicated by the caller.
    """
    base = dict(filters or {})
    if collection_id:
        base[PAYLOAD_COLLECTION_ID] = collection_id

    if document_ids:
        return [{**base, PAYLOAD_DOCUMENT_ID: value} for value in document_ids]
    if book_titles:
        return [{**base, PAYLOAD_BOOK_TITLE: value} for value in book_titles]
    return [base or None]


def _promote_citation_fields(doc: dict) -> dict:
    """Lift collection/book/section out of the nested payload onto the hit.

    ``vector_store.indexing`` nests everything it does not promote itself under
    ``original_metadata`` (stringified). Citations need those fields at the top
    level, so they are copied up here rather than every consumer knowing the
    payload layout.
    """
    nested = doc.get("metadata") or {}
    if not isinstance(nested, dict):
        return doc

    for key in ("collection_id", "book_title", "section", "document_id"):
        value = nested.get(key)
        if value not in (None, "", "None") and not doc.get(key):
            doc[key] = value

    if doc.get("page_number") in (None, ""):
        page = nested.get("page_number") or nested.get("page")
        if isinstance(page, str) and page.strip().isdigit():
            doc["page_number"] = int(page)
        elif isinstance(page, int):
            doc["page_number"] = page

    estimated = nested.get("page_is_estimated")
    if isinstance(estimated, str):
        doc["page_is_estimated"] = estimated.strip().lower() == "true"
    elif isinstance(estimated, bool):
        doc["page_is_estimated"] = estimated

    return doc


def retrieve_formatted(
    query: str,
    user_id: str | None = None,
    limit: int | None = None,
    use_reranking: bool = True,
    use_query_transform: bool = False,
    collection_name: str | None = None,
    filters: dict | None = None,
    collection_id: str | None = None,
    document_ids: list[str] | None = None,
    book_titles: list[str] | None = None,
) -> str:
    """Retrieve and format results as a string for LLM consumption.

    Same parameters as retrieve(). Returns a formatted string.
    """
    docs = retrieve(
        query=query,
        user_id=user_id,
        limit=limit,
        use_reranking=use_reranking,
        use_query_transform=use_query_transform,
        collection_name=collection_name,
        filters=filters,
        collection_id=collection_id,
        document_ids=document_ids,
        book_titles=book_titles,
    )

    if not docs:
        return "No relevant documents found."

    parts = []
    for doc in docs:
        parts.append(
            f"{doc['citation']}\n"
            f"Content: {doc['content']}"
        )

    return "\n\n---\n\n".join(parts)


def _deduplicate(documents: list[dict]) -> list[dict]:
    """Remove duplicate chunks, keeping the one with the highest score."""
    seen = {}
    for doc in documents:
        doc_id = doc.get("id")
        if doc_id not in seen or doc.get("score", 0) > seen[doc_id].get("score", 0):
            seen[doc_id] = doc
    return list(seen.values())


def _format_citation(doc: dict, rank: int) -> str:
    """Format a citation string for a retrieved document."""
    source = doc.get("book_title") or doc.get("source_filename", "Unknown source")
    page = doc.get("page_number")
    section = doc.get("section")
    score = doc.get("score", 0)
    chunk_idx = doc.get("chunk_index", "?")
    total = doc.get("total_chunks", "?")

    page_str = f" | Page: {page}" if page is not None else ""
    section_str = f" | Section: {section}" if section else ""
    return (
        f"[{rank}] Source: {source}{page_str}{section_str} | "
        f"Chunk: {chunk_idx}/{total} | Score: {score:.4f}"
    )
