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

    Returns:
        List of document dicts with content, scores, and citation info.
    """
    limit = limit or DEFAULT_SEARCH_LIMIT

    # Step 1: Query transformation
    if use_query_transform:
        queries = decompose_query(query)
        logger.info("Query decomposed into %d sub-queries: %s", len(queries), queries)
    else:
        queries = [query]

    # Step 2: Hybrid search for each sub-query
    # Fetch more results than needed so reranking has room to work
    fetch_limit = limit * 3 if use_reranking else limit
    all_results = []

    for q in queries:
        results = hybrid_search_rrf(
            query_text=q,
            user_id=user_id,
            limit=fetch_limit,
            collection_name=collection_name,
            filters=filters,
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

    # Step 5: Add citation formatting
    for i, doc in enumerate(unique_results):
        doc["citation"] = _format_citation(doc, rank=i + 1)

    return unique_results


def retrieve_formatted(
    query: str,
    user_id: str | None = None,
    limit: int | None = None,
    use_reranking: bool = True,
    use_query_transform: bool = False,
    collection_name: str | None = None,
    filters: dict | None = None,
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
    source = doc.get("source_filename", "Unknown source")
    page = doc.get("page_number")
    score = doc.get("score", 0)
    chunk_idx = doc.get("chunk_index", "?")
    total = doc.get("total_chunks", "?")

    page_str = f" | Page: {page}" if page is not None else ""
    return f"[{rank}] Source: {source}{page_str} | Chunk: {chunk_idx}/{total} | Score: {score:.4f}"
