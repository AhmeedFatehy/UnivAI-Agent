"""Cross-encoder reranking for search results.

Uses fastembed's TextCrossEncoder to re-score and reorder
hybrid search results for better precision.
"""
import logging

from config import RERANKER_MODEL

logger = logging.getLogger(__name__)

# Lazy-loaded reranker singleton
_reranker = None
_reranker_available = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _reranker, _reranker_available

    if _reranker_available is not None:
        return _reranker

    try:
        from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
        _reranker_available = True
        logger.info("Reranker loaded: %s", RERANKER_MODEL)
    except Exception as e:
        _reranker_available = False
        logger.warning("Reranker not available (%s). Falling back to score-based ordering.", e)

    return _reranker


def rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank documents using a cross-encoder model.

    If the reranker is unavailable, returns documents sorted by original score.

    Args:
        query: The search query.
        documents: List of document dicts (must have a "content" key).
        top_k: Number of top results to return after reranking.

    Returns:
        Reranked list of document dicts, trimmed to top_k.
    """
    if not documents:
        return []

    reranker = _get_reranker()

    if reranker is None:
        # Fallback: just return top_k by original score
        return sorted(documents, key=lambda d: d.get("score", 0), reverse=True)[:top_k]

    try:
        passages = [doc["content"] for doc in documents]
        rerank_results = list(reranker.rerank(query, passages, top_k=top_k))

        # rerank_results are objects/dicts with 'score' and 'index' (or 'doc_index')
        reranked = []
        for result in rerank_results:
            # Handle both dict-style and object-style results
            if isinstance(result, dict):
                idx = result.get("index", result.get("doc_index", 0))
                score = result.get("score", 0)
            else:
                idx = getattr(result, "index", getattr(result, "doc_index", 0))
                score = getattr(result, "score", 0)

            doc = documents[idx].copy()
            doc["rerank_score"] = score
            doc["original_score"] = doc.get("score", 0)
            doc["score"] = score  # Replace with rerank score
            reranked.append(doc)

        return reranked

    except Exception as e:
        logger.error("Reranking failed: %s. Returning original order.", e)
        return sorted(documents, key=lambda d: d.get("score", 0), reverse=True)[:top_k]
