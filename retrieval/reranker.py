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


def _by_original_score(documents: list[dict], top_k: int) -> list[dict]:
    """Fallback ordering when the cross-encoder cannot be used."""
    return sorted(documents, key=lambda d: d.get("score", 0), reverse=True)[:top_k]


def rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank documents using a cross-encoder model.

    ``TextCrossEncoder.rerank`` returns one float per document, **in the order
    the documents were passed in** — it does not return the ranking, and it does
    not accept a top_k. The caller owns pairing each score back to its document,
    sorting, and trimming; that is what this function does.

    If the reranker is unavailable, or returns a shape we do not recognise, the
    documents come back sorted by their original retrieval score instead.

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
        return _by_original_score(documents, top_k)

    try:
        passages = [doc["content"] for doc in documents]
        scores = [float(score) for score in reranker.rerank(query, passages)]

        # A score per document is the whole contract. If that ever stops holding,
        # fail loudly here rather than silently mapping every score onto one
        # document — which is exactly how this went unnoticed before.
        if len(scores) != len(documents):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(documents)} documents"
            )

        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

        reranked = []
        for index, score in ranked[:top_k]:
            doc = documents[index].copy()
            doc["original_score"] = doc.get("score", 0)
            doc["rerank_score"] = score
            doc["score"] = score  # Replace with rerank score
            reranked.append(doc)

        return reranked

    except Exception as e:
        logger.error("Reranking failed: %s. Returning original order.", e)
        return _by_original_score(documents, top_k)
