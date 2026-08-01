"""Query transformation — decompose complex queries into sub-queries.

This is the Advanced RAG feature (satisfies the "minimum 1" requirement).
Uses an LLM to break a complex user question into simpler sub-queries,
then merges results from all sub-queries.
"""
import logging
import json

from config import LLM_MODEL
from agents.prompts import PromptOperation, load_prompt_for

logger = logging.getLogger(__name__)

# Lazy-loaded LLM
_llm = None


def _get_llm():
    """Lazy-load the Ollama LLM for query transformation."""
    global _llm
    if _llm is not None:
        return _llm

    try:
        from langchain_ollama import ChatOllama
        _llm = ChatOllama(model=LLM_MODEL, temperature=0)
        logger.info("Query transform LLM loaded: %s", LLM_MODEL)
        return _llm
    except Exception as e:
        logger.warning("LLM not available for query transformation: %s", e)
        return None


def decompose_query(query: str) -> list[str]:
    """Decompose a complex query into simpler sub-queries using an LLM.

    If the LLM is unavailable or the query is simple, returns [query] unchanged.

    Args:
        query: The original user query.

    Returns:
        List of sub-queries (always includes the original query).
    """
    llm = _get_llm()
    if llm is None:
        return [query]

    prompt = load_prompt_for(PromptOperation.RETRIEVAL_DECOMPOSE).render(query=query)

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Try to extract JSON array from response
        # Handle cases where LLM wraps in markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        sub_queries = json.loads(content)

        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            # Always include the original query
            if query not in sub_queries:
                sub_queries.insert(0, query)
            logger.info("Decomposed into %d sub-queries", len(sub_queries))
            return sub_queries

    except Exception as e:
        logger.warning("Query decomposition failed: %s. Using original query.", e)

    return [query]


def expand_query(query: str) -> str:
    """Expand a query with related terms and synonyms using an LLM.

    Args:
        query: The original user query.

    Returns:
        An expanded version of the query.
    """
    llm = _get_llm()
    if llm is None:
        return query

    prompt = load_prompt_for(PromptOperation.RETRIEVAL_EXPAND).render(query=query)

    try:
        response = llm.invoke(prompt)
        expanded = response.content.strip()
        if expanded and len(expanded) < 500:
            return expanded
    except Exception as e:
        logger.warning("Query expansion failed: %s. Using original query.", e)

    return query
