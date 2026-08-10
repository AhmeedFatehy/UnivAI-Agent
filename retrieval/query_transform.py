"""Query transformation — decompose complex queries into sub-queries.

This is the Advanced RAG feature (satisfies the "minimum 1" requirement).
Uses an LLM to break a complex user question into simpler sub-queries,
then merges results from all sub-queries.
"""
import logging
import json
import re

from config import LLM_MODEL
from agents.prompts import PromptOperation, load_prompt_for
from guardrails.input import classify_user_input
from guardrails.prompt_boundary import split_prompt_roles

logger = logging.getLogger(__name__)

MAX_TRANSFORMED_QUERY_CHARS = 500
MAX_DECOMPOSED_QUERIES = 4


def _invoke_prompt(llm, prompt: str):
    roles = split_prompt_roles(prompt)
    request = (
        [("system", roles[0]), ("human", roles[1])]
        if roles is not None
        else prompt
    )
    return llm.invoke(request)


def _safe_generated_query(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > MAX_TRANSFORMED_QUERY_CHARS:
        return None
    if any(ord(char) < 32 and char not in "\t\n\r" for char in text):
        return None
    if not classify_user_input(text).safe:
        return None
    return text


def _shares_scope(original: str, candidate: str) -> bool:
    """Require at least one original content token in model-expanded search text."""
    source = set(re.findall(r"[a-z0-9]{2,}", original.casefold()))
    expanded = set(re.findall(r"[a-z0-9]{2,}", candidate.casefold()))
    return not source or bool(source & expanded)

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
        logger.warning(
            "LLM not available for query transformation (%s)", type(e).__name__
        )
        return None


def decompose_query(query: str) -> list[str]:
    """Decompose a complex query into simpler sub-queries using an LLM.

    If the LLM is unavailable or the query is simple, returns [query] unchanged.

    Args:
        query: The original user query.

    Returns:
        List of sub-queries (always includes the original query).
    """
    if not classify_user_input(query).safe:
        return [query]

    llm = _get_llm()
    if llm is None:
        return [query]

    prompt = load_prompt_for(PromptOperation.RETRIEVAL_DECOMPOSE).render(query=query)

    try:
        response = _invoke_prompt(llm, prompt)
        content = response.content.strip()

        # Try to extract JSON array from response
        # Handle cases where LLM wraps in markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        sub_queries = json.loads(content)

        if isinstance(sub_queries, list):
            safe: list[str] = [query]
            for candidate in sub_queries:
                transformed = _safe_generated_query(candidate)
                if (
                    transformed
                    and transformed not in safe
                    and _shares_scope(query, transformed)
                ):
                    safe.append(transformed)
                if len(safe) >= MAX_DECOMPOSED_QUERIES:
                    break
            logger.info("Decomposed into %d bounded sub-queries", len(safe))
            return safe

    except Exception as e:
        logger.warning(
            "Query decomposition failed (%s); using original query", type(e).__name__
        )

    return [query]


def expand_query(query: str) -> str:
    """Expand a query with related terms and synonyms using an LLM.

    Args:
        query: The original user query.

    Returns:
        An expanded version of the query.
    """
    if not classify_user_input(query).safe:
        return query

    llm = _get_llm()
    if llm is None:
        return query

    prompt = load_prompt_for(PromptOperation.RETRIEVAL_EXPAND).render(query=query)

    try:
        response = _invoke_prompt(llm, prompt)
        expanded = _safe_generated_query(response.content)
        if (
            expanded
            and len(expanded.split()) <= 50
            and _shares_scope(query, expanded)
        ):
            return expanded
    except Exception as e:
        logger.warning(
            "Query expansion failed (%s); using original query", type(e).__name__
        )

    return query
