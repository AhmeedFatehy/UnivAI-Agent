"""RAG Evaluation Metrics — Faithfulness and Answer Relevancy.

Uses LLM as a judge to evaluate RAG quality without needing ground-truth datasets.
"""
import logging
import json

from config import LLM_MODEL
from agents.prompts import PromptOperation, load_prompt_for
from retrieval.query_transform import _get_llm  # reuse the LLM singleton

logger = logging.getLogger(__name__)


def evaluate_retrieval(query: str, retrieved_docs: list[dict]) -> dict:
    """Evaluate context precision (are the retrieved docs relevant to the query?)."""
    llm = _get_llm()
    if llm is None or not retrieved_docs:
        return {"context_precision": 0.0, "reasoning": "LLM or docs unavailable"}

    # Evaluate the top 3 documents to save tokens/time
    docs_to_eval = retrieved_docs[:3]
    context = "\n\n".join(
        f"[Doc {i+1}] {doc['content']}" for i, doc in enumerate(docs_to_eval)
    )

    prompt = load_prompt_for(PromptOperation.EVALUATION_RETRIEVAL).render(
        query=query, context=context
    )

    try:
        response = llm.invoke(prompt)
        content = _extract_json(response.content)
        result = json.loads(content)
        return {
            "context_precision": float(result.get("score", 0.0)),
            "reasoning": str(result.get("reasoning", "No reasoning provided")),
        }
    except Exception as e:
        logger.error("Context precision evaluation failed: %s", e)
        return {"context_precision": 0.0, "reasoning": f"Error: {e}"}


def evaluate_generation(query: str, answer: str, context: str) -> dict:
    """Evaluate faithfulness (groundedness) and answer relevancy."""
    llm = _get_llm()
    if llm is None or not answer or not context:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

    prompt = load_prompt_for(PromptOperation.EVALUATION_GROUNDEDNESS).render(
        query=query, context=context, answer=answer
    )

    try:
        response = llm.invoke(prompt)
        content = _extract_json(response.content)
        result = json.loads(content)
        return {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "reasoning": str(result.get("reasoning", "No reasoning provided")),
        }
    except Exception as e:
        logger.error("Generation evaluation failed: %s", e)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "reasoning": f"Error: {e}"}


def _extract_json(text: str) -> str:
    """Helper to extract JSON block from LLM output."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()
