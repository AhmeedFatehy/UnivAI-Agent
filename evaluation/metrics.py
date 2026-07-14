"""RAG Evaluation Metrics — Faithfulness and Answer Relevancy.

Uses LLM as a judge to evaluate RAG quality without needing ground-truth datasets.
"""
import logging
import json

from config import LLM_MODEL
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

    prompt = f"""You are a RAG evaluator. Your task is to evaluate if the retrieved context is relevant to the query.
Score the context precision from 0.0 to 1.0 (where 1.0 means highly relevant context that fully answers the query, and 0.0 means completely irrelevant).

Query: {query}
Context:
{context}

Return ONLY a JSON object with two keys: "score" (float) and "reasoning" (brief string).
Example: {{"score": 0.8, "reasoning": "Doc 1 and 2 directly address the query, but Doc 3 is irrelevant."}}
"""

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

    prompt = f"""You are a RAG evaluator. Evaluate the generated answer based on the query and context.
Score two metrics from 0.0 to 1.0:
1. Faithfulness: Is the answer fully supported by the context? (1.0 = fully supported, no hallucinations)
2. Answer Relevancy: Does the answer directly address the user's query? (1.0 = direct and complete)

Query: {query}
Context: {context}
Answer: {answer}

Return ONLY a JSON object with keys: "faithfulness", "answer_relevancy", and "reasoning".
Example: {{"faithfulness": 0.9, "answer_relevancy": 1.0, "reasoning": "Answer is relevant and grounded, minor extra detail not in context."}}
"""

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
