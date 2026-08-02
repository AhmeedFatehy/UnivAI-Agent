"""Measurable RAG quality — deterministic label-based metrics and a strict judge.

Two families of metrics:

* **Deterministic, from relevance labels** — :func:`context_precision`,
  :func:`context_recall`, :func:`reciprocal_rank` and :func:`mrr` are pure
  functions of the relevant and retrieved document ids, so a reviewer can
  recompute them by hand. These feed the evaluation runner.
* **Judge-based** — faithfulness/groundedness and answer relevancy come from a
  judge model, validated strictly by :func:`parse_judge_output`. Malformed judge
  output raises :class:`JudgeOutputError`; it is never silently replaced with a
  zero. A missing judge raises :class:`JudgeUnavailableError`. There is no
  permissive fallback that pretends a failure was a score.

Nothing here estimates token counts, cost or judge scores.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Sequence

from pydantic import BaseModel, Field, field_validator

from agents.prompts import PromptOperation, load_prompt_for

logger = logging.getLogger(__name__)

METRICS_SCHEMA = "univai.agent.metrics"
METRICS_SCHEMA_VERSION = "1.0.0"


class JudgeOutputError(ValueError):
    """The judge replied with something that is not a valid scorecard."""


class JudgeUnavailableError(RuntimeError):
    """A judge-based metric was requested but no judge model is configured."""


# ── Deterministic retrieval metrics from relevance labels ─────────────


def context_precision(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    """RAGAS-style context precision.

    Averages ``precision@k`` over the rank positions that returned a relevant
    document. Returns ``0.0`` when no relevant document was retrieved (there is
    no position to be precise about).
    """
    relevant_set = set(relevant)
    hits = 0
    total = 0.0
    for index, document_id in enumerate(retrieved, start=1):
        if document_id in relevant_set:
            hits += 1
            total += hits / index
    return round(total / hits, 6) if hits else 0.0


def context_recall(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    """Fraction of relevant documents that appear in the retrieved set."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    found = sum(1 for document_id in retrieved if document_id in relevant_set)
    return round(found / len(relevant_set), 6)


def reciprocal_rank(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    """1/rank of the first relevant document; 0.0 when none was retrieved."""
    relevant_set = set(relevant)
    for index, document_id in enumerate(retrieved, start=1):
        if document_id in relevant_set:
            return round(1.0 / index, 6)
    return 0.0


def mrr(
    relevant_by_case: Sequence[Sequence[str]],
    retrieved_by_case: Sequence[Sequence[str]],
) -> float:
    """Mean reciprocal rank across cases. One label list per case."""
    if not relevant_by_case:
        return 0.0
    total = sum(
        reciprocal_rank(relevant, retrieved)
        for relevant, retrieved in zip(relevant_by_case, retrieved_by_case)
    )
    return round(total / len(relevant_by_case), 6)


# ── Strict judge output ───────────────────────────────────────────────


class JudgeScores(BaseModel):
    """A validated judge scorecard. Every score must be a real 0..1 number."""

    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    context_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None

    @field_validator("faithfulness", "answer_relevancy", "context_precision", mode="before")
    @classmethod
    def _scores_are_real_numbers(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("a score cannot be a boolean")
        return value

    def require(self, *fields: str) -> "JudgeScores":
        missing = [field for field in fields if getattr(self, field) is None]
        if missing:
            raise JudgeOutputError(
                f"judge output is missing required score field(s): {missing}"
            )
        return self


def parse_judge_output(raw: str) -> JudgeScores:
    """Parse and strictly validate a judge reply.

    Raises :class:`JudgeOutputError` for non-JSON replies, non-numeric scores,
    or scores outside 0..1. A malformed reply is a failure, never a zero.
    """
    from agents.schemas import extract_json

    text = (raw or "").strip()
    if not text:
        raise JudgeOutputError("judge returned an empty reply")

    try:
        payload = json.loads(extract_json(text))
    except (json.JSONDecodeError, TypeError) as error:
        raise JudgeOutputError(f"judge reply is not valid JSON: {error}") from error

    if not isinstance(payload, dict):
        raise JudgeOutputError(f"judge reply must be a JSON object, got {type(payload).__name__}")

    unknown = set(payload) - {"faithfulness", "answer_relevancy", "context_precision", "reasoning", "score"}
    if unknown:
        raise JudgeOutputError(f"judge reply has unknown field(s): {sorted(unknown)}")

    if "score" in payload:
        if "context_precision" in payload:
            raise JudgeOutputError("judge reply cannot set both 'score' and 'context_precision'")
        payload["context_precision"] = payload.pop("score")

    try:
        return JudgeScores.model_validate(payload)
    except Exception as error:  # noqa: BLE001 - re-raised as a typed judge failure
        raise JudgeOutputError(f"judge reply is not a valid scorecard: {error}") from error


def _render_context(retrieved_docs: Sequence[dict]) -> str:
    return "\n\n".join(
        f"[Doc {index}] {doc.get('content', '')}" for index, doc in enumerate(retrieved_docs, start=1)
    )


def _judge_callable():
    """The LLM judge singleton, or an explicit failure — never a silent zero."""
    from retrieval.query_transform import _get_llm

    judge = _get_llm()
    if judge is None:
        raise JudgeUnavailableError(
            "no judge model is configured (LLM_MODEL/LLM_BASE_URL); "
            "judge-based metrics require one"
        )
    return judge


def judge_retrieval(query: str, retrieved_docs: Sequence[dict], judge) -> JudgeScores:
    """Ask the judge how relevant the retrieved docs are to ``query``."""
    prompt = load_prompt_for(PromptOperation.EVALUATION_RETRIEVAL).render(
        query=query, context=_render_context(retrieved_docs)
    )
    response = judge.invoke(prompt)
    content = getattr(response, "content", response)
    text = content if isinstance(content, str) else str(content)
    return parse_judge_output(text).require("context_precision")


def judge_generation(query: str, answer: str, context: str, judge) -> JudgeScores:
    """Ask the judge for faithfulness/groundedness and answer relevancy."""
    prompt = load_prompt_for(PromptOperation.EVALUATION_GROUNDEDNESS).render(
        query=query, context=context, answer=answer
    )
    response = judge.invoke(prompt)
    content = getattr(response, "content", response)
    text = content if isinstance(content, str) else str(content)
    return parse_judge_output(text).require("faithfulness", "answer_relevancy")


# ── Compatibility entry points (strict; never silently zero) ──────────


def evaluate_retrieval(query: str, retrieved_docs: list[dict]) -> dict:
    """Context precision from a real judge model.

    Raises :class:`JudgeUnavailableError` without a configured judge and
    :class:`JudgeOutputError` on malformed judge output — a failed evaluation is
    never reported as a zero.
    """
    scores = judge_retrieval(query, retrieved_docs, _judge_callable())
    return {
        "context_precision": scores.context_precision,
        "reasoning": scores.reasoning or "No reasoning provided",
    }


def evaluate_generation(query: str, answer: str, context: str) -> dict:
    """Faithfulness/groundedness and answer relevancy from a real judge model.

    Same strictness contract as :func:`evaluate_retrieval`.
    """
    scores = judge_generation(query, answer, context, _judge_callable())
    return {
        "faithfulness": scores.faithfulness,
        "answer_relevancy": scores.answer_relevancy,
        "reasoning": scores.reasoning or "No reasoning provided",
    }


__all__ = [
    "METRICS_SCHEMA",
    "METRICS_SCHEMA_VERSION",
    "JudgeOutputError",
    "JudgeScores",
    "JudgeUnavailableError",
    "context_precision",
    "context_recall",
    "evaluate_generation",
    "evaluate_retrieval",
    "judge_generation",
    "judge_retrieval",
    "mrr",
    "parse_judge_output",
    "reciprocal_rank",
]
