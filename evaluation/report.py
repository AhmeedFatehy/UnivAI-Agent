"""Versioned evaluation reports tied to an exact dataset and runtime."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from telemetry.tracing import RuntimeFingerprint, ServingRecord

EVALUATION_REPORT_SCHEMA = "univai.agent.evaluation_report"
EVALUATION_REPORT_SCHEMA_VERSION = "1.0.0"


class DatasetInfo(BaseModel):
    """Which dataset the report measures. Missing data is never zero-filled."""

    path: str
    sha256: str
    case_count: int = Field(ge=0)
    version: str | None = None


class CaseResult(BaseModel):
    """One evaluated case with its trace and serving metadata."""

    id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    serving: ServingRecord | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None


class AggregateMetrics(BaseModel):
    """Mean metrics over the whole dataset."""

    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)


class Thresholds(BaseModel):
    """The pass/fail bar for critical grounding metrics."""

    context_recall: float = Field(default=0.60, ge=0.0, le=1.0)
    faithfulness: float = Field(default=0.70, ge=0.0, le=1.0)


class EvaluationReport(BaseModel):
    """The full, versioned result of one evaluation run."""

    schema_name: str = EVALUATION_REPORT_SCHEMA
    schema_version: str = EVALUATION_REPORT_SCHEMA_VERSION
    dataset: DatasetInfo
    runtime_config: RuntimeFingerprint
    metrics: AggregateMetrics
    per_case: list[CaseResult]
    thresholds: Thresholds
    passed: bool
    failures: list[str] = Field(default_factory=list)
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())


def build_report(
    *,
    dataset: DatasetInfo,
    runtime_config: RuntimeFingerprint,
    per_case: list[CaseResult],
    thresholds: Thresholds,
) -> EvaluationReport:
    """Aggregate per-case metrics and decide the pass/fail verdict.

    A critical grounding regression — aggregate context recall or faithfulness
    below its threshold — fails the report. So does a dataset with no cases.
    """
    failures: list[str] = []
    if dataset.case_count == 0:
        failures.append("dataset contains no cases")

    def mean(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        return round(sum(present) / len(present), 6)

    metrics = AggregateMetrics(
        context_precision=mean([case.context_precision for case in per_case]) or 0.0,
        context_recall=mean([case.context_recall for case in per_case]) or 0.0,
        mrr=mean([case.reciprocal_rank for case in per_case]) or 0.0,
        faithfulness=mean([case.faithfulness for case in per_case]),
        answer_relevancy=mean([case.answer_relevancy for case in per_case]),
    )

    if metrics.context_recall < thresholds.context_recall:
        failures.append(
            f"context recall {metrics.context_recall} below threshold "
            f"{thresholds.context_recall}"
        )
    if metrics.faithfulness is not None and metrics.faithfulness < thresholds.faithfulness:
        failures.append(
            f"faithfulness {metrics.faithfulness} below threshold "
            f"{thresholds.faithfulness}"
        )

    passed = not failures and len(per_case) == dataset.case_count
    if len(per_case) != dataset.case_count:
        failures.append(
            f"per-case count {len(per_case)} does not match dataset "
            f"{dataset.case_count}"
        )

    return EvaluationReport(
        dataset=dataset,
        runtime_config=runtime_config,
        metrics=metrics,
        per_case=per_case,
        thresholds=thresholds,
        passed=passed,
        failures=failures,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def render_markdown(report: EvaluationReport) -> str:
    """Human-readable report, one case per row, metrics at the top."""
    lines = [
        f"# RAG Evaluation Report",
        f"",
        f"- schema: {report.schema_name} v{report.schema_version}",
        f"- generated_at: {report.generated_at}",
        f"- dataset: `{report.dataset.path}`",
        f"  - sha256: `{report.dataset.sha256}`",
        f"  - cases: {report.dataset.case_count}",
        f"  - version: {report.dataset.version or 'unknown'}",
        f"- code revision: `{report.runtime_config.code_revision}`",
        f"- model: {report.runtime_config.model}",
        f"- dense embedding: {report.runtime_config.dense_embedding_model}",
        f"- sparse embedding: {report.runtime_config.sparse_embedding_model}",
        f"- reranker: {report.runtime_config.reranker_model}",
        f"- retrieval settings: {json.dumps(report.runtime_config.retrieval_settings, sort_keys=True)}",
        f"- prompt versions: {json.dumps(report.runtime_config.prompt_versions, sort_keys=True)}",
        f"",
        f"## Aggregate",
        f"",
        f"| metric | value |",
        f"| --- | --- |",
        f"| context_precision | {report.metrics.context_precision} |",
        f"| context_recall | {report.metrics.context_recall} |",
        f"| mrr | {report.metrics.mrr} |",
        f"| faithfulness | {report.metrics.faithfulness} |",
        f"| answer_relevancy | {report.metrics.answer_relevancy} |",
        f"",
        f"## Verdict",
        f"",
        f"- thresholds: context_recall >= {report.thresholds.context_recall}, "
        f"faithfulness >= {report.thresholds.faithfulness}",
        f"- **passed: {'YES' if report.passed else 'NO'}**",
        *(f"- failure: {failure}" for failure in report.failures),
        f"",
        f"## Per case",
        f"",
        f"| id | trace_id | cp | cr | rr | faithfulness | relevancy | prompt |",
        f"| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report.per_case:
        prompt = (
            f"{case.prompt_id}@{case.prompt_version}"
            if case.prompt_id and case.prompt_version
            else "unknown"
        )
        lines.append(
            f"| {case.id} | {case.trace_id[:8]} | {case.context_precision} | "
            f"{case.context_recall} | {case.reciprocal_rank} | "
            f"{case.faithfulness} | {case.answer_relevancy} | {prompt} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "EVALUATION_REPORT_SCHEMA",
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "AggregateMetrics",
    "CaseResult",
    "DatasetInfo",
    "EvaluationReport",
    "Thresholds",
    "build_report",
    "render_markdown",
]
