"""Deterministic RAG evaluation runner with a versioned, gated report.

Run from the Agent repository root::

    uv run python -m evaluation.runner --dataset ../UnivAI/tests/fixtures/evaluation/grounded-v1.jsonl --mode mock

**Dataset contract** (JSON Lines, one object per case):

.. code-block:: json

    {
      "id": "case-001",
      "query": "How does a hash table handle collisions?",
      "relevant": ["doc-1", "doc-2"],
      "retrieved": ["doc-1", "doc-3", "doc-2"],
      "judge": {"faithfulness": 0.9, "answer_relevancy": 0.8, "reasoning": "..."},
      "trace_id": "optional shared trace id",
      "prompt_id": "teaching/lecture_generation",
      "prompt_version": "1.0.0",
      "serving": {"provider": "ollama", "model": "qwen3:4b-instruct"}
    }

* ``relevant`` — the ground-truth document ids for the query.
* ``retrieved`` — the ids the pipeline actually returned, in rank order.
* ``judge`` — required in ``--mode mock``. It is validated with the same strict
  rules as live judge output, so a malformed entry fails the run explicitly.

**Modes**

* ``mock`` — no model, no network. Judge scores come from the dataset and are
  strictly validated.
* ``judge`` — faithfulness/answer relevancy come from a real judge model using
  the dataset's ``answer``/``context`` fields. No model call happens in CI.

**Exit codes**

* ``0`` — report passed all thresholds.
* ``1`` — the report failed: a critical grounding regression, an empty dataset,
  or malformed case data.
* ``2`` — the dataset file itself is missing or unreadable. A missing dataset is
  an explicit failure, never a zero-filled success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from evaluation.metrics import (
    JudgeOutputError,
    context_precision,
    context_recall,
    parse_judge_output,
    reciprocal_rank,
)
from evaluation.report import (
    CaseResult,
    DatasetInfo,
    EvaluationReport,
    Thresholds,
    build_report,
    render_markdown,
)
from telemetry.tracing import (
    RuntimeFingerprint,
    ServingRecord,
    new_trace_id,
    runtime_fingerprint,
)

RUNNER_SCHEMA = "univai.agent.evaluation_runner"
RUNNER_SCHEMA_VERSION = "1.0.0"

#: Fields the dataset entry's judge block is allowed to carry. Anything else is
#: a malformed judge and fails the run.
JUDGE_FIELDS = {
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "score",
    "reasoning",
}


class DatasetError(ValueError):
    """A dataset entry is malformed. The run fails explicitly."""


class DatasetEntry(BaseModel):
    id: str = Field(min_length=1)
    query: str = ""
    relevant: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)
    answer: str | None = None
    context: str | None = None
    judge: dict[str, Any] | None = None
    trace_id: str | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None
    serving: dict[str, Any] | None = None

    @property
    def has_judge(self) -> bool:
        return self.judge is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[list[DatasetEntry], str]:
    """Load and validate a JSONL dataset. Missing files fail explicitly."""
    if not path.is_file():
        raise FileNotFoundError(
            f"evaluation dataset not found: {path}. The approved 50-case Core "
            "fixture must be published before the real run can happen."
        )
    digest = _sha256(path)

    entries: list[DatasetEntry] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise DatasetError(
                    f"dataset line {line_number} is not valid JSON: {error}"
                ) from error
            try:
                entries.append(DatasetEntry.model_validate(payload))
            except ValidationError as error:
                raise DatasetError(
                    f"dataset line {line_number} is not a valid case: {error}"
                ) from error
    return entries, digest


def _judge_scores(entry: DatasetEntry, mode: str) -> tuple[float, float]:
    """Return validated (faithfulness, answer_relevancy) for one case."""
    if mode == "judge":
        if not entry.answer or not entry.context:
            raise DatasetError(
                f"case '{entry.id}': judge mode needs 'answer' and 'context' fields"
            )
        from evaluation.metrics import judge_generation

        from retrieval.query_transform import _get_llm

        judge = _get_llm()
        if judge is None:
            raise DatasetError(
                f"case '{entry.id}': no judge model is configured for judge mode"
            )
        scores = judge_generation(entry.query, entry.answer, entry.context, judge)
        return scores.faithfulness, scores.answer_relevancy

    if entry.judge is None:
        raise DatasetError(
            f"case '{entry.id}': mock mode needs a 'judge' block with "
            "faithfulness and answer_relevancy"
        )
    unknown = set(entry.judge) - JUDGE_FIELDS
    if unknown:
        raise DatasetError(
            f"case '{entry.id}': judge block has unknown field(s): {sorted(unknown)}"
        )
    try:
        scores = parse_judge_output(json.dumps(entry.judge)).require(
            "faithfulness", "answer_relevancy"
        )
    except JudgeOutputError as error:
        raise DatasetError(f"case '{entry.id}': malformed judge output: {error}") from error
    return scores.faithfulness, scores.answer_relevancy


def _serving_record(entry: DatasetEntry) -> ServingRecord | None:
    if not entry.serving:
        return None
    try:
        return ServingRecord.model_validate(entry.serving)
    except ValidationError as error:
        raise DatasetError(
            f"case '{entry.id}': serving metadata is malformed: {error}"
        ) from error


def evaluate_cases(
    entries: list[DatasetEntry], mode: str
) -> list[CaseResult]:
    """Compute every metric for every case. Malformed data raises."""
    results: list[CaseResult] = []
    for entry in entries:
        faithfulness, answer_relevancy = _judge_scores(entry, mode)
        results.append(
            CaseResult(
                id=entry.id,
                trace_id=entry.trace_id or new_trace_id(),
                context_precision=context_precision(entry.relevant, entry.retrieved),
                context_recall=context_recall(entry.relevant, entry.retrieved),
                reciprocal_rank=reciprocal_rank(entry.relevant, entry.retrieved),
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                serving=_serving_record(entry),
                prompt_id=entry.prompt_id,
                prompt_version=entry.prompt_version,
            )
        )
    return results


def run_evaluation(
    dataset_path: Path,
    mode: str,
    *,
    thresholds: Thresholds,
) -> EvaluationReport:
    """Load the dataset, measure it, and build the versioned report."""
    entries, digest = load_dataset(dataset_path)
    per_case = evaluate_cases(entries, mode)
    return build_report(
        dataset=DatasetInfo(
            path=str(dataset_path),
            sha256=digest,
            case_count=len(entries),
        ),
        runtime_config=runtime_fingerprint(),
        per_case=per_case,
        thresholds=thresholds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluation.runner",
        description="Run deterministic RAG evaluation and emit a versioned report.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL dataset path")
    parser.add_argument("--mode", choices=["mock", "judge"], default="mock")
    parser.add_argument("--context-recall-threshold", type=float, default=0.60)
    parser.add_argument("--faithfulness-threshold", type=float, default=0.70)
    parser.add_argument("--output", type=Path, help="write report markdown to this path")
    args = parser.parse_args(argv)

    thresholds = Thresholds(
        context_recall=args.context_recall_threshold,
        faithfulness=args.faithfulness_threshold,
    )

    try:
        report = run_evaluation(args.dataset, args.mode, thresholds=thresholds)
    except FileNotFoundError as error:
        print(f"EVALUATION NOT RUN: {error}", file=sys.stderr)
        return 2
    except (DatasetError, JudgeOutputError) as error:
        print(f"EVALUATION FAILED: {error}", file=sys.stderr)
        return 1

    markdown = render_markdown(report)
    print(markdown)

    if args.output:
        args.output.write_text(markdown, encoding="utf-8")

    if report.passed:
        print("\nEVALUATION PASSED", flush=True)
        return 0
    print("\nEVALUATION FAILED:", *report.failures, sep="\n- ", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
