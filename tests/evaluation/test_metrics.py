"""Measurable RAG quality: label-based metrics, strict judge validation, runner.

Every deterministic metric below is recomputed by hand so a reviewer can verify
it. Judge output and dataset entries are validated strictly: malformed data
fails explicitly and is never replaced with a zero.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.metrics import (
    JudgeOutputError,
    JudgeUnavailableError,
    JudgeScores,
    context_precision,
    context_recall,
    judge_generation,
    judge_retrieval,
    mrr,
    parse_judge_output,
    reciprocal_rank,
)
from evaluation.report import DatasetInfo, EvaluationReport, Thresholds, build_report
from evaluation.runner import (
    DatasetError,
    load_dataset,
    main,
    run_evaluation,
)


# ── Deterministic retrieval metrics (recomputed by hand) ──────────────


def test_context_precision_matches_the_hand_calculation():
    # relevant hits at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.833333
    assert context_precision(["doc1", "doc3"], ["doc1", "doc2", "doc3"]) == pytest.approx(0.833333)
    assert context_precision(["doc2", "doc4"], ["doc1", "doc2", "doc3"]) == pytest.approx(0.5)
    assert context_precision(["doc5"], ["doc1", "doc2"]) == 0.0


def test_context_recall_matches_the_hand_calculation():
    assert context_recall(["doc1", "doc3"], ["doc1", "doc2", "doc3"]) == 1.0
    assert context_recall(["doc2", "doc4"], ["doc1", "doc2", "doc3"]) == pytest.approx(0.5)
    assert context_recall([], ["doc1"]) == 0.0


def test_duplicate_retrieval_ids_do_not_inflate_metrics():
    assert context_recall(["doc1"], ["doc1", "doc1"]) == 1.0
    assert context_precision(["doc1"], ["doc1", "doc1"]) == 1.0


def test_reciprocal_rank_matches_the_hand_calculation():
    assert reciprocal_rank(["doc3"], ["doc1", "doc2", "doc3"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["doc2", "doc4"], ["doc1", "doc2"]) == pytest.approx(1 / 2)
    assert reciprocal_rank(["doc9"], ["doc1", "doc2"]) == 0.0


def test_mrr_averages_reciprocal_ranks():
    assert mrr(
        [["doc1"], ["doc2"], ["doc9"]],
        [["doc1", "doc2"], ["doc1", "doc2"], ["doc1", "doc2"]],
    ) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_retrieval_metrics_are_deterministic():
    relevant = ["a", "b"]
    retrieved = ["b", "c", "a"]
    assert context_precision(relevant, retrieved) == context_precision(relevant, retrieved)
    assert context_recall(relevant, retrieved) == context_recall(relevant, retrieved)
    assert reciprocal_rank(relevant, retrieved) == reciprocal_rank(relevant, retrieved)


# ── Strict judge output validation ────────────────────────────────────


def test_valid_judge_output_parses():
    scores = parse_judge_output(
        '{"faithfulness": 0.9, "answer_relevancy": 0.8, "reasoning": "supported"}'
    )
    assert scores.faithfulness == pytest.approx(0.9)
    assert scores.answer_relevancy == pytest.approx(0.8)
    assert scores.reasoning == "supported"


def test_judge_score_aliases_to_context_precision():
    scores = parse_judge_output('{"score": 0.7, "reasoning": "relevant"}')
    assert scores.context_precision == pytest.approx(0.7)
    scores.require("context_precision")


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        '{"faithfulness": 1.5, "answer_relevancy": 0.8}',
        '{"faithfulness": "high", "answer_relevancy": 0.8}',
        '{"faithfulness": true, "answer_relevancy": 0.8}',
        '{"faithfulness": 0.9, "answer_relevancy": 0.8, "made_up": 1}',
        '[1, 2, 3]',
        "",
    ],
)
def test_malformed_judge_output_fails_explicitly(raw):
    with pytest.raises(JudgeOutputError):
        parse_judge_output(raw)


def test_require_rejects_missing_score_fields():
    scores = parse_judge_output('{"faithfulness": 0.9}')
    with pytest.raises(JudgeOutputError, match="answer_relevancy"):
        scores.require("faithfulness", "answer_relevancy")


def test_judge_output_rejects_both_score_and_context_precision():
    with pytest.raises(JudgeOutputError, match="cannot set both"):
        parse_judge_output('{"score": 0.7, "context_precision": 0.8}')


# ── Judge-based metrics with a scripted judge ─────────────────────────


class FakeJudge:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply)


def test_judge_generation_validates_and_uses_the_registry_prompt():
    judge = FakeJudge('{"faithfulness": 0.9, "answer_relevancy": 0.85, "reasoning": "ok"}')
    scores = judge_generation("q", "answer", "context", judge)

    assert scores.faithfulness == pytest.approx(0.9)
    assert scores.answer_relevancy == pytest.approx(0.85)
    rendered = str(judge.prompts[0])
    assert "PROMPT BOUNDARY POLICY" in rendered
    assert 'name="query"' in rendered and "q" in rendered
    assert 'name="answer"' in rendered and "answer" in rendered


def test_judge_retrieval_uses_the_retrieval_prompt():
    judge = FakeJudge('{"score": 0.6, "reasoning": "partially"}')
    scores = judge_retrieval("q", [{"content": "some passage"}], judge)

    assert scores.context_precision == pytest.approx(0.6)
    rendered = str(judge.prompts[0])
    assert "PROMPT BOUNDARY POLICY" in rendered
    assert "[Doc 1] some passage" in rendered


def test_a_malformed_judge_reply_fails_the_generation_evaluation():
    judge = FakeJudge("sorry, no json here")
    with pytest.raises(JudgeOutputError):
        judge_generation("q", "a", "c", judge)


def test_evaluate_generation_without_a_judge_fails_explicitly(monkeypatch):
    monkeypatch.setattr("retrieval.query_transform._get_llm", lambda: None)
    from evaluation import metrics as metrics_module

    with pytest.raises(JudgeUnavailableError, match="no judge model"):
        metrics_module.evaluate_generation("q", "a", "c")


def test_evaluate_retrieval_without_a_judge_fails_explicitly(monkeypatch):
    monkeypatch.setattr("retrieval.query_transform._get_llm", lambda: None)
    from evaluation import metrics as metrics_module

    with pytest.raises(JudgeUnavailableError, match="no judge model"):
        metrics_module.evaluate_retrieval("q", [{"content": "x"}])


# ── Evaluation report ─────────────────────────────────────────────────


def report_for(per_case, *, context_recall=1.0, faithfulness=0.9):
    from telemetry.tracing import RuntimeFingerprint

    return build_report(
        dataset=DatasetInfo(path="fixture.jsonl", sha256="abc", case_count=len(per_case)),
        runtime_config=RuntimeFingerprint(code_revision="deadbeef"),
        per_case=per_case,
        thresholds=Thresholds(context_recall=context_recall, faithfulness=0.7),
    )


def test_build_report_aggregates_and_passes():
    from evaluation.report import CaseResult

    cases = [
        CaseResult(id="a", trace_id="t1", context_precision=1.0, context_recall=1.0, reciprocal_rank=1.0, faithfulness=0.9, answer_relevancy=0.8),
        CaseResult(id="b", trace_id="t2", context_precision=0.5, context_recall=1.0, reciprocal_rank=0.5, faithfulness=0.8, answer_relevancy=0.7),
    ]
    report = report_for(cases)

    assert report.passed is True
    assert report.metrics.context_recall == 1.0
    assert report.metrics.context_precision == pytest.approx(0.75)
    assert report.metrics.mrr == pytest.approx(0.75)
    assert report.metrics.faithfulness == pytest.approx(0.85)


def test_critical_grounding_regression_fails_the_report():
    from evaluation.report import CaseResult

    cases = [
        CaseResult(id="a", trace_id="t1", context_precision=0.2, context_recall=0.2, reciprocal_rank=0.2, faithfulness=0.9, answer_relevancy=0.8),
    ]
    report = report_for(cases, context_recall=0.6)

    assert report.passed is False
    assert any("context recall" in failure for failure in report.failures)


def test_low_faithfulness_fails_the_report():
    from evaluation.report import CaseResult

    cases = [
        CaseResult(id="a", trace_id="t1", context_precision=1.0, context_recall=1.0, reciprocal_rank=1.0, faithfulness=0.4, answer_relevancy=0.8),
    ]
    report = report_for(cases)

    assert report.passed is False
    assert any("faithfulness" in failure for failure in report.failures)


# ── Evaluation runner against a dataset ───────────────────────────────


def write_dataset(path: Path, cases: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    return path


def good_dataset(path: Path) -> Path:
    return write_dataset(
        path,
        [
            {
                "id": "case-001",
                "query": "hash table collisions",
                "relevant": ["doc-1", "doc-3"],
                "retrieved": ["doc-1", "doc-2", "doc-3"],
                "judge": {"faithfulness": 0.9, "answer_relevancy": 0.8},
                "trace_id": "trace-001",
                "prompt_id": "teaching/lecture_generation",
                "prompt_version": "1.0.0",
            },
            {
                "id": "case-002",
                "query": "sorting complexity",
                "relevant": ["doc-2"],
                "retrieved": ["doc-2"],
                "judge": {"faithfulness": 1.0, "answer_relevancy": 0.9},
            },
        ],
    )


def test_runner_builds_a_versioned_report(tmp_path):
    dataset = good_dataset(tmp_path / "grounded-v1.jsonl")
    report = run_evaluation(dataset, "mock", thresholds=Thresholds())

    assert isinstance(report, EvaluationReport)
    assert report.dataset.case_count == 2
    assert len(report.dataset.sha256) == 64
    assert report.dataset.sha256 == load_dataset(dataset)[1]
    assert report.runtime_config.code_revision
    assert report.passed is True


def test_runner_records_trace_prompt_and_serving_metadata(tmp_path):
    dataset = write_dataset(
        tmp_path / "meta.jsonl",
        [
            {
                "id": "case-001",
                "relevant": ["a"],
                "retrieved": ["a"],
                "judge": {"faithfulness": 0.9, "answer_relevancy": 0.9},
                "trace_id": "trace-001",
                "prompt_id": "teaching/lecture_generation",
                "prompt_version": "1.0.0",
                "serving": {"provider": "ollama", "model": "qwen3:4b-instruct", "attempts": 1},
            }
        ],
    )
    report = run_evaluation(dataset, "mock", thresholds=Thresholds())

    case = report.per_case[0]
    assert case.trace_id == "trace-001"
    assert case.prompt_id == "teaching/lecture_generation"
    assert case.prompt_version == "1.0.0"
    assert case.serving is not None
    assert case.serving.model == "qwen3:4b-instruct"


def test_runner_generates_a_trace_id_when_the_dataset_omits_it(tmp_path):
    dataset = write_dataset(
        tmp_path / "notrace.jsonl",
        [
            {
                "id": "case-001",
                "relevant": ["a"],
                "retrieved": ["a"],
                "judge": {"faithfulness": 0.9, "answer_relevancy": 0.9},
            }
        ],
    )
    report = run_evaluation(dataset, "mock", thresholds=Thresholds())

    assert len(report.per_case[0].trace_id) == 32


def test_runner_without_mock_judge_scores_fails_explicitly(tmp_path):
    dataset = write_dataset(
        tmp_path / "nojudge.jsonl",
        [{"id": "case-001", "relevant": ["a"], "retrieved": ["a"]}],
    )
    with pytest.raises(DatasetError, match="needs a 'judge' block"):
        run_evaluation(dataset, "mock", thresholds=Thresholds())


def test_runner_rejects_malformed_judge_output(tmp_path):
    dataset = write_dataset(
        tmp_path / "badjudge.jsonl",
        [
            {
                "id": "case-001",
                "relevant": ["a"],
                "retrieved": ["a"],
                "judge": {"faithfulness": 9.0, "answer_relevancy": "high"},
            }
        ],
    )
    with pytest.raises(DatasetError, match="malformed judge output"):
        run_evaluation(dataset, "mock", thresholds=Thresholds())


def test_runner_rejects_malformed_dataset_lines(tmp_path):
    path = tmp_path / "malformed.jsonl"
    path.write_text('{"id": "ok", "relevant": ["a"], "retrieved": ["a"]}\nnot json\n', encoding="utf-8")

    with pytest.raises(DatasetError, match="not valid JSON"):
        run_evaluation(path, "mock", thresholds=Thresholds())


def test_runner_with_missing_dataset_fails_explicitly(tmp_path):
    missing = tmp_path / "does-not-exist.jsonl"

    with pytest.raises(FileNotFoundError, match="evaluation dataset not found"):
        load_dataset(missing)


# ── Runner CLI exit codes ─────────────────────────────────────────────


def test_cli_exit_zero_when_the_report_passes(tmp_path):
    dataset = good_dataset(tmp_path / "pass.jsonl")
    assert main(["--dataset", str(dataset), "--mode", "mock"]) == 0


def test_cli_exit_one_on_critical_regression(tmp_path):
    dataset = write_dataset(
        tmp_path / "fail.jsonl",
        [
            {
                "id": "case-001",
                "relevant": ["a", "b", "c"],
                "retrieved": ["x", "y"],
                "judge": {"faithfulness": 0.2, "answer_relevancy": 0.1},
            }
        ],
    )
    assert main(["--dataset", str(dataset), "--mode", "mock"]) == 1


def test_cli_exit_two_when_the_dataset_is_missing(tmp_path):
    assert main(["--dataset", str(tmp_path / "missing.jsonl"), "--mode", "mock"]) == 2


def test_cli_writes_a_markdown_report(tmp_path):
    dataset = good_dataset(tmp_path / "src.jsonl")
    output = tmp_path / "report.md"
    assert main(["--dataset", str(dataset), "--mode", "mock", "--output", str(output)]) == 0

    rendered = output.read_text(encoding="utf-8")
    assert "# RAG Evaluation Report" in rendered
    assert "passed: YES" in rendered
    assert "case-001" in rendered
    assert "teaching/lecture_generation@1.0.0" in rendered
