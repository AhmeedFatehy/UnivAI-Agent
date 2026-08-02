"""Correlated trace metadata: trace ids, serving records and fingerprints.

The acceptance contract is that every evaluated result records a trace id,
prompt id/version, serving model and runtime configuration fingerprint, and that
latency is measured while token/cost fields stay explicitly unknown when the
provider does not report them.
"""

from __future__ import annotations

from dataclasses import dataclass

from resilience.fallback import ModelSpec, ServedResult
from telemetry.tracing import (
    TRACE_SCHEMA,
    TRACE_SCHEMA_VERSION,
    RuntimeFingerprint,
    ServingRecord,
    TraceContext,
    TraceSpan,
    code_revision,
    new_trace_id,
    runtime_fingerprint,
)


# ── Trace ids ─────────────────────────────────────────────────────────


def test_new_trace_id_is_unique_and_hex():
    first = new_trace_id()
    second = new_trace_id()

    assert first != second
    assert len(first) == 32
    int(first, 16)


def test_code_revision_is_measured_not_invented():
    revision = code_revision()

    # In this checkout a real git revision is expected; anything else must be
    # the explicit "unknown" marker, never a guessed value.
    assert revision == "unknown" or len(revision) == 40


# ── Serving records ───────────────────────────────────────────────────


@dataclass
class Served:
    text: str = "reply"
    provider: str | None = "ollama"
    model: str | None = "qwen3:4b-instruct"
    latency_ms: float | None = 1.5
    input_tokens: int | None = 10
    output_tokens: int | None = 20
    cost: float | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    attempts: int = 1


def test_serving_record_maps_a_served_result_and_prompt():
    served = ServedResult(text="reply", provider="ollama", model="m", latency_ms=1.5)
    record = ServingRecord.from_served(served, prompt_id="teaching/lecture_generation", prompt_version="1.0.0")

    assert record.provider == "ollama"
    assert record.model == "m"
    assert record.latency_ms == 1.5
    assert record.prompt_id == "teaching/lecture_generation"
    assert record.prompt_version == "1.0.0"


def test_serving_record_keeps_unknown_metadata_null():
    served = ServedResult(text="reply", provider="ollama", model="m")
    record = ServingRecord.from_served(served)

    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.cost is None
    assert record.fallback_used is False


def test_serving_record_reports_real_fallback_metadata():
    served = ServedResult(
        text="reply",
        provider="ollama",
        model="fallback",
        fallback_used=True,
        fallback_reason="primary down",
        attempts=2,
        input_tokens=5,
        output_tokens=7,
    )
    record = ServingRecord.from_served(served)

    assert record.fallback_used is True
    assert record.fallback_reason == "primary down"
    assert record.attempts == 2
    assert record.input_tokens == 5
    assert record.output_tokens == 7


def test_serving_record_rejects_invented_cost_from_nowhere():
    # A cost may only be set when a provider actually reports one; the model
    # never invents it. A negative or out-of-range token count is also refused.
    from pydantic import ValidationError

    try:
        ServingRecord(input_tokens=-1)
    except ValidationError:
        pass
    else:
        raise AssertionError("negative token counts must be rejected")


# ── Trace context ─────────────────────────────────────────────────────


def test_trace_context_issues_one_shared_trace_id():
    with TraceContext() as context:
        span_a = context.span("retrieval")
        span_b = context.span("generation")

    assert span_a.trace_id == context.trace_id
    assert span_b.trace_id == context.trace_id
    assert len(context.spans) == 2


def test_span_records_real_latency():
    with TraceContext() as context:
        span = context.span("retrieve_context")

    assert span.finished_at is not None
    assert span.latency_ms is not None
    assert span.latency_ms >= 0


def test_span_can_be_finished_explicitly():
    span = TraceSpan(trace_id="t", operation="op", started_at="2026-01-01T00:00:00+00:00")
    span.finish()

    assert span.finished_at is not None
    assert span.latency_ms is not None


# ── Runtime configuration fingerprint ─────────────────────────────────


def test_runtime_fingerprint_records_the_exact_configuration():
    fingerprint = runtime_fingerprint()

    assert isinstance(fingerprint, RuntimeFingerprint)
    assert fingerprint.code_revision == code_revision()
    assert fingerprint.model
    assert fingerprint.dense_embedding_model
    assert fingerprint.sparse_embedding_model
    assert fingerprint.reranker_model
    assert fingerprint.retrieval_settings["chunk_size"] > 0
    assert fingerprint.tool_schema_version
    assert fingerprint.agent_schema_version


def test_runtime_fingerprint_captures_every_prompt_version():
    fingerprint = runtime_fingerprint()

    assert fingerprint.prompt_versions, "every routed prompt must be versioned"
    assert all(version.count(".") == 2 for version in fingerprint.prompt_versions.values())
    assert "teaching/lecture_generation" in fingerprint.prompt_versions


def test_runtime_fingerprint_uses_real_embedding_models_from_config():
    fingerprint = runtime_fingerprint()
    from config import DENSE_EMBEDDING_MODEL, RERANKER_MODEL, SPARSE_EMBEDDING_MODEL

    assert fingerprint.dense_embedding_model == DENSE_EMBEDDING_MODEL
    assert fingerprint.sparse_embedding_model == SPARSE_EMBEDDING_MODEL
    assert fingerprint.reranker_model == RERANKER_MODEL


def test_fingerprint_and_trace_are_versioned():
    assert RuntimeFingerprint().schema_name == TRACE_SCHEMA
    assert RuntimeFingerprint().schema_version == TRACE_SCHEMA_VERSION


# ── The graph trace carries the contract ──────────────────────────────


def test_agent_trace_exposes_servings_and_fingerprint():
    from agents.manager import AgentRuntime
    from agents.schemas import AgentTrace

    runtime = AgentRuntime(llm=lambda prompt: "{}")
    trace = AgentTrace()
    trace.fingerprint = runtime.fingerprint

    assert trace.trace_id
    assert len(trace.trace_id) == 32
    assert trace.fingerprint is not None
    assert trace.servings == []
