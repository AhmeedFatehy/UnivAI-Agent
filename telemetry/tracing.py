"""One correlated trace across retrieval and generation, plus runtime metadata.

The Agent produces three kinds of observable metadata:

* **trace id** — one value shared by every retrieval call, LLM serving and
  evaluation case of a single run (:func:`new_trace_id`, :class:`TraceContext`).
* **serving metadata** — which provider/model served each reply, why a fallback
  happened, and the real latency. Token/cost fields are recorded only from real
  provider metadata; when a provider does not report them they remain ``None``
  and render as *unknown*. They are never estimated.
* **runtime configuration fingerprint** — an exact record of the code revision,
  model, dense/sparse embeddings, reranker, retrieval settings and every prompt
  ID/version, so an evaluated result can be reproduced and audited later.

Nothing here logs prompts, evidence blocks, learner data or credentials.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

TRACE_SCHEMA = "univai.agent.trace"
TRACE_SCHEMA_VERSION = "1.0.0"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def new_trace_id() -> str:
    """A fresh, unguessable trace id shared across a single run."""
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_revision() -> str:
    """The current git revision, or ``unknown`` when it cannot be read."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git repo must not crash tracing
        pass
    return "unknown"


class ServingRecord(BaseModel):
    """One model reply with the metadata that is actually known."""

    provider: str | None = None
    model: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    attempts: int = Field(default=1, ge=1)
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    prompt_id: str | None = None
    prompt_version: str | None = None

    @classmethod
    def from_served(cls, served, *, prompt_id: str | None = None, prompt_version: str | None = None) -> "ServingRecord":
        """Build from a :class:`resilience.fallback.ServedResult`."""

        from resilience.fallback import ServedResult

        assert isinstance(served, ServedResult)
        return cls(
            provider=served.provider,
            model=served.model,
            fallback_used=served.fallback_used,
            fallback_reason=served.fallback_reason,
            attempts=served.attempts,
            latency_ms=served.latency_ms,
            input_tokens=served.input_tokens,
            output_tokens=served.output_tokens,
            cost=served.cost,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )


class TraceSpan(BaseModel):
    """One measured operation inside a trace."""

    trace_id: str
    operation: str = Field(min_length=1)
    started_at: str
    finished_at: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    serving: ServingRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def finish(self) -> "TraceSpan":
        self.finished_at = _now()
        self.latency_ms = _millis_between(self.started_at, self.finished_at)
        return self


def _millis_between(started: str, finished: str) -> float:
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(finished)
        return round((end - start).total_seconds() * 1000, 4)
    except ValueError:
        return None


class RuntimeFingerprint(BaseModel):
    """The exact configuration an evaluated result was produced with."""

    schema_name: str = TRACE_SCHEMA
    schema_version: str = TRACE_SCHEMA_VERSION
    code_revision: str = "unknown"
    model: str | None = None
    dense_embedding_model: str | None = None
    sparse_embedding_model: str | None = None
    reranker_model: str | None = None
    retrieval_settings: dict[str, Any] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    tool_schema_version: str | None = None
    agent_schema_version: str | None = None


def _prompt_versions() -> dict[str, str]:
    """Every routed prompt id mapped to its semantic version."""
    try:
        from agents.prompts import validate_prompt_catalog

        catalog = validate_prompt_catalog()
        return {prompt.name.value: prompt.version for prompt in catalog.values()}
    except Exception:  # noqa: BLE001 - a broken catalog must not crash fingerprinting
        return {}


def runtime_fingerprint() -> RuntimeFingerprint:
    """Build the current runtime configuration fingerprint from live state."""
    from config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        DEFAULT_SEARCH_LIMIT,
        DENSE_EMBEDDING_MODEL,
        LLM_MODEL,
        PREFETCH_LIMIT,
        RERANKER_MODEL,
        SPARSE_EMBEDDING_MODEL,
    )

    from agents.schemas import AGENT_SCHEMA_VERSION
    from tools.registry import TOOL_SCHEMA_VERSION

    return RuntimeFingerprint(
        code_revision=code_revision(),
        model=LLM_MODEL,
        dense_embedding_model=DENSE_EMBEDDING_MODEL,
        sparse_embedding_model=SPARSE_EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        retrieval_settings={
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "default_search_limit": DEFAULT_SEARCH_LIMIT,
            "prefetch_limit": PREFETCH_LIMIT,
        },
        prompt_versions=_prompt_versions(),
        tool_schema_version=TOOL_SCHEMA_VERSION,
        agent_schema_version=AGENT_SCHEMA_VERSION,
    )


class TraceContext(AbstractContextManager):
    """One correlated run: a shared trace id and ordered, measured spans."""

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or new_trace_id()
        self.started_at = _now()
        self.fingerprint = runtime_fingerprint()
        self.spans: list[TraceSpan] = []

    def span(self, operation: str, **metadata: Any) -> TraceSpan:
        span = TraceSpan(
            trace_id=self.trace_id,
            operation=operation,
            started_at=_now(),
            metadata=metadata,
        )
        self.spans.append(span)
        return span

    def __enter__(self) -> "TraceContext":
        return self

    def __exit__(self, *exc) -> None:
        for span in self.spans:
            if span.finished_at is None:
                span.finish()
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "span_count": len(self.spans),
            "fingerprint": self.fingerprint.model_dump(),
            "spans": [span.model_dump() for span in self.spans],
        }


def timed_ms() -> "tuple[float, float]":
    """Return a (started_at_iso, latency_ms) pair around a caller's work.

    Convenience for recording latency without a full span.
    """
    started = time.perf_counter()
    started_iso = _now()

    def done() -> float:
        return round((time.perf_counter() - started) * 1000, 4)

    return started_iso, done()


__all__ = [
    "TRACE_SCHEMA",
    "TRACE_SCHEMA_VERSION",
    "RuntimeFingerprint",
    "ServingRecord",
    "TraceContext",
    "TraceSpan",
    "code_revision",
    "new_trace_id",
    "runtime_fingerprint",
    "timed_ms",
]
