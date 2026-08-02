"""Typed tool registry — the only way an agent touches the knowledge base.

Every tool has a Pydantic input model and a Pydantic output model, and
:func:`call_tool` validates both sides. An agent cannot pass a half-formed
argument, and it cannot receive an unvalidated blob back.

The tools:

``ingest_collection``      index several books under one collection identity
``retrieve_context``       grounded hybrid retrieval — passages *with citations*,
                           or an explicit refusal when the evidence is not there
``create_programme_plan``  deterministic, evidence-backed curriculum plan
``get_source_location``    resolve a citation back to its book, page and section

:class:`ToolContext` holds the side-effecting callables, so tests substitute a
fake retriever or indexer and exercise the real validation and grounding logic
without Qdrant, an embedding model or an LLM.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, model_validator

from document_processing.batch_ingestion import (
    BatchIngestionReport,
    IngestionBackend,
)
from document_processing.batch_ingestion import ingest_collection as _run_ingest_collection
from document_processing.metadata import SourceLocation, citation_from_payload
from planning.overlap import Topic, content_terms
from planning.programme_planner import ProgrammePlan, create_programme_plan
from planning.workload import DEFAULT_SEMESTER_CAPACITY_HOURS

logger = logging.getLogger(__name__)

TOOL_SCHEMA_VERSION = "1.0.0"

#: Fraction of the query's content terms a passage must contain to count as
#: evidence. Below this the passage is a lexical near-miss and citing it would
#: dress up an unsupported answer as a grounded one.
DEFAULT_MIN_TERM_COVERAGE = 0.34

REFUSAL_NO_HITS = "The indexed material contains nothing matching this question."
REFUSAL_NO_GROUNDING = (
    "Retrieved passages do not actually cover this question, so there is no "
    "evidence to cite."
)
REFUSAL_UNCITABLE = (
    "Matching passages exist but carry no book/page identity, so they cannot be "
    "cited. Re-ingest the source material."
)
REFUSAL_UNSAFE_SOURCE = (
    "Matching passages contain instruction-like source text and were excluded "
    "from model evidence."
)


# ── Errors ────────────────────────────────────────────────────────────


class ToolError(RuntimeError):
    """Base class for tool-layer failures."""


class ToolNotFound(ToolError):
    pass


class ToolInputError(ToolError):
    pass


class ToolOutputError(ToolError):
    pass


# ── Grounded retrieval contract ───────────────────────────────────────


class GroundedPassage(BaseModel):
    """One piece of evidence an agent is allowed to build an answer on."""

    passage_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    content: str = Field(min_length=1)
    score: float
    term_coverage: float = Field(ge=0.0, le=1.0)
    citation: SourceLocation


class Refusal(BaseModel):
    """An explicit 'I cannot answer this from the sources' — never an empty answer."""

    reason: str = Field(min_length=1)
    query: str
    scope: dict[str, Any] = Field(default_factory=dict)
    candidates_examined: int = Field(default=0, ge=0)


class GroundedContext(BaseModel):
    """Retrieval result: either cited passages, or a refusal. Never both."""

    schema_version: str = TOOL_SCHEMA_VERSION
    query: str
    grounded: bool
    passages: list[GroundedPassage] = Field(default_factory=list)
    refusal: Refusal | None = None

    @model_validator(mode="after")
    def _grounded_xor_refusal(self) -> "GroundedContext":
        if self.grounded and not self.passages:
            raise ValueError("a grounded context must carry at least one passage")
        if self.grounded and self.refusal is not None:
            raise ValueError("a grounded context cannot also carry a refusal")
        if not self.grounded:
            if self.passages:
                raise ValueError("a refusal must not carry passages")
            if self.refusal is None:
                raise ValueError("an ungrounded context must state a refusal reason")
        return self

    @property
    def citations(self) -> list[SourceLocation]:
        return [passage.citation for passage in self.passages]

    def by_id(self) -> dict[str, GroundedPassage]:
        return {passage.passage_id: passage for passage in self.passages}

    def as_prompt_block(self, *, max_chars: int = 1200) -> str:
        """Render the evidence for an LLM prompt, each passage tagged with its id.

        The tag is what the model must cite; it cannot invent a page number
        because it never sees one it could copy without the id attached.
        """
        if not self.grounded:
            reason = self.refusal.reason if self.refusal else "no evidence"
            return f"NO EVIDENCE AVAILABLE: {reason}"
        blocks = []
        for passage in self.passages:
            body = passage.content.strip()
            if len(body) > max_chars:
                body = body[:max_chars].rstrip() + "…"
            blocks.append(
                f"[{passage.passage_id}] {passage.citation.label()}\n{body}"
            )
        return "\n\n".join(blocks)


def term_coverage(query: str, passage: str) -> float:
    """Fraction of the query's content terms that appear in the passage."""
    wanted = content_terms(query)
    if not wanted:
        return 0.0
    return len(wanted & content_terms(passage)) / len(wanted)


# ── Tool input/output models ──────────────────────────────────────────


class IngestCollectionInput(BaseModel):
    paths: list[str] = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    collection_name: str | None = None
    max_attempts: int = Field(default=3, ge=1, le=10)


class RetrieveContextInput(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    collection_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    book_titles: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=50)
    use_reranking: bool = True
    use_query_transform: bool = False
    min_term_coverage: float = Field(default=DEFAULT_MIN_TERM_COVERAGE, ge=0.0, le=1.0)
    min_score: float | None = None


class CreateProgrammePlanInput(BaseModel):
    topics: list[Topic] = Field(min_length=1)
    programme_title: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    capacity_hours: float = Field(default=DEFAULT_SEMESTER_CAPACITY_HOURS, gt=0)
    max_semesters: int = Field(default=8, ge=1, le=16)


class GetSourceLocationInput(BaseModel):
    user_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_index: int | None = Field(default=None, ge=0)
    collection_id: str | None = None


class SourceLocationResult(BaseModel):
    found: bool
    location: SourceLocation | None = None
    excerpt: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _found_needs_location(self) -> "SourceLocationResult":
        if self.found and self.location is None:
            raise ValueError("a found source location must include the location")
        if not self.found and not self.reason:
            raise ValueError("a missing source location must say why")
        return self


# ── Tool context ──────────────────────────────────────────────────────


def _default_retriever(**kwargs) -> list[dict]:
    from retrieval.pipeline import retrieve

    return retrieve(**kwargs)


def _default_locator(*, user_id: str, document_id: str, chunk_index: int | None) -> list[dict]:
    from vector_store.collection_manager import fetch_document_chunks

    return fetch_document_chunks(
        user_id=user_id, document_id=document_id, chunk_index=chunk_index
    )


@dataclass
class ToolContext:
    """Injectable side effects. Defaults are the real Qdrant-backed pipeline."""

    retriever: Callable[..., list[dict]] = field(default=None)  # type: ignore[assignment]
    locator: Callable[..., list[dict]] = field(default=None)  # type: ignore[assignment]
    ingestion_backend: IngestionBackend | None = None
    collection_name: str | None = None

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = _default_retriever
        if self.locator is None:
            self.locator = _default_locator


# ── Tool handlers ─────────────────────────────────────────────────────


def ingest_collection_tool(
    payload: IngestCollectionInput, context: ToolContext
) -> BatchIngestionReport:
    report, _ = _run_ingest_collection(
        payload.paths,
        collection_id=payload.collection_id,
        user_id=payload.user_id,
        backend=context.ingestion_backend,
        collection_name=payload.collection_name or context.collection_name,
        max_attempts=payload.max_attempts,
    )
    return report


def retrieve_context_tool(
    payload: RetrieveContextInput, context: ToolContext
) -> GroundedContext:
    """Retrieve, then decide whether what came back is actually evidence."""
    scope: dict[str, Any] = {
        "user_id": payload.user_id,
        "collection_id": payload.collection_id,
        "document_ids": payload.document_ids,
        "book_titles": payload.book_titles,
    }

    hits = context.retriever(
        query=payload.query,
        user_id=payload.user_id,
        limit=payload.limit,
        use_reranking=payload.use_reranking,
        use_query_transform=payload.use_query_transform,
        collection_name=context.collection_name,
        collection_id=payload.collection_id,
        document_ids=payload.document_ids or None,
        book_titles=payload.book_titles or None,
    )

    if not hits:
        return GroundedContext(
            query=payload.query,
            grounded=False,
            refusal=Refusal(reason=REFUSAL_NO_HITS, query=payload.query, scope=scope),
        )

    passages: list[GroundedPassage] = []
    uncitable = 0
    unsafe = 0
    for hit in hits:
        if hit.get("source_injection_flagged") is True:
            unsafe += 1
            continue
        content = (hit.get("content") or "").strip()
        if not content:
            continue
        citation = citation_from_payload(hit)
        if citation is None:
            uncitable += 1
            continue
        coverage = term_coverage(payload.query, content)
        if coverage < payload.min_term_coverage:
            continue
        score = float(hit.get("score") or 0.0)
        if payload.min_score is not None and score < payload.min_score:
            continue
        passages.append(
            GroundedPassage(
                passage_id=f"S{len(passages) + 1}",
                rank=len(passages) + 1,
                content=content,
                score=score,
                term_coverage=round(coverage, 4),
                citation=citation,
            )
        )
        if len(passages) >= payload.limit:
            break

    if not passages:
        if unsafe == len(hits):
            reason = REFUSAL_UNSAFE_SOURCE
        elif uncitable + unsafe == len(hits) and uncitable:
            reason = REFUSAL_UNCITABLE
        else:
            reason = REFUSAL_NO_GROUNDING
        return GroundedContext(
            query=payload.query,
            grounded=False,
            refusal=Refusal(
                reason=reason,
                query=payload.query,
                scope=scope,
                candidates_examined=len(hits),
            ),
        )

    return GroundedContext(query=payload.query, grounded=True, passages=passages)


def create_programme_plan_tool(
    payload: CreateProgrammePlanInput, context: ToolContext
) -> ProgrammePlan:
    return create_programme_plan(
        payload.topics,
        programme_title=payload.programme_title,
        collection_id=payload.collection_id,
        capacity_hours=payload.capacity_hours,
        max_semesters=payload.max_semesters,
    )


def get_source_location_tool(
    payload: GetSourceLocationInput, context: ToolContext
) -> SourceLocationResult:
    """Resolve a citation back to the exact passage it claims to come from."""
    rows = context.locator(
        user_id=payload.user_id,
        document_id=payload.document_id,
        chunk_index=payload.chunk_index,
    )
    if not rows:
        return SourceLocationResult(
            found=False,
            reason=(
                f"No indexed chunk for document '{payload.document_id}'"
                + (f" at chunk {payload.chunk_index}" if payload.chunk_index is not None else "")
                + f" owned by '{payload.user_id}'."
            ),
        )

    row = rows[0]
    location = citation_from_payload(row)
    if location is None:
        return SourceLocationResult(
            found=False,
            reason=(
                f"Chunk for document '{payload.document_id}' predates the citation schema "
                "and has no book/page identity."
            ),
        )
    if payload.collection_id and location.collection_id != payload.collection_id:
        return SourceLocationResult(
            found=False,
            reason=(
                f"Document '{payload.document_id}' belongs to collection "
                f"'{location.collection_id}', not '{payload.collection_id}'."
            ),
        )

    excerpt = (row.get("content") or row.get("page_content") or "").strip()
    return SourceLocationResult(
        found=True,
        location=location,
        excerpt=excerpt[:600] or None,
    )


# ── Registry ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[Any, ToolContext], BaseModel]

    def json_schema(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
        }


TOOL_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in TOOL_REGISTRY:
        raise ValueError(f"tool '{spec.name}' is already registered")
    TOOL_REGISTRY[spec.name] = spec
    return spec


register(
    ToolSpec(
        name="ingest_collection",
        version=TOOL_SCHEMA_VERSION,
        description=(
            "Index several books into one collection. Each book is ingested "
            "independently: a failure is reported, not fatal, and transient "
            "failures are retried."
        ),
        input_model=IngestCollectionInput,
        output_model=BatchIngestionReport,
        handler=ingest_collection_tool,
    )
)

register(
    ToolSpec(
        name="retrieve_context",
        version=TOOL_SCHEMA_VERSION,
        description=(
            "Hybrid (semantic + keyword) retrieval filtered by user, collection, "
            "document or book. Returns passages with book/page/section citations, "
            "or an explicit refusal when the sources do not cover the question."
        ),
        input_model=RetrieveContextInput,
        output_model=GroundedContext,
        handler=retrieve_context_tool,
    )
)

register(
    ToolSpec(
        name="create_programme_plan",
        version=TOOL_SCHEMA_VERSION,
        description=(
            "Build a semester-by-semester programme plan from evidence-backed "
            "topics, deciding overlap, prerequisites and workload with citations."
        ),
        input_model=CreateProgrammePlanInput,
        output_model=ProgrammePlan,
        handler=create_programme_plan_tool,
    )
)

register(
    ToolSpec(
        name="get_source_location",
        version=TOOL_SCHEMA_VERSION,
        description=(
            "Resolve a document id (and optional chunk index) back to its book, "
            "page and section, with the indexed excerpt, so a citation can be verified."
        ),
        input_model=GetSourceLocationInput,
        output_model=SourceLocationResult,
        handler=get_source_location_tool,
    )
)


def get_tool(name: str) -> ToolSpec:
    try:
        return TOOL_REGISTRY[name]
    except KeyError as error:
        known = ", ".join(sorted(TOOL_REGISTRY))
        raise ToolNotFound(f"unknown tool '{name}'; available: {known}") from error


def call_tool(
    name: str,
    payload: dict | BaseModel,
    context: ToolContext | None = None,
) -> BaseModel:
    """Validate the input, run the tool, validate the output."""
    spec = get_tool(name)
    context = context or ToolContext()

    try:
        validated = (
            payload
            if isinstance(payload, spec.input_model)
            else spec.input_model.model_validate(
                payload.model_dump() if isinstance(payload, BaseModel) else payload
            )
        )
    except Exception as error:  # noqa: BLE001 — re-raised as a typed tool error
        raise ToolInputError(f"invalid input for '{name}': {error}") from error

    result = spec.handler(validated, context)

    if not isinstance(result, spec.output_model):
        try:
            result = spec.output_model.model_validate(
                result.model_dump() if isinstance(result, BaseModel) else result
            )
        except Exception as error:  # noqa: BLE001 — re-raised as a typed tool error
            raise ToolOutputError(f"invalid output from '{name}': {error}") from error

    logger.info("tool %s v%s completed", spec.name, spec.version)
    return result


def tool_manifest() -> list[dict]:
    """Machine-readable description of every tool, for agent binding."""
    return [spec.json_schema() for spec in TOOL_REGISTRY.values()]


__all__ = [
    "DEFAULT_MIN_TERM_COVERAGE",
    "REFUSAL_NO_GROUNDING",
    "REFUSAL_NO_HITS",
    "REFUSAL_UNCITABLE",
    "TOOL_REGISTRY",
    "TOOL_SCHEMA_VERSION",
    "CreateProgrammePlanInput",
    "GetSourceLocationInput",
    "GroundedContext",
    "GroundedPassage",
    "IngestCollectionInput",
    "Refusal",
    "RetrieveContextInput",
    "SourceLocationResult",
    "ToolContext",
    "ToolError",
    "ToolInputError",
    "ToolNotFound",
    "ToolOutputError",
    "ToolSpec",
    "call_tool",
    "get_tool",
    "register",
    "term_coverage",
    "tool_manifest",
]
