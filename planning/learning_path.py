"""Deterministic, evidence-backed ordering of complete books.

Model output is only a proposal.  This module resolves proposals against the
approved inventories, validates their evidence and builds the serial artifact
consumed by the App.  It never repairs a cycle by silently deleting an edge.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from document_processing.metadata import SourceLocation
from planning.semester_planner import ChapterInventory, SemesterWeekPlan, plan_semester

LEARNING_PATH_SCHEMA = "univai.learning-path"
LEARNING_PATH_VERSION = "1.0.0"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70


class ApprovalState(str, Enum):
    PENDING = "pending_exact_plan_approval"
    BLOCKED = "blocked_human_decision"
    APPROVED = "approved"


class ApprovedBook(BaseModel):
    book_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    edition: str | None = None
    inventory: ChapterInventory

    @model_validator(mode="after")
    def _title_matches_inventory(self) -> "ApprovedBook":
        if self.title != self.inventory.book_title:
            raise ValueError("approved book title must match its chapter inventory")
        return self


class EvidenceExcerpt(BaseModel):
    quote: str = Field(min_length=1)
    citation: SourceLocation


class PrerequisiteProposal(BaseModel):
    """Untrusted model proposal, kept distinct from the validated edge."""

    prerequisite_book_id: str = Field(min_length=1)
    dependent_book_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    ordering_is_ambiguous: bool = False
    prerequisite_evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    dependent_evidence: list[EvidenceExcerpt] = Field(default_factory=list)


class PrerequisiteEdge(PrerequisiteProposal):
    prerequisite_evidence: list[EvidenceExcerpt] = Field(min_length=1)
    dependent_evidence: list[EvidenceExcerpt] = Field(min_length=1)


class PrerequisiteAnalysisDraft(BaseModel):
    """Named schema for the complete AI-produced proposal batch."""

    proposals: list[PrerequisiteProposal] = Field(default_factory=list)


class LearningPathWarning(BaseModel):
    warning_id: str = Field(min_length=1)
    kind: Literal[
        "cycle",
        "low_confidence",
        "missing_evidence",
        "ambiguous_ordering",
        "duplicate_edition",
        "overlap",
        "disconnected_books",
        "invalid_reference",
    ]
    message: str = Field(min_length=1)
    book_ids: list[str] = Field(min_length=1)
    evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    alternatives: list[list[str]] = Field(default_factory=list)
    blocks_approval: bool = True


class OrderedBook(BaseModel):
    position: int = Field(ge=1)
    book_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    edition: str | None = None
    starts_at_global_week: int = Field(ge=1)
    ends_at_global_week: int = Field(ge=1)
    week_plan: SemesterWeekPlan


class LearningPathV1(BaseModel):
    schema_name: str = LEARNING_PATH_SCHEMA
    schema_version: str = LEARNING_PATH_VERSION
    collection_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    approved_book_ids: list[str] = Field(min_length=1)
    ordered_books: list[OrderedBook] = Field(min_length=1)
    prerequisite_edges: list[PrerequisiteEdge] = Field(default_factory=list)
    warnings: list[LearningPathWarning] = Field(default_factory=list)
    approval_state: ApprovalState
    approved_warning_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _contract_invariants(self) -> "LearningPathV1":
        actual = [book.book_id for book in self.ordered_books]
        if len(actual) != len(set(actual)) or set(actual) != set(self.approved_book_ids):
            raise ValueError("every approved book must appear exactly once")
        if [book.position for book in self.ordered_books] != list(range(1, len(actual) + 1)):
            raise ValueError("book positions must be contiguous")
        previous_end = 0
        for book in self.ordered_books:
            if book.starts_at_global_week != previous_end + 1:
                raise ValueError("books must form one serial, gap-free path")
            if book.week_plan.weeks[0].week != 1:
                raise ValueError("each book must restart at its own week 1")
            previous_end = book.ends_at_global_week
        if not any(warning.kind == "cycle" for warning in self.warnings):
            positions = {book_id: index for index, book_id in enumerate(actual)}
            for edge in self.prerequisite_edges:
                if positions[edge.prerequisite_book_id] >= positions[edge.dependent_book_id]:
                    raise ValueError("a dependent book cannot precede its prerequisite")
        if self.approval_state is ApprovalState.APPROVED:
            unresolved = {w.warning_id for w in self.warnings if w.blocks_approval} - set(self.approved_warning_ids)
            if unresolved:
                raise ValueError(f"approval requires explicit warning overrides: {sorted(unresolved)}")
        return self


def generate_learning_path(
    books: list[ApprovedBook],
    proposals: list[PrerequisiteProposal],
    *,
    collection_id: str,
    user_id: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> LearningPathV1:
    """Validate proposals and produce a stable, reviewable serial plan."""
    if not books:
        raise ValueError("at least one approved book is required")
    if len({book.book_id for book in books}) != len(books):
        raise ValueError("approved book IDs must be unique")
    if any(
        excerpt.citation.collection_id != collection_id
        for proposal in proposals
        for excerpt in [*proposal.prerequisite_evidence, *proposal.dependent_evidence]
    ):
        raise ValueError("prerequisite evidence crosses the requested collection")

    by_id = {book.book_id: book for book in books}
    warnings: list[LearningPathWarning] = []
    edges: list[PrerequisiteEdge] = []
    for index, proposal in enumerate(proposals, start=1):
        pair = [proposal.prerequisite_book_id, proposal.dependent_book_id]
        if any(book_id not in by_id for book_id in pair) or pair[0] == pair[1]:
            warnings.append(_warning(index, "invalid_reference", pair, "Proposal references an unknown book or itself."))
            continue
        expected_documents = (by_id[pair[0]].document_id, by_id[pair[1]].document_id)
        evidence_ok = (
            proposal.prerequisite_evidence
            and proposal.dependent_evidence
            and all(item.citation.document_id == expected_documents[0] for item in proposal.prerequisite_evidence)
            and all(item.citation.document_id == expected_documents[1] for item in proposal.dependent_evidence)
            and all(
                item.citation.page is not None and item.citation.section
                for item in [*proposal.prerequisite_evidence, *proposal.dependent_evidence]
            )
        )
        if not evidence_ok:
            warnings.append(_warning(index, "missing_evidence", pair, "Edge lacks resolvable evidence from both books.", proposal))
            continue
        edge = PrerequisiteEdge.model_validate(proposal.model_dump())
        edges.append(edge)
        if edge.confidence < confidence_threshold:
            warnings.append(_warning(index, "low_confidence", pair, f"Edge confidence {edge.confidence:.2f} is below {confidence_threshold:.2f}.", proposal))
        if edge.ordering_is_ambiguous:
            warnings.append(_warning(index, "ambiguous_ordering", pair, "Evidence supports more than one ordering; human selection is required.", proposal))

    order, cyclic = _topological_order([book.book_id for book in books], edges)
    if cyclic:
        alternatives = _cycle_alternatives([book.book_id for book in books], edges, cyclic)
        warnings.append(LearningPathWarning(
            warning_id="cycle-01", kind="cycle", book_ids=cyclic,
            message="Prerequisite cycle blocks deterministic publication; choose an evidence-backed edge to override.",
            evidence=_edge_evidence(edges, cyclic), alternatives=alternatives,
        ))
        order = [book.book_id for book in books]

    connected = {value for edge in edges for value in (edge.prerequisite_book_id, edge.dependent_book_id)}
    disconnected = [book.book_id for book in books if book.book_id not in connected]
    if proposals and disconnected:
        warnings.append(LearningPathWarning(
            warning_id="disconnected-01", kind="disconnected_books", book_ids=disconnected,
            message="Some books have no supported relationship; their stable upload order needs human review.",
            alternatives=[order], blocks_approval=True,
        ))

    duplicate_groups: dict[tuple[str, str | None], list[str]] = {}
    for book in books:
        duplicate_groups.setdefault((book.title.casefold(), book.edition), []).append(book.book_id)
    for index, ids in enumerate((ids for ids in duplicate_groups.values() if len(ids) > 1), start=1):
        warnings.append(LearningPathWarning(
            warning_id=f"duplicate-{index:02d}", kind="duplicate_edition", book_ids=ids,
            message="Duplicate title and edition require a human choice.", alternatives=[[item] for item in ids],
        ))

    chapter_titles = {
        book.book_id: {chapter.title.strip().casefold() for chapter in book.inventory.chapters}
        for book in books
    }
    overlap_index = 0
    for left_index, left in enumerate(books):
        for right in books[left_index + 1 :]:
            shared = sorted(chapter_titles[left.book_id] & chapter_titles[right.book_id])
            if not shared:
                continue
            overlap_index += 1
            warnings.append(LearningPathWarning(
                warning_id=f"overlap-{overlap_index:02d}", kind="overlap",
                book_ids=[left.book_id, right.book_id],
                message=f"Books overlap on chapter titles: {', '.join(shared)}; retain both or choose an edition.",
                alternatives=[[left.book_id, right.book_id], [left.book_id], [right.book_id]],
            ))

    ordered: list[OrderedBook] = []
    global_week = 1
    for position, book_id in enumerate(order, start=1):
        book = by_id[book_id]
        week_plan = plan_semester(book.inventory).validate_against(book.inventory)
        ordered.append(OrderedBook(
            position=position, book_id=book.book_id, document_id=book.document_id,
            title=book.title, edition=book.edition, starts_at_global_week=global_week,
            ends_at_global_week=global_week + week_plan.week_count - 1, week_plan=week_plan,
        ))
        global_week += week_plan.week_count

    return LearningPathV1(
        collection_id=collection_id, user_id=user_id,
        approved_book_ids=[book.book_id for book in books], ordered_books=ordered,
        prerequisite_edges=edges, warnings=warnings,
        approval_state=ApprovalState.BLOCKED if any(w.blocks_approval for w in warnings) else ApprovalState.PENDING,
    )


def approve_learning_path(plan: LearningPathV1, *, schema_version: str, warning_overrides: list[str] | None = None) -> LearningPathV1:
    """Record exact-plan approval; callers must name its version and warnings."""
    if schema_version != plan.schema_version:
        raise ValueError("approval schema version does not match the exact plan")
    overrides = sorted(set(warning_overrides or []))
    required = {warning.warning_id for warning in plan.warnings if warning.blocks_approval}
    if not required.issubset(overrides):
        raise ValueError(f"unresolved warnings: {sorted(required - set(overrides))}")
    return plan.model_copy(update={"approval_state": ApprovalState.APPROVED, "approved_warning_ids": overrides})


def _warning(index: int, kind: str, ids: list[str], message: str, proposal: PrerequisiteProposal | None = None) -> LearningPathWarning:
    return LearningPathWarning(
        warning_id=f"{kind}-{index:02d}", kind=kind, book_ids=ids, message=message,
        evidence=[] if proposal is None else [*proposal.prerequisite_evidence, *proposal.dependent_evidence],
        alternatives=[ids, list(reversed(ids))] if len(ids) == 2 else [ids],
    )


def _topological_order(book_ids: list[str], edges: list[PrerequisiteEdge]) -> tuple[list[str], list[str]]:
    rank = {book_id: index for index, book_id in enumerate(book_ids)}
    parents = {book_id: set() for book_id in book_ids}
    for edge in edges:
        parents[edge.dependent_book_id].add(edge.prerequisite_book_id)
    remaining = {node: set(values) for node, values in parents.items()}
    result: list[str] = []
    while remaining:
        ready = sorted((node for node, values in remaining.items() if not values), key=rank.get)
        if not ready:
            return result + sorted(remaining, key=rank.get), sorted(remaining, key=rank.get)
        result.extend(ready)
        for node in ready:
            del remaining[node]
        for values in remaining.values():
            values.difference_update(ready)
    return result, []


def _edge_evidence(edges: list[PrerequisiteEdge], ids: list[str]) -> list[EvidenceExcerpt]:
    selected: list[EvidenceExcerpt] = []
    scope = set(ids)
    for edge in edges:
        if {edge.prerequisite_book_id, edge.dependent_book_id} <= scope:
            selected.extend([*edge.prerequisite_evidence, *edge.dependent_evidence])
    return selected


def _cycle_alternatives(book_ids: list[str], edges: list[PrerequisiteEdge], cyclic: list[str]) -> list[list[str]]:
    """Show each valid ordering obtained by explicitly overriding one cycle edge."""
    scope = set(cyclic)
    alternatives: list[list[str]] = []
    for removed in edges:
        if {removed.prerequisite_book_id, removed.dependent_book_id} <= scope:
            candidate, still_cyclic = _topological_order(
                book_ids, [edge for edge in edges if edge is not removed]
            )
            if not still_cyclic and candidate not in alternatives:
                alternatives.append(candidate)
    return alternatives or [book_ids]


__all__ = [
    "LEARNING_PATH_SCHEMA", "LEARNING_PATH_VERSION", "ApprovalState", "ApprovedBook",
    "EvidenceExcerpt", "LearningPathV1", "LearningPathWarning", "OrderedBook",
    "PrerequisiteAnalysisDraft", "PrerequisiteEdge", "PrerequisiteProposal",
    "approve_learning_path", "generate_learning_path",
]
