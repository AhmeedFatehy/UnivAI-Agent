from __future__ import annotations

import pytest

from document_processing.metadata import SourceLocation
from planning.learning_path import ApprovalState, ApprovedBook, EvidenceExcerpt, PrerequisiteProposal, approve_learning_path, generate_learning_path
from planning.semester_planner import Chapter, ChapterInventory

COLLECTION = "collection-1"
USER = "user-1"


def book(book_id: str, *titles: str, edition: str | None = None) -> ApprovedBook:
    chapters = [Chapter(chapter_id=f"C{index:03d}", title=title, start_page=index * 10 - 9, end_page=index * 10, source_ids=[f"{book_id}-S{index}"]) for index, title in enumerate(titles or (f"{book_id} chapter 1",), start=1)]
    return ApprovedBook(book_id=book_id, document_id=f"doc-{book_id}", title=f"Book {book_id}", edition=edition, inventory=ChapterInventory(book_title=f"Book {book_id}", chapters=chapters))


def excerpt(book_id: str, section: str = "Introduction") -> EvidenceExcerpt:
    return EvidenceExcerpt(quote=f"Evidence in {book_id}", citation=SourceLocation(collection_id=COLLECTION, document_id=f"doc-{book_id}", book_title=f"Book {book_id}", page=1, section=section))


def edge(parent: str, child: str, confidence: float = 0.95) -> PrerequisiteProposal:
    return PrerequisiteProposal(prerequisite_book_id=parent, dependent_book_id=child, rationale=f"{child} explicitly assumes {parent}", confidence=confidence, prerequisite_evidence=[excerpt(parent, "Foundations")], dependent_evidence=[excerpt(child, "Assumed knowledge")])


def generate(books, edges):
    return generate_learning_path(books, edges, collection_id=COLLECTION, user_id=USER)


def test_chain_is_serial_and_each_book_restarts_at_chapter_one():
    plan = generate([book("C", "C one"), book("A", "A one", "A two"), book("B", "B one")], [edge("A", "B"), edge("B", "C")])
    assert [item.book_id for item in plan.ordered_books] == ["A", "B", "C"]
    a, b, _ = plan.ordered_books
    assert a.ends_at_global_week + 1 == b.starts_at_global_week
    assert b.week_plan.weeks[0].week == 1
    assert b.week_plan.weeks[0].chapters[0].chapter_id == "C001"
    assert plan.approval_state is ApprovalState.PENDING


def test_diamond_is_deterministic_and_contains_every_book_once():
    books = [book(item) for item in ("D", "C", "B", "A")]
    proposals = [edge("A", "B"), edge("A", "C"), edge("B", "D"), edge("C", "D")]
    first, second = generate(books, proposals), generate(books, proposals)
    assert [item.book_id for item in first.ordered_books] == ["A", "C", "B", "D"]
    assert first == second


def test_cycle_blocks_publication_and_retains_evidence_backed_alternatives():
    plan = generate([book("A"), book("B"), book("C")], [edge("A", "B"), edge("B", "C"), edge("C", "A")])
    warning = next(item for item in plan.warnings if item.kind == "cycle")
    assert plan.approval_state is ApprovalState.BLOCKED
    assert warning.evidence and warning.alternatives
    with pytest.raises(ValueError, match="unresolved warnings"):
        approve_learning_path(plan, schema_version="1.0.0")


def test_low_confidence_and_ambiguous_evidence_require_human_decision():
    ambiguous = edge("B", "C", 0.64).model_copy(update={"ordering_is_ambiguous": True})
    plan = generate([book("A"), book("B"), book("C")], [edge("A", "C", 0.65), ambiguous])
    assert {warning.kind for warning in plan.warnings} >= {"low_confidence", "ambiguous_ordering"}
    assert plan.approval_state is ApprovalState.BLOCKED


def test_overlap_is_not_silently_merged():
    plan = generate([book("A", "Shared", edition="2"), book("B", "Shared", edition="2")], [])
    assert any(warning.kind == "overlap" for warning in plan.warnings)
    assert len(plan.ordered_books) == 2


def test_missing_evidence_never_becomes_an_edge():
    proposal = edge("A", "B").model_copy(update={"dependent_evidence": []})
    plan = generate([book("A"), book("B")], [proposal])
    assert not plan.prerequisite_edges
    assert any(warning.kind == "missing_evidence" for warning in plan.warnings)


def test_no_prerequisites_preserve_stable_user_reviewable_order():
    plan = generate([book("C"), book("A"), book("B")], [])
    assert [item.book_id for item in plan.ordered_books] == ["C", "A", "B"]
    assert plan.approval_state is ApprovalState.PENDING


def test_tenant_isolation_rejects_cross_collection_evidence():
    proposal = edge("A", "B")
    foreign = proposal.dependent_evidence[0].citation.model_copy(update={"collection_id": "other"})
    proposal = proposal.model_copy(update={"dependent_evidence": [proposal.dependent_evidence[0].model_copy(update={"citation": foreign})]})
    with pytest.raises(ValueError, match="crosses the requested collection"):
        generate([book("A"), book("B")], [proposal])


def test_exact_version_and_warning_overrides_are_required_for_approval():
    plan = generate([book("A", "Shared"), book("B", "Shared")], [])
    with pytest.raises(ValueError, match="version"):
        approve_learning_path(plan, schema_version="2.0.0")
    approved = approve_learning_path(plan, schema_version="1.0.0", warning_overrides=[warning.warning_id for warning in plan.warnings])
    assert approved.approval_state is ApprovalState.APPROVED
