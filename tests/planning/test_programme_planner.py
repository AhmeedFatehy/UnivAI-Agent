"""Programme planning: overlap, prerequisites, workload and semester decisions.

Every assertion here is about a decision being *explained and evidenced*, not
just made. The planner is fully deterministic, so these run without an LLM.
"""

from __future__ import annotations

import pytest

from document_processing.metadata import SourceLocation
from planning.overlap import (
    Topic,
    TopicEvidence,
    find_overlaps,
    jaccard,
    merge_overlapping,
    topic_terms,
)
from planning.prerequisites import (
    build_graph,
    dependency_depth,
    teaching_order,
    validate_prerequisites,
)
from planning.programme_planner import ProgrammePlan, create_programme_plan
from planning.workload import (
    estimate_all,
    estimate_topic,
    pack_semesters,
    required_semesters,
)
from tests.conftest import COLLECTION_ID

ALGORITHMS = "6ce579d1-algorithms"
DATABASES = "c45826ed-databases"
LEARNING = "9a1b2c3d-learning"

BOOK_TITLES = {
    ALGORITHMS: "Foundations of Algorithms",
    DATABASES: "Database Systems",
    LEARNING: "Machine Learning Basics",
}


def location(document_id: str, page: int, section: str) -> SourceLocation:
    return SourceLocation(
        collection_id=COLLECTION_ID,
        document_id=document_id,
        book_title=BOOK_TITLES[document_id],
        page=page,
        section=section,
        chunk_index=page - 1,
        source_filename=f"{document_id}.md",
        page_is_estimated=True,
    )


def topic(
    topic_id: str,
    title: str,
    summary: str,
    *,
    document_id: str,
    pages: list[int],
    section: str,
    keywords: list[str] | None = None,
    prerequisites: list[str] | None = None,
    difficulty: int = 3,
    quote: str | None = None,
) -> Topic:
    return Topic(
        topic_id=topic_id,
        title=title,
        summary=summary,
        keywords=keywords or [],
        prerequisites=prerequisites or [],
        difficulty=difficulty,
        evidence=[
            TopicEvidence(
                quote=quote or summary,
                location=location(document_id, page, section),
            )
            for page in pages
        ],
    )


@pytest.fixture
def topics() -> list[Topic]:
    """A small programme spanning all three books, with a real prerequisite chain."""
    return [
        topic(
            "T01",
            "Asymptotic Analysis",
            "Describing running time as growth in input size using big-O notation.",
            document_id=ALGORITHMS,
            pages=[1, 2],
            section="Asymptotic Analysis",
            keywords=["big-o", "growth", "complexity"],
            difficulty=2,
        ),
        topic(
            "T02",
            "Sorting Algorithms",
            "Comparison sorts, the n log n bound, merge sort and quicksort.",
            document_id=ALGORITHMS,
            pages=[2, 3],
            section="Sorting Algorithms",
            keywords=["merge sort", "quicksort", "pivot"],
            prerequisites=["T01"],
            difficulty=3,
        ),
        topic(
            "T03",
            "Hash Tables",
            "Bucket arrays, hash functions, collision handling by chaining or probing.",
            document_id=ALGORITHMS,
            pages=[3, 4],
            section="Hash Tables",
            keywords=["hashing", "collision", "chaining", "load factor"],
            difficulty=3,
        ),
        topic(
            "T04",
            "Relational Model",
            "Relations as sets of tuples, candidate keys and primary keys.",
            document_id=DATABASES,
            pages=[1],
            section="The Relational Model",
            keywords=["relation", "tuple", "primary key"],
            difficulty=2,
        ),
        topic(
            "T05",
            "Normalisation",
            "Decomposing relations along functional dependencies to remove redundancy.",
            document_id=DATABASES,
            pages=[2],
            section="Normalisation",
            keywords=["functional dependency", "third normal form"],
            prerequisites=["T04"],
            difficulty=4,
        ),
        topic(
            "T06",
            "Supervised Learning",
            "Fitting a function from labelled examples and judging it on held-out data.",
            document_id=LEARNING,
            pages=[1, 2],
            section="Supervised Learning",
            keywords=["labels", "training", "validation"],
            difficulty=2,
        ),
        topic(
            "T07",
            "Overfitting and Regularisation",
            "Capturing training noise, and constraining capacity with L2, dropout and early stopping.",
            document_id=LEARNING,
            pages=[2, 3],
            section="Overfitting and Regularisation",
            keywords=["overfitting", "regularisation", "dropout"],
            prerequisites=["T06"],
            difficulty=4,
        ),
    ]


@pytest.fixture
def near_duplicate_topics() -> list[Topic]:
    """The same material taught by two different books."""
    shared = (
        "A hash index stores entries in buckets chosen by a hash function, giving "
        "expected constant time equality lookup. Collisions are handled by chaining "
        "or by probing, and the structure is rebuilt once the load factor rises."
    )
    return [
        topic(
            "T01",
            "Hash Tables",
            shared,
            document_id=ALGORITHMS,
            pages=[3],
            section="Hash Tables",
            keywords=["hashing", "collision", "chaining", "probing", "load factor"],
            quote=shared,
        ),
        topic(
            "T02",
            "Hash Indexes",
            shared,
            document_id=DATABASES,
            pages=[2],
            section="Hash Indexes",
            keywords=["hashing", "collision", "chaining", "probing", "load factor"],
            quote=shared,
        ),
        topic(
            "T03",
            "Transactions and Isolation",
            "Atomicity and durability through a write-ahead log; isolation levels.",
            document_id=DATABASES,
            pages=[4],
            section="Transactions and Isolation",
            keywords=["atomicity", "durability", "isolation"],
            prerequisites=["T02"],
        ),
    ]


# ── Overlap ───────────────────────────────────────────────────────────


def test_the_same_material_in_two_books_is_detected_as_overlap(near_duplicate_topics):
    overlaps = find_overlaps(near_duplicate_topics)
    merge = [item for item in overlaps if item.decision == "merge"]

    assert merge, "near-identical topics from two books must be flagged"
    pair = merge[0]
    assert {pair.topic_a, pair.topic_b} == {"T01", "T02"}
    assert pair.similarity > 0.55
    assert "collision" in pair.shared_terms
    assert set(pair.shared_terms) <= topic_terms(near_duplicate_topics[0])
    assert len(pair.evidence) == 2
    assert {item.document_id for item in pair.evidence} == {ALGORITHMS, DATABASES}


def test_unrelated_topics_are_not_reported_as_overlapping(topics):
    overlaps = find_overlaps(topics)
    pairs = {frozenset((item.topic_a, item.topic_b)) for item in overlaps}
    assert frozenset(("T01", "T06")) not in pairs


def test_merging_keeps_the_evidence_from_both_books(near_duplicate_topics):
    overlaps = find_overlaps(near_duplicate_topics)
    merged, replaced = merge_overlapping(near_duplicate_topics, overlaps)

    assert replaced == {"T02": "T01"}
    assert len(merged) == 2

    keeper = next(item for item in merged if item.topic_id == "T01")
    assert {item.location.document_id for item in keeper.evidence} == {
        ALGORITHMS,
        DATABASES,
    }


def test_a_prerequisite_on_a_merged_topic_is_rewritten_not_dropped(near_duplicate_topics):
    overlaps = find_overlaps(near_duplicate_topics)
    merged, _ = merge_overlapping(near_duplicate_topics, overlaps)

    transactions = next(item for item in merged if item.topic_id == "T03")
    assert transactions.prerequisites == ["T01"], (
        "T02 was folded into T01; the dependency must follow it"
    )


def test_jaccard_is_symmetric_and_bounded():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_topic_terms_drop_stopwords(topics):
    terms = topic_terms(topics[0])
    assert "the" not in terms
    assert "asymptotic" in terms


# ── Prerequisites ─────────────────────────────────────────────────────


def test_teaching_order_puts_prerequisites_first(topics):
    order, issues = teaching_order(topics)

    assert not [issue for issue in issues if issue.kind == "cycle"]
    assert order.index("T01") < order.index("T02")
    assert order.index("T04") < order.index("T05")
    assert order.index("T06") < order.index("T07")
    assert len(order) == len(topics)


def test_teaching_order_is_stable(topics):
    assert teaching_order(topics)[0] == teaching_order(topics)[0]


def test_an_unknown_prerequisite_is_reported_and_dropped(topics):
    topics[1] = topics[1].model_copy(update={"prerequisites": ["T01", "T99"]})

    issues = validate_prerequisites(topics)
    unknown = [issue for issue in issues if issue.kind == "unknown_prerequisite"]

    assert len(unknown) == 1
    assert "T99" in unknown[0].detail
    assert build_graph(topics)["T02"] == ["T01"]


def test_a_self_reference_is_reported_and_ignored(topics):
    topics[0] = topics[0].model_copy(update={"prerequisites": ["T01"]})
    issues = validate_prerequisites(topics)

    assert [issue.kind for issue in issues if issue.kind == "self_reference"] == [
        "self_reference"
    ]
    assert build_graph(topics)["T01"] == []


def test_a_cycle_is_reported_and_broken_rather_than_hanging(topics):
    topics[0] = topics[0].model_copy(update={"prerequisites": ["T02"]})

    order, issues = teaching_order(topics)
    cycles = [issue for issue in issues if issue.kind == "cycle"]

    assert cycles, "a circular prerequisite must be reported"
    assert set(cycles[0].topics) == {"T01", "T02"}
    assert len(order) == len(topics), "every topic is still scheduled"
    repaired = build_graph(topics)
    assert not (repaired["T01"] and repaired["T02"]), (
        "one real edge must be removed; reporting a cycle is not enough"
    )


def test_dependency_depth_counts_the_chain(topics):
    depth = dependency_depth(topics)
    assert depth["T01"] == 0
    assert depth["T02"] == 1
    assert depth["T05"] == 1


# ── Workload ──────────────────────────────────────────────────────────


def test_workload_scales_with_pages_and_difficulty(topics):
    easy = estimate_topic(topics[0])  # 2 pages, difficulty 2
    hard = estimate_topic(topics[6])  # 2 pages, difficulty 4

    assert hard.contact_hours > easy.contact_hours
    assert easy.total_hours == pytest.approx(
        easy.contact_hours + easy.self_study_hours
    )
    assert "difficulty 2" in easy.basis
    assert str(easy.pages) in easy.basis


def test_required_semesters_follows_from_hours(topics):
    estimates = estimate_all(topics)
    total = sum(item.total_hours for item in estimates.values())

    assert required_semesters(estimates, capacity_hours=total * 2) == 1
    assert required_semesters(estimates, capacity_hours=total / 3) >= 3


def test_packing_never_puts_a_topic_before_its_prerequisite(topics):
    estimates = estimate_all(topics)
    order, _ = teaching_order(topics)
    graph = build_graph(topics)

    packed = pack_semesters(order, estimates, graph, capacity_hours=5.0, max_semesters=8)
    placed = {
        topic_id: index for index, semester in enumerate(packed) for topic_id in semester
    }

    for node, parents in graph.items():
        for parent in parents:
            assert placed[parent] < placed[node], f"{node} must follow {parent}"


def test_packing_respects_the_semester_cap(topics):
    estimates = estimate_all(topics)
    order, _ = teaching_order(topics)
    packed = pack_semesters(
        order, estimates, build_graph(topics), capacity_hours=1.0, max_semesters=2
    )
    assert len(packed) == 2, "overflow is kept, not dropped, once the cap is reached"
    assert sum(len(semester) for semester in packed) == len(topics)


def test_related_topics_never_override_a_prerequisite_boundary(topics):
    estimates = estimate_all(topics)
    order, _ = teaching_order(topics)
    graph = build_graph(topics)
    packed = pack_semesters(
        order,
        estimates,
        graph,
        capacity_hours=1000.0,
        keep_together=[("T01", "T02")],
    )
    placed = {
        topic_id: index for index, semester in enumerate(packed) for topic_id in semester
    }
    assert placed["T01"] < placed["T02"]


# ── The plan itself ───────────────────────────────────────────────────


def test_independent_topics_under_a_generous_capacity_fit_one_semester(topics):
    """Nothing depends on anything and the hours fit, so nothing forces a split."""
    independent = [
        item.model_copy(update={"prerequisites": []})
        for item in topics
        if item.topic_id in {"T01", "T03", "T04", "T06"}
    ]

    plan = create_programme_plan(
        independent,
        programme_title="Computer Science Foundations",
        collection_id=COLLECTION_ID,
        capacity_hours=1000.0,
    )

    assert isinstance(plan, ProgrammePlan)
    assert len(plan.semesters) == 1
    assert plan.is_multi_semester is False

    decision = next(item for item in plan.decisions if item.kind == "semester")
    assert "one semester" in decision.rationale
    assert "no prerequisite forces a split" in decision.rationale
    assert decision.evidence


def test_a_prerequisite_chain_splits_semesters_even_with_room_to_spare(topics):
    """A prerequisite taught alongside its dependent gives no time to learn it."""
    plan = create_programme_plan(
        topics,
        programme_title="Computer Science Foundations",
        collection_id=COLLECTION_ID,
        capacity_hours=1000.0,
    )

    assert plan.is_multi_semester
    placed = {
        planned.topic_id: semester.index
        for semester in plan.semesters
        for planned in semester.topics
    }
    assert placed["T01"] < placed["T02"]
    assert placed["T04"] < placed["T05"]
    assert placed["T06"] < placed["T07"]


def test_a_tight_capacity_yields_several_semesters_and_says_why(topics):
    plan = create_programme_plan(
        topics,
        programme_title="Computer Science Foundations",
        collection_id=COLLECTION_ID,
        capacity_hours=6.0,
    )

    assert plan.is_multi_semester
    decision = next(item for item in plan.decisions if item.kind == "semester")
    assert f"{len(plan.semesters)} semesters" in decision.rationale
    assert "capacity" in decision.rationale
    assert decision.evidence


def test_every_decision_carries_checkable_evidence(topics):
    plan = create_programme_plan(
        topics,
        programme_title="Computer Science Foundations",
        collection_id=COLLECTION_ID,
        capacity_hours=6.0,
    )

    assert plan.decisions
    for decision in plan.decisions:
        assert decision.evidence, f"{decision.decision_id} has no evidence"
        for citation in decision.evidence:
            assert citation.document_id in BOOK_TITLES
            assert citation.book_title
            assert citation.page >= 1
            assert citation.section


def test_the_plan_covers_the_expected_decision_kinds(topics):
    plan = create_programme_plan(
        topics,
        programme_title="Computer Science Foundations",
        collection_id=COLLECTION_ID,
        capacity_hours=6.0,
    )
    kinds = {decision.kind for decision in plan.decisions}
    assert {"prerequisite", "workload", "semester", "coverage"} <= kinds


def test_every_scheduled_topic_is_cited_and_costed(topics):
    plan = create_programme_plan(
        topics,
        programme_title="Computer Science Foundations",
        collection_id=COLLECTION_ID,
        capacity_hours=20.0,
    )

    positions = []
    for semester in plan.semesters:
        for planned in semester.topics:
            assert planned.citations
            assert planned.total_hours > 0
            assert planned.workload_basis
            positions.append(planned.order)

    assert positions == sorted(positions)
    assert len(positions) == plan.topics_scheduled


def test_the_plan_names_all_three_source_books(topics):
    plan = create_programme_plan(
        topics,
        programme_title="Computer Science Foundations",
        collection_id=COLLECTION_ID,
    )
    assert set(plan.source_documents) == {ALGORITHMS, DATABASES, LEARNING}

    coverage = next(item for item in plan.decisions if item.kind == "coverage")
    for title in BOOK_TITLES.values():
        assert title in coverage.rationale


def test_merged_duplicates_are_reported_in_the_plan(near_duplicate_topics):
    plan = create_programme_plan(
        near_duplicate_topics,
        programme_title="Storage Structures",
        collection_id=COLLECTION_ID,
    )

    assert plan.topics_scheduled < plan.topics_considered
    overlap_decisions = [item for item in plan.decisions if item.kind == "overlap"]
    assert overlap_decisions
    assert any("folded" in item.summary for item in overlap_decisions)
    assert any("keeps the evidence from both books" in item.rationale for item in overlap_decisions)


def test_planning_with_no_topics_is_refused_not_faked():
    with pytest.raises(ValueError, match="no evidence-backed topics"):
        create_programme_plan(
            [], programme_title="Empty", collection_id=COLLECTION_ID
        )


def test_a_topic_cannot_be_built_without_evidence():
    with pytest.raises(ValueError):
        Topic(
            topic_id="T01",
            title="Uncited",
            summary="No source backs this.",
            evidence=[],
        )
