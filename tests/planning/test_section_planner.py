"""Deterministic section-pack rules: evidence, budget, uniqueness, refusals.

Every invariant is asserted against a hand-built grounded context, so a reviewer
can re-derive each expected value without a fake vector store or a model.
"""

from __future__ import annotations

import pytest

from agents.schemas import (
    SectionActivityDraftLLM,
    SectionDraftLLM,
    SectionExampleDraftLLM,
    SectionStepDraftLLM,
    SectionTodoDraftLLM,
)
from document_processing.metadata import SourceLocation
from planning.section_planner import (
    DEFAULT_SECTION_BUDGET,
    SectionBudget,
    SectionIdentity,
    SectionPlanError,
    build_section_pack,
    grounded_content_refusal,
)
from tools.registry import GroundedContext, GroundedPassage


def context(*source_ids: str) -> GroundedContext:
    loc = SourceLocation(
        collection_id="cs", document_id="book", book_title="Foundations", page=1, section="Intro"
    )
    passages: list[GroundedPassage] = []
    for index, source_id in enumerate(source_ids):
        citation = loc.model_copy(update={"page": index + 1, "section": f"S-{index + 1}"})
        passages.append(
            GroundedPassage(
                passage_id=source_id,
                rank=index + 1,
                content=f"Supporting text tagged {source_id}.",
                score=0.9 - 0.1 * index,
                term_coverage=1.0,
                citation=citation,
            )
        )
    return GroundedContext(query="hash tables", grounded=True, passages=passages)


def identity(**overrides) -> SectionIdentity:
    values = dict(
        programme_title="Computer Science Foundations",
        plan_schema="univai.programme.plan",
        plan_version="1.0.0",
        user_id="student-fixture",
        collection_id="cs-programme-2026",
        course_id="CS101",
        week_number=3,
        topic_id="T01",
        lecture_title="Hashing and Collisions",
        created_at="2026-08-03T00:00:00+00:00",
    )
    values.update(overrides)
    return SectionIdentity(**values)


def draft(**overrides) -> SectionDraftLLM:
    base = dict(
        title="Hashing — Section Practice",
        objectives=["Resolve collisions by chaining and probing."],
        activities=[
            SectionActivityDraftLLM(
                title="Chain a hash table by hand",
                description="Exercises a chained bucket for a small key set.",
                duration_minutes=15,
                source_ids=["S1"],
            ),
            SectionActivityDraftLLM(
                title="Trace a linear-probe table",
                description="Probe for each key and record the bucket path.",
                duration_minutes=20,
                source_ids=["S2"],
            ),
        ],
        examples=[
            SectionExampleDraftLLM(
                order=1,
                prompt="Insert 3 keys into a chained table of size 3.",
                steps=[
                    SectionStepDraftLLM(
                        step="hash each key to a bucket",
                        explanation="the hash function maps each key to a bucket index",
                        source_ids=["S1"],
                    ),
                    SectionStepDraftLLM(
                        step="append to the bucket list",
                        explanation="chaining stores colliding keys in one list",
                        source_ids=["S1"],
                    ),
                ],
                conclusion="all three keys resolve with predictable lookup.",
                source_ids=["S1"],
            )
        ],
        todos=[
            SectionTodoDraftLLM(
                text="Rework the probe example without the lecture notes open.",
                time_box_minutes=10,
                source_ids=["S2"],
            )
        ],
    )
    base.update(overrides)
    return SectionDraftLLM(**base)


def test_a_valid_draft_builds_a_versioned_grounded_pack():
    pack = build_section_pack(draft(), identity(), context("S1", "S2"))

    assert pack.schema_name == "univai.section.pack"
    assert pack.session_type == "section"  # never mistaken for a lecture
    assert pack.plan_version == "1.0.0"
    assert pack.topic_id == "T01"
    assert pack.week_number == 3
    assert pack.total_minutes == 35          # 15 + 20
    assert len(pack.activities) == 2
    assert len(pack.examples) == 1
    assert len(pack.todos) == 1
    assert pack.passage_ids == ["S1", "S2"]
    assert pack.citation_count() >= 2
    assert all(activity.citations for activity in pack.activities)
    assert all(example.citations for example in pack.examples)


def test_every_example_step_and_todo_carries_resolved_citations():
    pack = build_section_pack(draft(), identity(), context("S1", "S2"))

    for example in pack.examples:
        assert example.citations
        assert all(step.citations for step in example.steps)
    for todo in pack.todos:
        assert todo.citations
        assert todo.time_box_minutes >= 0


def test_an_uncited_source_id_fails_the_pack():
    bad = draft(
        examples=[
            SectionExampleDraftLLM(
                order=1,
                prompt="p",
                steps=[
                    SectionStepDraftLLM(
                        step="one move",
                        explanation="claimed from nowhere",
                        source_ids=["S99"],
                    )
                ],
                conclusion="c",
                source_ids=["S99"],
            )
        ]
    )
    with pytest.raises(SectionPlanError):
        build_section_pack(bad, identity(), context("S1", "S2"))


def test_a_duration_outside_the_budget_is_refused():
    pack_draft = draft()  # 35 minutes
    tight = SectionBudget(min_total_minutes=40, max_total_minutes=45)
    with pytest.raises(SectionPlanError, match="budget"):
        build_section_pack(pack_draft, identity(), context("S1", "S2"), budget=tight)


def test_duplicate_objectives_are_refused():
    dup = draft(objectives=["same", "same"])
    with pytest.raises(SectionPlanError, match="objectives"):
        build_section_pack(dup, identity(), context("S1", "S2"))


def test_missing_examples_or_todos_yields_a_refusal_not_a_pack():
    no_content = draft(examples=[], todos=[])
    refusal = grounded_content_refusal(no_content, reason="a fitting-a-heap example")
    assert refusal is not None
    assert "no worked example" in refusal.reason or "No section content" in refusal.reason


def test_a_pack_with_no_todos_is_refused():
    no_todos = draft(todos=[])
    refusal = grounded_content_refusal(no_todos)
    assert refusal is not None
    assert "TODO" in refusal.reason


def test_default_budget_spans_thirty_to_one_hundred_twenty_minutes():
    assert DEFAULT_SECTION_BUDGET.min_total_minutes == 30
    assert DEFAULT_SECTION_BUDGET.max_total_minutes == 120