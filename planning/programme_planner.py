"""Turn evidence-backed topics into a programme plan.

The planner never invents structure. Every semester boundary, every merge, every
dropped prerequisite is a :class:`PlanningDecision` that carries the source
locations it was derived from, so a reviewer can open the book and check it.

The number of semesters is an *output*, not an input: it comes from total
estimated hours against semester capacity, and the decision that produced it is
recorded with the arithmetic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from document_processing.metadata import SourceLocation
from planning.overlap import (
    MERGE_THRESHOLD,
    RELATED_THRESHOLD,
    Topic,
    TopicOverlap,
    find_overlaps,
    merge_overlapping,
)
from planning.prerequisites import (
    PrerequisiteIssue,
    build_graph,
    dependency_depth,
    teaching_order,
)
from planning.workload import (
    DEFAULT_SEMESTER_CAPACITY_HOURS,
    WorkloadEstimate,
    estimate_all,
    pack_semesters,
    required_semesters,
    semester_hours,
)

PROGRAMME_PLAN_SCHEMA = "univai.agent.programme_plan"
PROGRAMME_PLAN_VERSION = "1.0.0"

DecisionKind = Literal["overlap", "prerequisite", "workload", "semester", "coverage"]


class PlanningDecision(BaseModel):
    """A choice the planner made, why, and the evidence behind it."""

    decision_id: str = Field(min_length=1)
    kind: DecisionKind
    summary: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence: list[SourceLocation] = Field(min_length=1)


class PlannedTopic(BaseModel):
    """A topic as scheduled: its position, its cost and its citations."""

    topic_id: str
    title: str
    order: int = Field(ge=1)
    summary: str
    difficulty: int = Field(ge=1, le=5)
    prerequisites: list[str] = Field(default_factory=list)
    contact_hours: float = Field(gt=0)
    self_study_hours: float = Field(ge=0)
    total_hours: float = Field(gt=0)
    workload_basis: str
    citations: list[SourceLocation] = Field(min_length=1)


class SemesterPlan(BaseModel):
    index: int = Field(ge=1)
    title: str
    topics: list[PlannedTopic] = Field(min_length=1)
    total_hours: float = Field(ge=0)
    capacity_hours: float = Field(gt=0)

    @property
    def over_capacity(self) -> bool:
        return self.total_hours > self.capacity_hours


class ProgrammePlan(BaseModel):
    """The deliverable: an ordered, costed, evidence-backed curriculum."""

    schema_name: str = PROGRAMME_PLAN_SCHEMA
    schema_version: str = PROGRAMME_PLAN_VERSION

    programme_title: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    semesters: list[SemesterPlan] = Field(min_length=1)
    decisions: list[PlanningDecision] = Field(min_length=1)
    prerequisite_issues: list[PrerequisiteIssue] = Field(default_factory=list)
    overlaps: list[TopicOverlap] = Field(default_factory=list)
    topics_considered: int = Field(ge=1)
    topics_scheduled: int = Field(ge=1)
    source_documents: list[str] = Field(min_length=1)
    generated_at: str

    @model_validator(mode="after")
    def _every_topic_is_cited(self) -> "ProgrammePlan":
        for semester in self.semesters:
            for topic in semester.topics:
                if not topic.citations:
                    raise ValueError(
                        f"topic '{topic.topic_id}' has no citation; an uncited topic "
                        "cannot be scheduled"
                    )
        return self

    @property
    def total_hours(self) -> float:
        return round(sum(semester.total_hours for semester in self.semesters), 2)

    @property
    def is_multi_semester(self) -> bool:
        return len(self.semesters) > 1

    def all_citations(self) -> list[SourceLocation]:
        seen: dict[tuple, SourceLocation] = {}
        for semester in self.semesters:
            for topic in semester.topics:
                for citation in topic.citations:
                    seen.setdefault(
                        (citation.document_id, citation.page, citation.section), citation
                    )
        return list(seen.values())


def create_programme_plan(
    topics: list[Topic],
    *,
    programme_title: str,
    collection_id: str,
    capacity_hours: float = DEFAULT_SEMESTER_CAPACITY_HOURS,
    max_semesters: int = 8,
    merge_threshold: float = MERGE_THRESHOLD,
    related_threshold: float = RELATED_THRESHOLD,
    generated_at: str | None = None,
) -> ProgrammePlan:
    """Build a plan from evidence-backed topics.

    Raises ``ValueError`` when there are no topics — an empty corpus produces a
    refusal upstream, never an empty plan dressed up as a result.
    """
    if not topics:
        raise ValueError(
            "cannot plan a programme with no evidence-backed topics; "
            "index source material first"
        )
    if not collection_id.strip():
        raise ValueError("collection_id is required")

    considered = len(topics)
    decisions: list[PlanningDecision] = []

    # 1. Overlap — teach shared material once.
    overlaps = find_overlaps(
        topics, merge_threshold=merge_threshold, related_threshold=related_threshold
    )
    scheduled_topics, replaced_by = merge_overlapping(topics, overlaps)
    decisions.extend(_overlap_decisions(overlaps, replaced_by))

    # 2. Prerequisites — nothing before what it depends on.
    order, issues = teaching_order(scheduled_topics)
    graph = build_graph(scheduled_topics)
    depth = dependency_depth(scheduled_topics)
    decisions.extend(_prerequisite_decisions(scheduled_topics, issues, graph, depth))

    # 3. Workload — how many hours the books actually contain.
    estimates = estimate_all(scheduled_topics)
    needed = required_semesters(estimates, capacity_hours=capacity_hours)
    decisions.append(_workload_decision(scheduled_topics, estimates, capacity_hours, needed))

    # 4. Semesters — pack in teaching order, honouring prerequisites.
    keep_together = [
        (overlap.topic_a, overlap.topic_b)
        for overlap in overlaps
        if overlap.decision == "sequence"
        and overlap.topic_a not in replaced_by
        and overlap.topic_b not in replaced_by
    ]
    packed = pack_semesters(
        order,
        estimates,
        graph,
        capacity_hours=capacity_hours,
        max_semesters=max_semesters,
        keep_together=keep_together,
    )

    by_id = {topic.topic_id: topic for topic in scheduled_topics}
    semesters: list[SemesterPlan] = []
    position = 0
    for index, semester_ids in enumerate(packed, start=1):
        planned: list[PlannedTopic] = []
        for topic_id in semester_ids:
            topic = by_id[topic_id]
            estimate = estimates[topic_id]
            position += 1
            planned.append(
                PlannedTopic(
                    topic_id=topic_id,
                    title=topic.title,
                    order=position,
                    summary=topic.summary,
                    difficulty=topic.difficulty,
                    prerequisites=list(graph.get(topic_id, [])),
                    contact_hours=estimate.contact_hours,
                    self_study_hours=estimate.self_study_hours,
                    total_hours=estimate.total_hours,
                    workload_basis=estimate.basis,
                    citations=topic.citations,
                )
            )
        semesters.append(
            SemesterPlan(
                index=index,
                title=f"Semester {index}",
                topics=planned,
                total_hours=semester_hours(semester_ids, estimates),
                capacity_hours=capacity_hours,
            )
        )

    decisions.append(_semester_decision(semesters, scheduled_topics, capacity_hours, needed))
    decisions.append(_coverage_decision(scheduled_topics))

    source_documents = sorted(
        {
            citation.document_id
            for topic in scheduled_topics
            for citation in topic.citations
        }
    )

    return ProgrammePlan(
        programme_title=programme_title,
        collection_id=collection_id,
        semesters=semesters,
        decisions=decisions,
        prerequisite_issues=issues,
        overlaps=overlaps,
        topics_considered=considered,
        topics_scheduled=len(scheduled_topics),
        source_documents=source_documents,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
    )


# ── Decision builders ─────────────────────────────────────────────────


def _overlap_decisions(
    overlaps: list[TopicOverlap], replaced_by: dict[str, str]
) -> list[PlanningDecision]:
    if not overlaps:
        return []

    decisions = [
        PlanningDecision(
            decision_id=f"overlap-{index:02d}",
            kind="overlap",
            summary=(
                f"{overlap.decision.replace('_', ' ')}: {overlap.topic_a} / {overlap.topic_b} "
                f"({overlap.similarity:.0%} shared terms)"
            ),
            rationale=overlap.rationale,
            evidence=overlap.evidence,
        )
        for index, overlap in enumerate(overlaps, start=1)
    ]

    if replaced_by:
        merged = ", ".join(f"{absorbed}→{keeper}" for absorbed, keeper in sorted(replaced_by.items()))
        decisions.append(
            PlanningDecision(
                decision_id="overlap-merge-summary",
                kind="overlap",
                summary=f"{len(replaced_by)} duplicate topic(s) folded into their counterpart",
                rationale=(
                    f"Merged {merged}. The surviving topic keeps the evidence from both books, "
                    "so the material is still taught from every source that covers it."
                ),
                evidence=[
                    location
                    for overlap in overlaps
                    if overlap.decision == "merge"
                    for location in overlap.evidence
                ][:6],
            )
        )
    return decisions


def _prerequisite_decisions(
    topics: list[Topic],
    issues: list[PrerequisiteIssue],
    graph: dict[str, list[str]],
    depth: dict[str, int],
) -> list[PlanningDecision]:
    by_id = {topic.topic_id: topic for topic in topics}
    decisions: list[PlanningDecision] = []

    constrained = {node: parents for node, parents in graph.items() if parents}
    if constrained:
        described = "; ".join(
            f"'{by_id[node].title}' after "
            + ", ".join(f"'{by_id[parent].title}'" for parent in parents)
            for node, parents in sorted(constrained.items())
        )
        evidence: list[SourceLocation] = []
        for node, parents in sorted(constrained.items()):
            evidence.append(by_id[node].evidence[0].location)
            for parent in parents:
                evidence.append(by_id[parent].evidence[0].location)
        decisions.append(
            PlanningDecision(
                decision_id="prerequisite-order",
                kind="prerequisite",
                summary=f"{len(constrained)} topic(s) constrained by prerequisites",
                rationale=(
                    f"Teaching order respects: {described}. "
                    f"Deepest prerequisite chain is {max(depth.values(), default=0)} level(s)."
                ),
                evidence=evidence[:8],
            )
        )

    for index, issue in enumerate(issues, start=1):
        anchors = [
            by_id[topic_id].evidence[0].location
            for topic_id in issue.topics
            if topic_id in by_id
        ]
        if not anchors and topics:
            anchors = [topics[0].evidence[0].location]
        decisions.append(
            PlanningDecision(
                decision_id=f"prerequisite-issue-{index:02d}",
                kind="prerequisite",
                summary=f"{issue.kind.replace('_', ' ')} resolved",
                rationale=issue.detail,
                evidence=anchors,
            )
        )
    return decisions


def _workload_decision(
    topics: list[Topic],
    estimates: dict[str, WorkloadEstimate],
    capacity_hours: float,
    needed: int,
) -> PlanningDecision:
    total = round(sum(item.total_hours for item in estimates.values()), 2)
    heaviest = max(estimates.values(), key=lambda item: item.total_hours)
    return PlanningDecision(
        decision_id="workload-total",
        kind="workload",
        summary=f"{total} total hours across {len(estimates)} topic(s)",
        rationale=(
            f"Estimated from cited page counts and difficulty. Heaviest topic is "
            f"'{heaviest.title}' at {heaviest.total_hours} h ({heaviest.basis}). "
            f"At {capacity_hours} h per semester this needs at least {needed} semester(s)."
        ),
        evidence=[topic.evidence[0].location for topic in topics][:6],
    )


def _semester_decision(
    semesters: list[SemesterPlan],
    topics: list[Topic],
    capacity_hours: float,
    needed: int,
) -> PlanningDecision:
    count = len(semesters)
    breakdown = "; ".join(
        f"S{semester.index}: {len(semester.topics)} topic(s), {semester.total_hours} h"
        for semester in semesters
    )
    if count == 1:
        rationale = (
            f"All material fits one semester: {semesters[0].total_hours} h against a "
            f"{capacity_hours} h capacity, and no prerequisite forces a split. ({breakdown})"
        )
    else:
        rationale = (
            f"Split into {count} semesters because {needed} semester(s) are needed for "
            f"{round(sum(s.total_hours for s in semesters), 2)} h at {capacity_hours} h "
            f"capacity, and prerequisites must not be taught alongside what depends on "
            f"them. ({breakdown})"
        )
    over = [semester.index for semester in semesters if semester.over_capacity]
    if over:
        rationale += (
            f" Semester(s) {', '.join(str(index) for index in over)} exceed capacity because "
            "the semester cap was reached; material was kept rather than dropped."
        )

    return PlanningDecision(
        decision_id="semester-layout",
        kind="semester",
        summary=f"{count} semester(s)",
        rationale=rationale,
        evidence=[
            semester.topics[0].citations[0] for semester in semesters
        ],
    )


def _coverage_decision(topics: list[Topic]) -> PlanningDecision:
    by_document: dict[str, list[SourceLocation]] = {}
    titles: dict[str, str] = {}
    for topic in topics:
        for citation in topic.citations:
            by_document.setdefault(citation.document_id, []).append(citation)
            titles[citation.document_id] = citation.book_title

    described = ", ".join(
        f"'{titles[document_id]}' ({len(locations)} cited passage(s))"
        for document_id, locations in sorted(by_document.items(), key=lambda item: titles[item[0]])
    )
    return PlanningDecision(
        decision_id="coverage-sources",
        kind="coverage",
        summary=f"{len(by_document)} source book(s) contribute to the programme",
        rationale=f"Every scheduled topic cites indexed material: {described}.",
        evidence=[locations[0] for locations in by_document.values()],
    )


__all__ = [
    "PROGRAMME_PLAN_SCHEMA",
    "PROGRAMME_PLAN_VERSION",
    "PlannedTopic",
    "PlanningDecision",
    "ProgrammePlan",
    "SemesterPlan",
    "create_programme_plan",
]
