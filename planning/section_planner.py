"""Deterministic rules for a grounded post-lecture section pack.

The model may propose activities, worked examples and TODOs, but this module
owns the constraints: everything must cite supplied evidence, the total session
duration must sit inside the configured section budget, objectives must be
non-empty and distinct, and a pack must never be mistaken for the lecture it
follows. Any violation is a :class:`SectionPlanError` — the caller either repairs
or refuses; it never ships a rule-breaking pack.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.schemas import (
    SECTION_SCHEMA,
    SECTION_SCHEMA_VERSION,
    SectionActivity,
    SectionDraftLLM,
    SectionExample,
    SectionPackV1,
    SectionStep,
    SectionTodo,
    resolve_citations,
)
from tools.registry import GroundedContext, Refusal

SECTION_PLAN_SCHEMA = SECTION_SCHEMA
SECTION_PLAN_VERSION = SECTION_SCHEMA_VERSION


class SectionPlanError(ValueError):
    """A section draft broke one of the deterministic invariants."""


class SectionBudget(BaseModel):
    """The configured time box (in minutes) a section total must stay inside."""

    min_total_minutes: int = Field(default=30, ge=1)
    max_total_minutes: int = Field(default=120, ge=1)

    @property
    def label(self) -> str:
        return f"{self.min_total_minutes}-{self.max_total_minutes}"

    def validate_total(self, total: int) -> None:
        if not (self.min_total_minutes <= total <= self.max_total_minutes):
            raise SectionPlanError(
                f"section total of {total} minutes is outside the {self.label} budget"
            )


DEFAULT_SECTION_BUDGET = SectionBudget()


class SectionIdentity(BaseModel):
    """Everything needed to pin a section to exactly one approved lecture/plan."""

    programme_title: str = Field(min_length=1)
    plan_schema: str = Field(min_length=1)
    plan_version: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    course_id: str | None = None
    week_number: int | None = Field(default=None, ge=1)
    topic_id: str = Field(min_length=1)
    lecture_title: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


def total_draft_minutes(draft: SectionDraftLLM) -> int:
    return sum(activity.duration_minutes for activity in draft.activities)


def ensure_unique(values: list[str], kind: str) -> None:
    """Reject duplicate objectives/titles, which a real section must not repeat."""
    normalised = [value.strip().lower() for value in values if value.strip()]
    if len(values) != len({value.strip() for value in values}):
        raise SectionPlanError(f"a section must not repeat its {kind}")
    if len(normalised) != len(set(normalised)):
        raise SectionPlanError(f"a section must not repeat its {kind}")
    if len(values) != len(normalised):
        raise SectionPlanError(f"a section {kind} cannot be empty or blank")


def grounded_content_refusal(
    draft: SectionDraftLLM, reason: str = ""
) -> Refusal | None:
    """A refusal when evidence supports no worked example or actionable TODO.

    When the model returns an empty ``examples``/``todos`` list, or the user
    asked for something the evidence cannot support, the fix is an explicit
    :class:`~tools.registry.Refusal` — never a generic teaching pack.
    """
    detail = reason or "the supplied evidence supports no worked example or actionable TODO"
    if not draft.examples and not draft.todos:
        return Refusal(reason=f"No section content to publish: {detail}", query=reason)
    if not draft.examples:
        return Refusal(reason="No worked example is supported by the supplied evidence", query=reason)
    if not draft.todos:
        return Refusal(reason="No actionable TODO is supported by the supplied evidence", query=reason)
    return None


def _verify_citations(source_ids: list[str], context: GroundedContext) -> None:
    from agents.schemas import UngroundedCitation

    try:
        resolve_citations(source_ids, context)
    except UngroundedCitation as error:
        raise SectionPlanError(str(error)) from error


def build_section_pack(
    draft: SectionDraftLLM,
    identity: SectionIdentity,
    context: GroundedContext,
    *,
    focus: str = "the material covered in the lecture",
    budget: SectionBudget = DEFAULT_SECTION_BUDGET,
) -> SectionPackV1:
    """Turn a model draft into a validated, evidence-bound section pack.

    Enforces, in order:
        1. every claimed passage id resolves against the supplied evidence;
        2. the total activity duration stays inside the configured budget;
        3. objectives and activity titles are non-empty and distinct;
        4. each activity/example/TODO carries real resolved citations.
    """
    for activity in draft.activities:
        _verify_citations(activity.source_ids, context)
    for example in draft.examples:
        _verify_citations(example.source_ids, context)
        for step in example.steps:
            _verify_citations(step.source_ids, context)
    for todo in draft.todos:
        _verify_citations(todo.source_ids, context)

    total = total_draft_minutes(draft)
    budget.validate_total(total)
    ensure_unique(draft.objectives, "objectives")
    ensure_unique([activity.title for activity in draft.activities], "activities")

    activities = [
        SectionActivity(
            order=order,
            title=activity.title,
            description=activity.description,
            duration_minutes=activity.duration_minutes,
            citations=resolve_citations(activity.source_ids, context),
        )
        for order, activity in enumerate(draft.activities, start=1)
    ]
    examples = [
        SectionExample(
            order=example.order,
            prompt=example.prompt,
            steps=[
                SectionStep(
                    step=step.step,
                    explanation=step.explanation,
                    citations=resolve_citations(step.source_ids, context),
                )
                for step in example.steps
            ],
            conclusion=example.conclusion,
            citations=resolve_citations(example.source_ids, context),
        )
        for example in draft.examples
    ]
    todos = [
        SectionTodo(
            order=order,
            text=todo.text,
            time_box_minutes=todo.time_box_minutes,
            citations=resolve_citations(todo.source_ids, context),
        )
        for order, todo in enumerate(draft.todos, start=1)
    ]

    passage_ids: list[str] = []
    for source_ids in _all_source_ids(draft):
        for source_id in source_ids:
            if source_id not in passage_ids:
                passage_ids.append(source_id)

    return SectionPackV1(
        schema_name=SECTION_SCHEMA,
        schema_version=SECTION_SCHEMA_VERSION,
        programme_title=identity.programme_title,
        plan_schema=identity.plan_schema,
        plan_version=identity.plan_version,
        user_id=identity.user_id,
        collection_id=identity.collection_id,
        course_id=identity.course_id,
        week_number=identity.week_number,
        topic_id=identity.topic_id,
        lecture_title=identity.lecture_title,
        title=draft.title,
        focus=focus,
        objectives=draft.objectives,
        activities=sorted(activities, key=lambda item: item.order),
        examples=sorted(examples, key=lambda item: item.order),
        todos=todos,
        total_minutes=total,
        passage_ids=passage_ids,
        created_at=identity.created_at,
    )


def _all_source_ids(draft: SectionDraftLLM) -> list[list[str]]:
    ids: list[list[str]] = [activity.source_ids for activity in draft.activities]
    for example in draft.examples:
        ids.append(example.source_ids)
        ids.extend(step.source_ids for step in example.steps)
    ids.extend(todo.source_ids for todo in draft.todos)
    return ids


__all__ = [
    "DEFAULT_SECTION_BUDGET",
    "SECTION_PLAN_SCHEMA",
    "SECTION_PLAN_VERSION",
    "SectionBudget",
    "SectionIdentity",
    "SectionPlanError",
    "build_section_pack",
    "ensure_unique",
    "grounded_content_refusal",
    "total_draft_minutes",
]