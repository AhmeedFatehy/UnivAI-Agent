"""Contracts for the agent graph: handoffs, drafts, traces and LLM validation.

Two rules are enforced here and nowhere else, so they cannot be bypassed:

1. **Nothing an LLM produces is trusted.** :func:`generate_structured` parses,
   validates against a Pydantic model, and on failure sends exactly one repair
   prompt carrying the validation error. A second failure raises — malformed
   output is never accepted, and the repair loop is bounded at one.

2. **Citations are resolved, never authored.** The model cites passage ids
   (``S1``, ``S2``) from the evidence block it was given. :func:`resolve_citations`
   maps those ids back to real :class:`~document_processing.metadata.SourceLocation`
   objects, so a page number the model never saw is a page number it cannot invent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from agents.prompts import PromptOperation, PromptTemplate, load_prompt, load_prompt_for
from document_processing.metadata import SourceLocation
from planning.overlap import Topic
from planning.programme_planner import ProgrammePlan
from planning.learning_path import (
    LearningPathV1,
    PrerequisiteAnalysisDraft,
    PrerequisiteProposal,
)
from telemetry.tracing import (
    RuntimeFingerprint,
    ServingRecord,
    new_trace_id,
)
from tools.registry import GroundedContext, Refusal

AGENT_SCHEMA = "univai.agent.graph"
AGENT_SCHEMA_VERSION = "1.0.0"

#: The grounded post-lecture section pack: the artifact UnivAI-live consumes.
SECTION_SCHEMA = "univai.section.pack"
SECTION_SCHEMA_VERSION = "1.0.0"

#: One repair attempt, as the issue specifies. Not configurable upward by
#: accident — a caller that wants more has to say so explicitly.
DEFAULT_REPAIR_ATTEMPTS = 1

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_PASSAGE_ID_RE = re.compile(r"^S\d+$")

ModelT = TypeVar("ModelT", bound=BaseModel)


# ── Observable task state ─────────────────────────────────────────────


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    REFUSED = "refused"
    FAILED = "failed"


class AgentName(str, Enum):
    MANAGER = "manager"
    CURRICULUM = "curriculum"
    CONTENT = "content"
    ASSESSMENT = "assessment"


class AssessmentType(str, Enum):
    DIAGNOSTIC = "diagnostic"
    PRACTICE = "practice"
    QUIZ = "quiz"
    ASSIGNMENT = "assignment"
    MIDTERM = "midterm"
    FINAL = "final"
    ORAL_EXAM = "oral_exam"


class ToolCallRecord(BaseModel):
    """One tool invocation, as it happened."""

    tool: str
    ok: bool
    detail: str = ""
    grounded: bool | None = None
    citations: int = Field(default=0, ge=0)


class PromptUseRecord(BaseModel):
    """The stable prompt revision selected for one LLM operation."""

    operation: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class TaskRecord(BaseModel):
    """Observable state for one unit of agent work."""

    task_id: str
    agent: AgentName
    objective: str
    state: TaskState = TaskState.PENDING
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=2, ge=1)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    prompts: list[PromptUseRecord] = Field(default_factory=list)
    llm_calls: int = Field(default=0, ge=0)
    servings: list[ServingRecord] = Field(default_factory=list)
    error: str | None = None
    refusal: Refusal | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def start(self) -> None:
        self.state = TaskState.RUNNING
        self.attempts += 1
        self.started_at = self.started_at or _now()

    def succeed(self) -> None:
        self.state = TaskState.SUCCEEDED
        self.finished_at = _now()

    def refuse(self, refusal: Refusal) -> None:
        self.state = TaskState.REFUSED
        self.refusal = refusal
        self.error = refusal.reason
        self.finished_at = _now()

    def fail(self, error: str) -> None:
        self.state = TaskState.FAILED
        self.error = error
        self.finished_at = _now()

    def record_serving(self, serving: ServingRecord) -> None:
        self.servings.append(serving)

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts


class AgentTrace(BaseModel):
    """Everything the graph did, in order, under one correlated trace id."""

    trace_id: str = Field(default_factory=new_trace_id)
    fingerprint: RuntimeFingerprint | None = None
    tasks: list[TaskRecord] = Field(default_factory=list)
    steps: int = Field(default=0, ge=0)

    def add(self, task: TaskRecord) -> TaskRecord:
        self.tasks.append(task)
        return task

    def by_agent(self, agent: AgentName) -> list[TaskRecord]:
        return [task for task in self.tasks if task.agent is agent]

    @property
    def llm_calls(self) -> int:
        return sum(task.llm_calls for task in self.tasks)

    @property
    def tool_calls(self) -> int:
        return sum(len(task.tool_calls) for task in self.tasks)

    @property
    def refusals(self) -> list[Refusal]:
        return [task.refusal for task in self.tasks if task.refusal is not None]

    @property
    def servings(self) -> list[ServingRecord]:
        return [serving for task in self.tasks for serving in task.servings]

    def states(self) -> dict[str, str]:
        return {task.task_id: task.state.value for task in self.tasks}


# ── Structured handoffs ───────────────────────────────────────────────


class Handoff(BaseModel):
    """A typed instruction from one agent to another. Never free-form text."""

    handoff_id: str
    from_agent: AgentName
    to_agent: AgentName
    objective: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_self_handoff(self) -> "Handoff":
        if self.from_agent is self.to_agent:
            raise ValueError("an agent cannot hand off to itself")
        return self


# ── What the LLM is allowed to return ─────────────────────────────────


def clean_passage_ids(values: list[str]) -> list[str]:
    """Normalise and check the passage ids a model claims to have cited."""
    cleaned = [value.strip().upper() for value in values if value and value.strip()]
    bad = [value for value in cleaned if not _PASSAGE_ID_RE.match(value)]
    if bad:
        raise ValueError(
            f"source_ids must be passage ids like 'S1'; got {bad}. "
            "Cite only the ids shown in the evidence block."
        )
    return list(dict.fromkeys(cleaned))


class ExtractedTopic(BaseModel):
    """A topic proposed by the model, citing passage ids it was shown."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    difficulty: int = Field(default=3, ge=1, le=5)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)


class TopicExtraction(BaseModel):
    topics: list[ExtractedTopic] = Field(min_length=1)


class DraftSegment(BaseModel):
    """One slide's narration, with the passage ids it is built from."""

    slide: int = Field(ge=1)
    heading: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)


class LectureDraftLLM(BaseModel):
    title: str = Field(min_length=1)
    segments: list[DraftSegment] = Field(min_length=1)


class DraftQuestion(BaseModel):
    prompt: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=6)
    correct_option: Literal["A", "B", "C", "D", "E", "F"]
    source: Literal["lecture", "self_study"] = "lecture"
    source_ids: list[str] = Field(min_length=1)
    difficulty: int = Field(default=2, ge=1, le=5)
    learning_objectives: list[str] = Field(default_factory=list)
    rubric: list[str] = Field(default_factory=list)
    follow_up_prompts: list[str] = Field(default_factory=list)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)

    @field_validator("options")
    @classmethod
    def _unique_nonblank_options(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("answer options cannot be blank")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("answer options must be unique")
        return cleaned


class AssessmentDraftLLM(BaseModel):
    assessment_type: AssessmentType = AssessmentType.QUIZ
    questions: list[DraftQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _assessment_option_contract(self):
        expected = 6 if self.assessment_type in {
            AssessmentType.MIDTERM,
            AssessmentType.FINAL,
        } else 4
        allowed_labels = set("ABCDEF"[:expected])
        for question in self.questions:
            if len(question.options) != expected:
                raise ValueError(
                    f"{self.assessment_type.value} questions require exactly "
                    f"{expected} answer options"
                )
            if question.correct_option not in allowed_labels:
                raise ValueError(
                    f"correct_option must be one of {''.join(sorted(allowed_labels))}"
                )
        return self


class AnswerExplanationLLM(BaseModel):
    explanation: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)


class SectionActivityDraftLLM(BaseModel):
    """One guided-practice activity the model proposes for the section."""

    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    duration_minutes: int = Field(ge=1, le=120)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)


class SectionStepDraftLLM(BaseModel):
    """One step of a worked example, each step citing its evidence."""

    step: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)


class SectionExampleDraftLLM(BaseModel):
    """A worked example: a problem solved step by step against the evidence."""

    order: int = Field(ge=1)
    prompt: str = Field(min_length=1)
    steps: list[SectionStepDraftLLM] = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)


class SectionTodoDraftLLM(BaseModel):
    """An actionable learner TODO. Empty lists mean evidence could not support one."""

    text: str = Field(min_length=1)
    time_box_minutes: int = Field(ge=1, le=60)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)


class SectionDraftLLM(BaseModel):
    """The unvalidated section pack as the model returns it.

    ``examples`` and ``todos`` may be empty: the planner treats an empty list as
    an explicit refusal signal when there is evidence to support nothing, and a
    schema failure raises otherwise. This is what lets an unsupported request
    produce a grounded refusal instead of generic teaching content.
    """

    title: str = Field(min_length=1)
    objectives: list[str] = Field(min_length=1)
    activities: list[SectionActivityDraftLLM] = Field(min_length=1)
    examples: list[SectionExampleDraftLLM] = Field(default_factory=list)
    todos: list[SectionTodoDraftLLM] = Field(default_factory=list)


# ── Grounded results the graph returns ────────────────────────────────


class LectureSegment(BaseModel):
    slide: int = Field(ge=1)
    heading: str
    text: str
    citations: list[SourceLocation] = Field(min_length=1)


class Lecture(BaseModel):
    """A lecture that cannot exist without citations."""

    topic_id: str
    title: str = Field(min_length=1)
    segments: list[LectureSegment] = Field(min_length=1)

    @property
    def citations(self) -> list[SourceLocation]:
        return [citation for segment in self.segments for citation in segment.citations]


class AssessmentQuestion(BaseModel):
    prompt: str
    options: list[str] = Field(min_length=4, max_length=6)
    correct_option: Literal["A", "B", "C", "D", "E", "F"]
    source: Literal["lecture", "self_study"]
    citations: list[SourceLocation] = Field(min_length=1)
    difficulty: int = Field(default=2, ge=1, le=5)
    learning_objectives: list[str] = Field(default_factory=list)
    rubric: list[str] = Field(default_factory=list)
    follow_up_prompts: list[str] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def _unique_nonblank_options(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("answer options cannot be blank")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("answer options must be unique")
        return cleaned


class Assessment(BaseModel):
    topic_id: str
    title: str
    assessment_type: AssessmentType = AssessmentType.QUIZ
    covered_scope: list[str] = Field(default_factory=list)
    questions: list[AssessmentQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _assessment_option_contract(self):
        expected = 6 if self.assessment_type in {
            AssessmentType.MIDTERM,
            AssessmentType.FINAL,
        } else 4
        allowed_labels = set("ABCDEF"[:expected])
        for question in self.questions:
            if len(question.options) != expected:
                raise ValueError(
                    f"{self.assessment_type.value} questions require exactly "
                    f"{expected} answer options"
                )
            if question.correct_option not in allowed_labels:
                raise ValueError(
                    f"correct_option must be one of {''.join(sorted(allowed_labels))}"
                )
        return self


class SectionActivity(BaseModel):
    """A guided-practice activity with its resolved citations."""

    order: int = Field(ge=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    duration_minutes: int = Field(ge=1, le=120)
    citations: list[SourceLocation] = Field(min_length=1)


class SectionStep(BaseModel):
    """One worked-example step, mandatory provenance."""
    step: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    citations: list[SourceLocation] = Field(min_length=1)


class SectionExample(BaseModel):
    """A worked example: every step and the whole carry resolved citations."""
    order: int = Field(ge=1)
    prompt: str = Field(min_length=1)
    steps: list[SectionStep] = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    citations: list[SourceLocation] = Field(min_length=1)

    @property
    def step_citations(self) -> list[SourceLocation]:
        return [citation for step in self.steps for citation in step.citations]


class SectionTodo(BaseModel):
    """An actionable TODO: a concrete learner action tied to evidence."""
    order: int = Field(ge=1)
    text: str = Field(min_length=1)
    time_box_minutes: int = Field(ge=1, le=60)
    citations: list[SourceLocation] = Field(min_length=1)


class SectionPackV1(BaseModel):
    """The grounded post-lecture section pack artifact.

    This is deliberately distinct from :class:`Lecture`: it carries
    ``session_type="section"``, its own schema id/version, and is persisted under
    a separate namespace so a live-delivery consumer cannot confuse a section
    follow-up with the lecture narration.
    """

    schema_name: str = SECTION_SCHEMA
    schema_version: str = SECTION_SCHEMA_VERSION
    session_type: Literal["section", "lecture"] = "section"

    programme_title: str = Field(min_length=1)
    plan_schema: str = Field(min_length=1)
    plan_version: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    course_id: str | None = None
    week_number: int | None = Field(default=None, ge=1)

    topic_id: str = Field(min_length=1)
    lecture_title: str = Field(min_length=1)
    title: str = Field(min_length=1)
    focus: str = Field(default="the material covered in the lecture")

    objectives: list[str] = Field(min_length=1)
    activities: list[SectionActivity] = Field(min_length=1)
    examples: list[SectionExample] = Field(default_factory=list)
    todos: list[SectionTodo] = Field(default_factory=list)

    total_minutes: int = Field(ge=1)
    passage_ids: list[str] = Field(default_factory=list)
    created_at: str

    @property
    def citations(self) -> list[SourceLocation]:
        seen: list[SourceLocation] = []
        for activity in self.activities:
            for citation in activity.citations:
                if citation not in seen:
                    seen.append(citation)
        for example in self.examples:
            for citation in [*example.citations, *example.step_citations]:
                if citation not in seen:
                    seen.append(citation)
        for todo in self.todos:
            for citation in todo.citations:
                if citation not in seen:
                    seen.append(citation)
        return seen

    def citation_count(self) -> int:
        return len(self.citations)

    @model_validator(mode="after")
    def _total_minutes_are_self_consistent(self) -> "SectionPackV1":
        expected = sum(activity.duration_minutes for activity in self.activities)
        if self.total_minutes != expected:
            raise ValueError(
                f"total_minutes ({self.total_minutes}) must equal the sum of "
                f"activity durations ({expected})"
            )
        return self


class SectionOutcome(BaseModel):
    """The result of a section request: a valid pack, or an explicit refusal."""

    section: SectionPackV1 | None = None
    refusal: Refusal | None = None

    @model_validator(mode="after")
    def _pack_xor_refusal(self) -> "SectionOutcome":
        if self.section is not None and self.refusal is not None:
            raise ValueError("a section outcome cannot carry both a pack and a refusal")
        if self.section is None and self.refusal is None:
            raise ValueError("a section outcome must carry a pack or a refusal")
        return self


class GraphResult(BaseModel):
    """What a full run of the agent graph produced."""

    schema_name: str = AGENT_SCHEMA
    schema_version: str = AGENT_SCHEMA_VERSION

    collection_id: str
    user_id: str
    programme_title: str
    plan: ProgrammePlan | None = None
    lectures: list[Lecture] = Field(default_factory=list)
    assessments: list[Assessment] = Field(default_factory=list)
    refusals: list[Refusal] = Field(default_factory=list)
    trace: AgentTrace = Field(default_factory=AgentTrace)
    completed: bool = False

    @model_validator(mode="after")
    def _incomplete_runs_explain_themselves(self) -> "GraphResult":
        failed = any(task.state is TaskState.FAILED for task in self.trace.tasks)
        if (
            not self.completed
            and self.plan is None
            and not self.refusals
            and not failed
        ):
            raise ValueError(
                "a run without a plan must record why — a refusal or a failed task"
            )
        return self

    def citation_count(self) -> int:
        return sum(len(lecture.citations) for lecture in self.lectures) + sum(
            len(question.citations)
            for assessment in self.assessments
            for question in assessment.questions
        )


# ── LLM output validation with one bounded repair ─────────────────────


class StructuredOutputError(RuntimeError):
    """The model could not produce schema-valid output within the repair budget."""

    def __init__(self, schema: str, attempts: int, last_error: str, last_raw: str):
        super().__init__(
            f"{schema} invalid after {attempts} attempt(s): {last_error}"
        )
        self.schema = schema
        self.attempts = attempts
        self.last_error = last_error
        self.last_raw = last_raw


def extract_json(raw: str) -> str:
    """Pull the JSON body out of a model reply that may be fenced or chatty."""
    text = (raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start = min(
        (index for index in (text.find("{"), text.find("[")) if index != -1),
        default=-1,
    )
    if start == -1:
        return text

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def generate_structured(
    llm: Callable[[str], str],
    prompt: str,
    schema: type[ModelT],
    *,
    repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
    on_call: Callable[[], None] | None = None,
    on_served: Callable[[ServingRecord], None] | None = None,
) -> ModelT:
    """Call the model and return a validated instance of ``schema``.

    On invalid output, one repair prompt is sent containing the exact validation
    error and the JSON schema. If that also fails, :class:`StructuredOutputError`
    is raised — arbitrary malformed output is never returned to a caller.

    ``on_served`` is invoked after every model attempt with a
    :class:`~telemetry.tracing.ServingRecord` describing which provider/model
    served the reply (when the LLM exposes one) — the wiring that puts serving
    metadata on the task trace without the agent reading private prompts.
    """
    if repair_attempts < 0:
        raise ValueError("repair_attempts cannot be negative")

    current_prompt = prompt
    last_error = ""
    last_raw = ""

    for attempt in range(repair_attempts + 1):
        if on_call is not None:
            on_call()
        last_raw = llm(current_prompt) or ""
        _record_serving(llm, on_served)
        try:
            payload = json.loads(extract_json(last_raw))
        except (json.JSONDecodeError, TypeError) as error:
            last_error = f"response is not valid JSON: {error}"
        else:
            try:
                return schema.model_validate(payload)
            except ValidationError as error:
                last_error = _compact_validation_error(error)

        if attempt >= repair_attempts:
            break
        current_prompt = _repair_prompt(prompt, last_raw, last_error, schema)

    raise StructuredOutputError(schema.__name__, repair_attempts + 1, last_error, last_raw)


def _record_serving(
    llm: Callable[[str], str],
    on_served: Callable[[ServingRecord], None] | None,
) -> None:
    """Forward the LLM's last served result, when it exposes one and a listener wants it."""
    if on_served is None:
        return
    last_served = getattr(llm, "last_served", None)
    if last_served is None:
        return
    on_served(ServingRecord.from_served(last_served))


def _compact_validation_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors()[:6]
    )


def _repair_prompt(
    original: str, bad_output: str, error: str, schema: type[BaseModel]
) -> str:
    return load_prompt_for(PromptOperation.SHARED_REPAIR_JSON).render(
        original_prompt=original,
        previous_reply=bad_output[:2000],
        validation_errors=error,
        json_schema=json.dumps(schema.model_json_schema(), indent=2)[:3000],
    )


# ── Citation resolution ───────────────────────────────────────────────


class UngroundedCitation(ValueError):
    """The model cited a passage id it was never shown."""


def resolve_citations(
    source_ids: list[str], context: GroundedContext
) -> list[SourceLocation]:
    """Map model-supplied passage ids to the real locations behind them.

    Raises :class:`UngroundedCitation` for any id that was not in the evidence
    block, which is what stops a model from citing a book it never saw.
    """
    available = context.by_id()
    unknown = [value for value in source_ids if value not in available]
    if unknown:
        raise UngroundedCitation(
            f"cited unknown passage id(s) {unknown}; available: {sorted(available)}"
        )

    locations: list[SourceLocation] = []
    for value in source_ids:
        location = available[value].citation
        if location not in locations:
            locations.append(location)
    return locations


def topics_from_extraction(
    extraction: TopicExtraction,
    context: GroundedContext,
    *,
    prefix: str = "T",
) -> list[Topic]:
    """Turn model-proposed topics into evidence-backed :class:`Topic` objects.

    Prerequisite titles are resolved to the ids of the topics in this same
    extraction; a prerequisite naming something the model did not extract is
    dropped here and reported later by the planner.
    """
    from planning.overlap import TopicEvidence

    ids_by_title = {
        item.title.strip().lower(): f"{prefix}{index:02d}"
        for index, item in enumerate(extraction.topics, start=1)
    }
    available = context.by_id()

    topics: list[Topic] = []
    for index, item in enumerate(extraction.topics, start=1):
        # Raises for any id the model was not shown — this is the grounding gate.
        resolve_citations(item.source_ids, context)
        evidence = [
            TopicEvidence(
                quote=available[source_id].content[:800],
                location=available[source_id].citation,
            )
            for source_id in item.source_ids
        ]
        topics.append(
            Topic(
                topic_id=f"{prefix}{index:02d}",
                title=item.title,
                summary=item.summary,
                keywords=item.keywords,
                prerequisites=[
                    ids_by_title[name.strip().lower()]
                    for name in item.prerequisites
                    if name.strip().lower() in ids_by_title
                ],
                difficulty=item.difficulty,
                evidence=evidence,
            )
        )
    return topics


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AGENT_SCHEMA",
    "AGENT_SCHEMA_VERSION",
    "SECTION_SCHEMA",
    "SECTION_SCHEMA_VERSION",
    "DEFAULT_REPAIR_ATTEMPTS",
    "AgentName",
    "AgentTrace",
    "Assessment",
    "AssessmentDraftLLM",
    "AssessmentQuestion",
    "AssessmentType",
    "AnswerExplanationLLM",
    "DraftQuestion",
    "DraftSegment",
    "ExtractedTopic",
    "GraphResult",
    "Handoff",
    "Lecture",
    "LectureDraftLLM",
    "LectureSegment",
    "LearningPathV1",
    "PromptTemplate",
    "PromptUseRecord",
    "PrerequisiteAnalysisDraft",
    "PrerequisiteProposal",
    "RuntimeFingerprint",
    "SectionActivity",
    "SectionActivityDraftLLM",
    "SectionDraftLLM",
    "SectionExample",
    "SectionExampleDraftLLM",
    "SectionOutcome",
    "SectionPackV1",
    "SectionStep",
    "SectionStepDraftLLM",
    "SectionTodo",
    "SectionTodoDraftLLM",
    "ServingRecord",
    "StructuredOutputError",
    "TaskRecord",
    "TaskState",
    "ToolCallRecord",
    "TopicExtraction",
    "UngroundedCitation",
    "extract_json",
    "generate_structured",
    "load_prompt",
    "resolve_citations",
    "topics_from_extraction",
]
