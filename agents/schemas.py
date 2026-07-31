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
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from document_processing.metadata import SourceLocation
from planning.overlap import Topic
from planning.programme_planner import ProgrammePlan
from tools.registry import GroundedContext, Refusal

AGENT_SCHEMA = "univai.agent.graph"
AGENT_SCHEMA_VERSION = "1.0.0"

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

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


class ToolCallRecord(BaseModel):
    """One tool invocation, as it happened."""

    tool: str
    ok: bool
    detail: str = ""
    grounded: bool | None = None
    citations: int = Field(default=0, ge=0)


class TaskRecord(BaseModel):
    """Observable state for one unit of agent work."""

    task_id: str
    agent: AgentName
    objective: str
    state: TaskState = TaskState.PENDING
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=2, ge=1)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    llm_calls: int = Field(default=0, ge=0)
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

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts


class AgentTrace(BaseModel):
    """Everything the graph did, in order."""

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
    options: list[str] = Field(min_length=4, max_length=4)
    correct_option: Literal["A", "B", "C", "D"]
    source: Literal["lecture", "self_study"] = "lecture"
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _passage_ids(cls, values: list[str]) -> list[str]:
        return clean_passage_ids(values)


class AssessmentDraftLLM(BaseModel):
    questions: list[DraftQuestion] = Field(min_length=1)


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
    options: list[str] = Field(min_length=4, max_length=4)
    correct_option: Literal["A", "B", "C", "D"]
    source: Literal["lecture", "self_study"]
    citations: list[SourceLocation] = Field(min_length=1)


class Assessment(BaseModel):
    topic_id: str
    title: str
    questions: list[AssessmentQuestion] = Field(min_length=1)


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
) -> ModelT:
    """Call the model and return a validated instance of ``schema``.

    On invalid output, one repair prompt is sent containing the exact validation
    error and the JSON schema. If that also fails, :class:`StructuredOutputError`
    is raised — arbitrary malformed output is never returned to a caller.
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


def _compact_validation_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors()[:6]
    )


def _repair_prompt(
    original: str, bad_output: str, error: str, schema: type[BaseModel]
) -> str:
    return (
        f"{original}\n\n"
        "--- REPAIR ---\n"
        "Your previous reply was rejected by schema validation.\n"
        f"Previous reply:\n{bad_output[:2000]}\n\n"
        f"Validation errors: {error}\n\n"
        f"Required JSON schema:\n{json.dumps(schema.model_json_schema(), indent=2)[:3000]}\n\n"
        "Reply with corrected JSON only. No prose, no code fences."
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


# ── Prompt loading ────────────────────────────────────────────────────


class PromptTemplate(BaseModel):
    """A versioned prompt with an explicit list of the variables it expects.

    Substitution replaces only the declared ``{variable}`` names, so the JSON
    examples and schemas these prompts carry keep their braces intact.
    """

    name: str
    version: str
    description: str = ""
    variables: list[str] = Field(default_factory=list)
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)

    def render(self, **values: Any) -> str:
        """Fill the template. A missing declared variable is an error."""
        missing = [name for name in self.variables if name not in values]
        if missing:
            raise KeyError(
                f"prompt '{self.name}' v{self.version} needs {missing}, which were not supplied"
            )

        body = self.user
        for name in self.variables:
            body = body.replace("{" + name + "}", str(values[name]))

        leftover = [
            name for name in self.variables if "{" + name + "}" in body
        ]
        if leftover:  # pragma: no cover — only reachable if a value re-introduces a token
            raise ValueError(f"prompt '{self.name}' still contains {leftover} after render")

        return f"{self.system.strip()}\n\n{body.strip()}"


@lru_cache(maxsize=16)
def load_prompt(name: str, prompts_dir: str | None = None) -> PromptTemplate:
    """Load and validate a prompt template from ``prompts/<name>.yaml``."""
    directory = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"prompt template not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"prompt template {path} must be a YAML mapping")
    data.setdefault("name", name)
    return PromptTemplate.model_validate(data)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AGENT_SCHEMA",
    "AGENT_SCHEMA_VERSION",
    "DEFAULT_REPAIR_ATTEMPTS",
    "AgentName",
    "AgentTrace",
    "Assessment",
    "AssessmentDraftLLM",
    "AssessmentQuestion",
    "DraftQuestion",
    "DraftSegment",
    "ExtractedTopic",
    "GraphResult",
    "Handoff",
    "Lecture",
    "LectureDraftLLM",
    "LectureSegment",
    "PromptTemplate",
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
