"""Manager agent — the only node that decides what happens next.

The hierarchy is bounded by construction: the manager issues typed
:class:`~agents.schemas.Handoff` objects to exactly three specialists, and the
specialists never call each other. There is no free-form delegation, so there is
no path by which the graph can wander.

Three separate bounds keep a run finite:

* ``max_steps`` — total manager turns;
* ``max_attempts`` per task — a failed specialist is retried, then abandoned;
* ``repair_attempts`` — one, inside :func:`~agents.schemas.generate_structured`.

The manager also owns the run's observable state: every task it dispatches is a
:class:`~agents.schemas.TaskRecord` on the trace, with its state, attempts, tool
calls and LLM calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from agents.schemas import (
    DEFAULT_REPAIR_ATTEMPTS,
    AgentName,
    AgentTrace,
    Assessment,
    AssessmentType,
    GraphResult,
    Handoff,
    Lecture,
    TaskRecord,
)
from planning.programme_planner import ProgrammePlan
from telemetry.tracing import RuntimeFingerprint, runtime_fingerprint
from tools.registry import Refusal, ToolContext

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 12
DEFAULT_MAX_ATTEMPTS = 2
#: How many planned topics get lecture + assessment drafts in one run.
DEFAULT_TOPIC_BUDGET = 3


@dataclass
class AgentRuntime:
    """Everything the agents need that touches the outside world.

    ``llm`` is a plain ``str -> str`` callable so tests inject a scripted
    responder and CI never makes a paid call. :func:`ollama_llm` builds the real
    one for integrated runs.
    """

    llm: Callable[[str], str]
    tool_context: ToolContext = field(default_factory=ToolContext)
    repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS
    max_steps: int = DEFAULT_MAX_STEPS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    topic_budget: int = DEFAULT_TOPIC_BUDGET
    fingerprint: RuntimeFingerprint | None = None

    def __post_init__(self) -> None:
        if self.fingerprint is None:
            self.fingerprint = runtime_fingerprint()


@dataclass
class ProgrammeRequest:
    """What a caller asks the graph for."""

    programme_title: str
    collection_id: str
    user_id: str
    seed_queries: list[str]
    capacity_hours: float = 120.0
    max_semesters: int = 8
    max_topics: int = 8
    slide_count: int = 4
    question_count: int = 4
    assessment_type: AssessmentType = AssessmentType.QUIZ
    assessment_scope: list[str] = field(default_factory=list)
    difficulty_distribution: str = "mostly easy and medium"
    allowed_question_formats: list[str] = field(default_factory=lambda: ["mcq"])
    #: Restrict lecture/assessment retrieval to the books the topic's own
    #: evidence came from. Off by default: a topic is usually covered by more
    #: than one book, and narrowing to the first book that happened to match
    #: hides the rest of the collection from the draft.
    restrict_to_topic_sources: bool = False

    def __post_init__(self) -> None:
        if not self.programme_title.strip():
            raise ValueError("programme_title is required")
        if not self.collection_id.strip():
            raise ValueError("collection_id is required")
        if not self.user_id.strip():
            raise ValueError("user_id is required")
        if not self.seed_queries:
            raise ValueError("at least one seed query is required")


class ManagerAgent:
    """Decides the next handoff and records the outcome of the last one."""

    name = AgentName.MANAGER

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self._handoffs = 0

    # ── Handoff construction ──────────────────────────────────────────

    def _handoff(
        self,
        to_agent: AgentName,
        *,
        objective: str,
        request: ProgrammeRequest,
        payload: dict,
        constraints: dict,
    ) -> Handoff:
        self._handoffs += 1
        return Handoff(
            handoff_id=f"H{self._handoffs:03d}",
            from_agent=AgentName.MANAGER,
            to_agent=to_agent,
            objective=objective,
            collection_id=request.collection_id,
            user_id=request.user_id,
            payload=payload,
            constraints=constraints,
        )

    def curriculum_handoff(self, request: ProgrammeRequest) -> Handoff:
        return self._handoff(
            AgentName.CURRICULUM,
            objective=f"Plan the programme '{request.programme_title}' from indexed books",
            request=request,
            payload={
                "programme_title": request.programme_title,
                "seed_queries": list(request.seed_queries),
            },
            constraints={
                "max_topics": request.max_topics,
                "capacity_hours": request.capacity_hours,
                "max_semesters": request.max_semesters,
            },
        )

    def _topic_payload(self, request: ProgrammeRequest, topic) -> dict:
        return {
            "topic_id": topic.topic_id,
            "topic_title": topic.title,
            "topic_summary": topic.summary,
            "document_ids": (
                sorted({citation.document_id for citation in topic.citations})
                if request.restrict_to_topic_sources
                else []
            ),
        }

    def content_handoff(self, request: ProgrammeRequest, topic) -> Handoff:
        return self._handoff(
            AgentName.CONTENT,
            objective=f"Draft the lecture for '{topic.title}'",
            request=request,
            payload=self._topic_payload(request, topic),
            constraints={"slide_count": request.slide_count},
        )

    def assessment_handoff(self, request: ProgrammeRequest, topic) -> Handoff:
        return self._handoff(
            AgentName.ASSESSMENT,
            objective=f"Write assessment questions for '{topic.title}'",
            request=request,
            payload=self._topic_payload(request, topic),
            constraints={
                "question_count": request.question_count,
                "assessment_type": request.assessment_type.value,
                "covered_scope": request.assessment_scope or [topic.title],
                "difficulty_distribution": request.difficulty_distribution,
                "allowed_formats": request.allowed_question_formats,
            },
        )

    # ── Task bookkeeping ──────────────────────────────────────────────

    def open_task(self, trace: AgentTrace, handoff: Handoff) -> TaskRecord:
        return trace.add(
            TaskRecord(
                task_id=handoff.handoff_id,
                agent=handoff.to_agent,
                objective=handoff.objective,
                max_attempts=self.runtime.max_attempts,
            )
        )

    def selected_topics(self, plan: ProgrammePlan) -> list:
        """The topics this run will produce material for, in teaching order."""
        ordered = [
            topic for semester in plan.semesters for topic in semester.topics
        ]
        ordered.sort(key=lambda topic: topic.order)
        return ordered[: self.runtime.topic_budget]

    def summarise(
        self,
        request: ProgrammeRequest,
        plan: ProgrammePlan | None,
        lectures: list[Lecture],
        assessments: list[Assessment],
        trace: AgentTrace,
    ) -> GraphResult:
        refusals: list[Refusal] = list(trace.refusals)
        selected = self.selected_topics(plan) if plan is not None else []
        expected_ids = {topic.topic_id for topic in selected}
        completed = plan is not None and {
            lecture.topic_id for lecture in lectures
        } == expected_ids and {
            assessment.topic_id for assessment in assessments
        } == expected_ids
        return GraphResult(
            collection_id=request.collection_id,
            user_id=request.user_id,
            programme_title=request.programme_title,
            plan=plan,
            lectures=lectures,
            assessments=assessments,
            refusals=refusals,
            trace=trace,
            completed=completed,
        )


def ollama_llm(model: str | None = None, base_url: str | None = None) -> Callable[[str], str]:
    """Build the real ``str -> str`` LLM callable used by integrated runs.

    Imported lazily and never used by the unit tests, which inject their own
    callable instead.
    """
    from langchain_ollama import ChatOllama

    from config import LLM_BASE_URL, LLM_MODEL
    from guardrails.prompt_boundary import split_prompt_roles

    client = ChatOllama(
        model=model or LLM_MODEL,
        base_url=base_url or LLM_BASE_URL,
        temperature=0,
    )

    def call(prompt: str) -> str:
        roles = split_prompt_roles(prompt)
        request = (
            [("system", roles[0]), ("human", roles[1])]
            if roles is not None
            else prompt
        )
        response = client.invoke(request)
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    return call


def resilient_ollama_llm() -> "ResilientLLM":
    """The integrated resilient model: primary with a configured fallback.

    Wraps :class:`resilience.fallback.ResilientLLM` so the same agent graph that
    accepts a plain ``str -> str`` callable also records which model served each
    reply and why a fallback happened. Unit tests never call this — they inject
    their own backends via :func:`agents.manager.AgentRuntime`.
    """
    from resilience.fallback import build_resilient_llm

    return build_resilient_llm()


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_TOPIC_BUDGET",
    "AgentRuntime",
    "ManagerAgent",
    "ProgrammeRequest",
    "ollama_llm",
    "resilient_ollama_llm",
]
