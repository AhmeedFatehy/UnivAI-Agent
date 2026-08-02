"""The minimal agent graph: Manager → Curriculum, Content, Assessment.

A LangGraph ``StateGraph`` with four nodes. The manager is the only router; the
specialists always return to it. Every edge out of the manager is chosen from a
typed handoff, never from parsed prose, and the router stops once ``max_steps``
is reached — so the graph cannot loop.

    START → manager ─┬→ curriculum ─┐
                     ├→ content ────┤→ manager → … → END
                     └→ assessment ─┘

Progress is carried in the state itself: ``stage_status`` and ``stage_attempts``
say, for every (topic, specialist) pair, what happened and how many times it was
tried. That is what makes the retries bounded and the run inspectable — the same
two maps the manager routes on are the ones a reviewer reads afterwards.

Run it::

    from agents.graph import run_programme
    result = run_programme(request, runtime)
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.assessment import AssessmentAgent
from agents.content import ContentAgent
from agents.curriculum import CurriculumAgent
from agents.manager import AgentRuntime, ManagerAgent, ProgrammeRequest
from agents.schemas import (
    AgentName,
    AgentTrace,
    Assessment,
    GraphResult,
    Handoff,
    Lecture,
    TaskRecord,
    TaskState,
)
from planning.programme_planner import ProgrammePlan

logger = logging.getLogger(__name__)

#: Stage key for the one-off curriculum task, which has no topic of its own.
PROGRAMME_STAGE = "programme"

#: States that mean "do not hand this off again".
SETTLED_STATES = (TaskState.SUCCEEDED, TaskState.REFUSED)


def _append(left: list, right: list) -> list:
    return (left or []) + (right or [])


def _merge(left: dict, right: dict) -> dict:
    return {**(left or {}), **(right or {})}


def stage_key(topic_id: str, agent: AgentName) -> str:
    return f"{topic_id}|{agent.value}"


class GraphState(TypedDict, total=False):
    """State carried between nodes. Lists append, dicts merge, scalars replace."""

    request: ProgrammeRequest
    plan: ProgrammePlan | None
    lectures: Annotated[list[Lecture], _append]
    assessments: Annotated[list[Assessment], _append]
    stage_status: Annotated[dict[str, str], _merge]
    stage_attempts: Annotated[dict[str, int], _merge]
    pending_topics: list[Any]
    current_topic: Any
    handoff: Handoff | None
    next_agent: str
    trace: AgentTrace
    steps: int
    manager: ManagerAgent
    runtime: AgentRuntime


# ── Progress helpers ──────────────────────────────────────────────────


def _settled(state: GraphState, key: str, max_attempts: int) -> bool:
    """Is this stage finished, one way or another?

    Finished means it succeeded, it refused for lack of evidence, or it failed
    ``max_attempts`` times. The third case is the bounded retry.
    """
    status = (state.get("stage_status") or {}).get(key)
    if status in {item.value for item in SETTLED_STATES}:
        return True
    return (state.get("stage_attempts") or {}).get(key, 0) >= max_attempts


def _record(state: GraphState, key: str, task: TaskRecord) -> dict:
    attempts = (state.get("stage_attempts") or {}).get(key, 0) + 1
    return {
        "stage_status": {key: task.state.value},
        "stage_attempts": {key: attempts},
    }


# ── Nodes ─────────────────────────────────────────────────────────────


def manager_node(state: GraphState) -> dict:
    """Decide the next handoff, or stop.

    Order of business: get a plan, then a lecture and an assessment for each
    budgeted topic. A stage that is settled is never handed off again.
    """
    manager: ManagerAgent = state["manager"]
    runtime: AgentRuntime = state["runtime"]
    request: ProgrammeRequest = state["request"]
    trace: AgentTrace = state["trace"]
    steps = state.get("steps", 0) + 1
    trace.steps = steps

    stop = {"steps": steps, "next_agent": "done", "handoff": None}

    if steps > runtime.max_steps:
        logger.warning("Manager stopped: step budget %d exhausted", runtime.max_steps)
        return stop

    # 1. The programme plan comes first; nothing else is possible without it.
    plan = state.get("plan")
    if plan is None:
        key = stage_key(PROGRAMME_STAGE, AgentName.CURRICULUM)
        if _settled(state, key, runtime.max_attempts):
            logger.info("Manager stopped: no programme plan could be produced")
            return stop
        return {
            "steps": steps,
            "next_agent": AgentName.CURRICULUM.value,
            "handoff": manager.curriculum_handoff(request),
        }

    # 2. Walk the budgeted topics, content then assessment.
    pending = list(state.get("pending_topics") or [])
    current = state.get("current_topic")

    while True:
        if current is None:
            if not pending:
                return stop
            current = pending.pop(0)

        content_key = stage_key(current.topic_id, AgentName.CONTENT)
        if not _settled(state, content_key, runtime.max_attempts):
            return {
                "steps": steps,
                "next_agent": AgentName.CONTENT.value,
                "handoff": manager.content_handoff(request, current),
                "pending_topics": pending,
                "current_topic": current,
            }

        assessment_key = stage_key(current.topic_id, AgentName.ASSESSMENT)
        if not _settled(state, assessment_key, runtime.max_attempts):
            return {
                "steps": steps,
                "next_agent": AgentName.ASSESSMENT.value,
                "handoff": manager.assessment_handoff(request, current),
                "pending_topics": pending,
                "current_topic": current,
            }

        current = None  # this topic is done; try the next one


def curriculum_node(state: GraphState) -> dict:
    manager: ManagerAgent = state["manager"]
    handoff: Handoff = state["handoff"]
    task = manager.open_task(state["trace"], handoff)

    plan = CurriculumAgent(state["runtime"]).run(handoff, task)
    update = _record(state, stage_key(PROGRAMME_STAGE, AgentName.CURRICULUM), task)

    if plan is None:
        return update
    return {**update, "plan": plan, "pending_topics": manager.selected_topics(plan)}


def content_node(state: GraphState) -> dict:
    manager: ManagerAgent = state["manager"]
    handoff: Handoff = state["handoff"]
    task = manager.open_task(state["trace"], handoff)

    lecture = ContentAgent(state["runtime"]).run(handoff, task)
    key = stage_key(handoff.payload["topic_id"], AgentName.CONTENT)
    return {
        **_record(state, key, task),
        "lectures": [lecture] if lecture is not None else [],
    }


def assessment_node(state: GraphState) -> dict:
    manager: ManagerAgent = state["manager"]
    handoff: Handoff = state["handoff"]
    task = manager.open_task(state["trace"], handoff)

    assessment = AssessmentAgent(state["runtime"]).run(handoff, task)
    key = stage_key(handoff.payload["topic_id"], AgentName.ASSESSMENT)
    return {
        **_record(state, key, task),
        "assessments": [assessment] if assessment is not None else [],
    }


# ── Routing ───────────────────────────────────────────────────────────


def route_from_manager(state: GraphState) -> str:
    return state.get("next_agent", "done")


# ── Assembly ──────────────────────────────────────────────────────────


def build_graph():
    """Compile the manager/specialist graph."""
    builder = StateGraph(GraphState)
    builder.add_node("manager", manager_node)
    builder.add_node("curriculum", curriculum_node)
    builder.add_node("content", content_node)
    builder.add_node("assessment", assessment_node)

    builder.add_edge(START, "manager")
    builder.add_conditional_edges(
        "manager",
        route_from_manager,
        {
            AgentName.CURRICULUM.value: "curriculum",
            AgentName.CONTENT.value: "content",
            AgentName.ASSESSMENT.value: "assessment",
            "done": END,
        },
    )
    builder.add_edge("curriculum", "manager")
    builder.add_edge("content", "manager")
    builder.add_edge("assessment", "manager")
    return builder.compile()


def run_programme(
    request: ProgrammeRequest, runtime: AgentRuntime, graph=None
) -> GraphResult:
    """Run the graph end to end and return the validated result."""
    graph = graph or build_graph()
    manager = ManagerAgent(runtime)
    trace = AgentTrace()
    trace.fingerprint = runtime.fingerprint

    initial: GraphState = {
        "request": request,
        "plan": None,
        "lectures": [],
        "assessments": [],
        "stage_status": {},
        "stage_attempts": {},
        "pending_topics": [],
        "current_topic": None,
        "handoff": None,
        "next_agent": AgentName.CURRICULUM.value,
        "trace": trace,
        "steps": 0,
        "manager": manager,
        "runtime": runtime,
    }

    # LangGraph's recursion guard is a second, independent bound: each manager
    # turn costs two graph steps, plus the entry and exit hops.
    final = graph.invoke(initial, {"recursion_limit": runtime.max_steps * 2 + 4})

    return manager.summarise(
        request,
        final.get("plan"),
        list(final.get("lectures") or []),
        list(final.get("assessments") or []),
        trace,
    )


__all__ = [
    "PROGRAMME_STAGE",
    "SETTLED_STATES",
    "GraphState",
    "assessment_node",
    "build_graph",
    "content_node",
    "curriculum_node",
    "manager_node",
    "route_from_manager",
    "run_programme",
    "stage_key",
]
