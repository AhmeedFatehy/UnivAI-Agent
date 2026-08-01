"""The agent graph end to end: handoffs, grounding, bounded retries, refusal.

The LLM is a scripted ``str -> str`` callable and the vector store is the
in-memory fake from ``conftest``, so this is the whole demo contract — three
books indexed, a cited answer, a grounded refusal and a multi-semester plan —
without a live model or a running Qdrant.
"""

from __future__ import annotations

import json

import pytest

from agents.prompts import (
    PromptId,
    PromptOperation,
    load_prompt_for,
    validate_prompt_catalog,
)
from agents.graph import (
    PROGRAMME_STAGE,
    build_graph,
    manager_node,
    route_from_manager,
    run_programme,
    stage_key,
)
from agents.manager import AgentRuntime, ManagerAgent, ProgrammeRequest
from agents.schemas import (
    AgentName,
    AgentTrace,
    AssessmentType,
    GraphResult,
    Handoff,
    LectureDraftLLM,
    StructuredOutputError,
    TaskState,
    TopicExtraction,
    UngroundedCitation,
    extract_json,
    generate_structured,
    load_prompt,
    resolve_citations,
)
from tests.conftest import COLLECTION_ID, USER_ID, ScriptedLLM
from tools.registry import (
    TOOL_REGISTRY,
    TOOL_SCHEMA_VERSION,
    GroundedContext,
    RetrieveContextInput,
    ToolContext,
    ToolInputError,
    ToolNotFound,
    call_tool,
    tool_manifest,
)

SEEDS = [
    "hash tables and collision handling",
    "sorting algorithms and asymptotic analysis",
    "supervised learning and overfitting",
]


# ── Scripted model replies ────────────────────────────────────────────

TOPICS_JSON = json.dumps(
    {
        "topics": [
            {
                "title": "Hashing and Collisions",
                "summary": "Bucket arrays, hash functions and collision handling by chaining or probing.",
                "keywords": ["hashing", "collision", "chaining"],
                "prerequisites": [],
                "difficulty": 3,
                "source_ids": ["S1"],
            },
            {
                "title": "Sorting and Complexity",
                "summary": "Comparison sorts, the n log n bound, merge sort and quicksort.",
                "keywords": ["merge sort", "quicksort", "big-o"],
                "prerequisites": ["Hashing and Collisions"],
                "difficulty": 4,
                "source_ids": ["S2"],
            },
            {
                "title": "Learning and Overfitting",
                "summary": "Fitting from labelled examples and constraining capacity to generalise.",
                "keywords": ["supervised", "overfitting", "regularisation"],
                "prerequisites": [],
                "difficulty": 3,
                "source_ids": ["S3"],
            },
        ]
    }
)

LECTURE_JSON = json.dumps(
    {
        "title": "Hashing and Collisions",
        "segments": [
            {
                "slide": 1,
                "heading": "Why hashing is fast",
                "text": "A hash table chooses a bucket by applying a hash function to the key, which is what makes lookup, insertion and deletion expected constant time.",
                "source_ids": ["S1"],
            },
            {
                "slide": 2,
                "heading": "Handling collisions",
                "text": "Collisions become unavoidable as keys approach buckets. Chaining keeps a list per bucket, probing looks for the next free slot, and the table is resized once it fills.",
                "source_ids": ["S1"],
            },
        ],
    }
)

QUIZ_JSON = json.dumps(
    {
        "questions": [
            {
                "prompt": "What happens to a hash table as its load factor rises?",
                "options": [
                    "A) Lookup degrades and the table is resized",
                    "B) The hash function is discarded",
                    "C) Keys are sorted automatically",
                    "D) Collisions become impossible",
                ],
                "correct_option": "A",
                "source": "lecture",
                "source_ids": ["S1"],
            }
        ]
    }
)


def scripted_llm(**overrides) -> ScriptedLLM:
    """A model that answers each prompt type, keyed on its system preamble."""
    script = {
        "curriculum analyst": [TOPICS_JSON] * 4,
        "lecturer": [LECTURE_JSON] * 8,
        "assessment author": [QUIZ_JSON] * 8,
    }
    script.update(overrides)
    return ScriptedLLM(script)


@pytest.fixture
def request_() -> ProgrammeRequest:
    return ProgrammeRequest(
        programme_title="Computer Science Foundations",
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        seed_queries=SEEDS,
        capacity_hours=4.0,
        max_semesters=4,
        max_topics=4,
        slide_count=2,
        question_count=1,
    )


@pytest.fixture
def runtime(retriever) -> AgentRuntime:
    return AgentRuntime(llm=scripted_llm(), tool_context=ToolContext(retriever=retriever))


# ── The demo contract ─────────────────────────────────────────────────


def test_the_graph_produces_a_cited_plan_lectures_and_questions(request_, runtime):
    result = run_programme(request_, runtime)

    assert isinstance(result, GraphResult)
    assert result.completed is True
    assert result.plan is not None
    assert result.lectures
    assert result.assessments
    assert result.citation_count() > 0


def test_the_plan_spans_several_semesters_and_explains_the_split(request_, runtime):
    result = run_programme(request_, runtime)

    assert result.plan.is_multi_semester, "a 4 h capacity cannot hold this material"
    decision = next(item for item in result.plan.decisions if item.kind == "semester")
    assert "capacity" in decision.rationale
    assert decision.evidence


def test_every_lecture_segment_cites_a_real_book_page_and_section(request_, runtime):
    result = run_programme(request_, runtime)

    assert result.lectures
    for lecture in result.lectures:
        assert lecture.segments
        for segment in lecture.segments:
            assert segment.citations, "narration without a citation must not ship"
            for citation in segment.citations:
                assert citation.collection_id == COLLECTION_ID
                assert citation.book_title
                assert citation.page >= 1
                assert citation.section


def test_every_question_is_cited_and_matches_the_exam_contract(request_, runtime):
    result = run_programme(request_, runtime)

    assert result.assessments
    for assessment in result.assessments:
        for question in assessment.questions:
            assert len(question.options) == 4
            assert question.correct_option in "ABCD"
            assert question.source in ("lecture", "self_study")
            assert question.citations


def test_the_plan_draws_on_more_than_one_book(request_, runtime):
    result = run_programme(request_, runtime)
    books = {
        citation.book_title
        for citation in result.plan.all_citations()
    }
    assert len(books) > 1, "a multi-book programme must cite more than one book"


# ── Bounded hierarchy and observable state ────────────────────────────


def test_the_manager_is_the_only_router(request_, runtime):
    result = run_programme(request_, runtime)
    agents = {task.agent for task in result.trace.tasks}

    assert AgentName.MANAGER not in agents, "the manager delegates, it does not do the work"
    assert agents <= {AgentName.CURRICULUM, AgentName.CONTENT, AgentName.ASSESSMENT}


def test_handoffs_are_typed_not_free_text(request_, runtime):
    manager = ManagerAgent(runtime)
    handoff = manager.curriculum_handoff(request_)

    assert handoff.from_agent is AgentName.MANAGER
    assert handoff.to_agent is AgentName.CURRICULUM
    assert handoff.collection_id == COLLECTION_ID
    assert handoff.payload["seed_queries"] == SEEDS
    assert handoff.constraints["max_topics"] == 4


def test_an_agent_cannot_hand_off_to_itself(request_, runtime):
    manager = ManagerAgent(runtime)
    payload = manager.curriculum_handoff(request_).model_dump()
    payload["to_agent"] = AgentName.MANAGER.value

    with pytest.raises(ValueError, match="cannot hand off to itself"):
        Handoff.model_validate(payload)


def test_every_task_has_an_observable_state_and_tool_calls(request_, runtime):
    result = run_programme(request_, runtime)
    trace = result.trace

    assert trace.tasks
    assert trace.steps > 0
    for task in trace.tasks:
        assert task.state in (TaskState.SUCCEEDED, TaskState.REFUSED, TaskState.FAILED)
        assert task.attempts >= 1
        assert task.started_at and task.finished_at

    assert trace.tool_calls > 0
    assert trace.llm_calls > 0
    assert all(record.tool for task in trace.tasks for record in task.tool_calls)


def test_the_curriculum_agent_calls_both_of_its_tools(request_, runtime):
    result = run_programme(request_, runtime)
    task = result.trace.by_agent(AgentName.CURRICULUM)[0]
    tools_used = {record.tool for record in task.tool_calls}

    assert "retrieve_context" in tools_used
    assert "create_programme_plan" in tools_used


def test_the_manager_stops_when_the_step_budget_runs_out(request_, retriever):
    runtime = AgentRuntime(
        llm=scripted_llm(),
        tool_context=ToolContext(retriever=retriever),
        max_steps=1,
    )
    result = run_programme(request_, runtime)

    assert result.trace.steps <= 2
    assert result.plan is None or not result.lectures


def test_the_manager_stops_routing_once_a_stage_is_settled(request_, runtime):
    state = {
        "request": request_,
        "plan": None,
        "trace": AgentTrace(),
        "manager": ManagerAgent(runtime),
        "runtime": runtime,
        "steps": 0,
        "stage_status": {stage_key(PROGRAMME_STAGE, AgentName.CURRICULUM): "refused"},
        "stage_attempts": {stage_key(PROGRAMME_STAGE, AgentName.CURRICULUM): 1},
    }
    update = manager_node(state)

    assert route_from_manager(update) == "done"
    assert update["handoff"] is None


# ── Bounded retries and one bounded repair ────────────────────────────


def test_malformed_output_is_repaired_once_and_then_accepted(request_, retriever):
    llm = scripted_llm(**{"lecturer": ["not json at all"] + [LECTURE_JSON] * 8})
    runtime = AgentRuntime(llm=llm, tool_context=ToolContext(retriever=retriever))

    result = run_programme(request_, runtime)

    assert result.lectures, "one repair attempt must be enough to recover"
    content_tasks = result.trace.by_agent(AgentName.CONTENT)
    assert content_tasks[0].llm_calls == 2, "one call plus exactly one repair"


def test_output_that_stays_malformed_fails_the_task_without_looping(request_, retriever):
    llm = ScriptedLLM(
        {
            "curriculum analyst": [TOPICS_JSON] * 4,
            "lecturer": ["still not json"] * 20,
            "assessment author": [QUIZ_JSON] * 8,
        }
    )
    runtime = AgentRuntime(
        llm=llm, tool_context=ToolContext(retriever=retriever), max_attempts=2
    )

    result = run_programme(request_, runtime)

    content_tasks = result.trace.by_agent(AgentName.CONTENT)
    assert content_tasks, "the manager must have tried"
    assert all(task.state is TaskState.FAILED for task in content_tasks)
    assert all(task.llm_calls == 2 for task in content_tasks), "one call plus one repair"
    assert result.lectures == []
    assert result.completed is False
    assert result.trace.steps <= runtime.max_steps + 1


def test_curriculum_failure_returns_an_explained_result(request_, retriever):
    llm = ScriptedLLM({"curriculum analyst": ["not json"] * 20})
    runtime = AgentRuntime(
        llm=llm, tool_context=ToolContext(retriever=retriever), max_attempts=1
    )

    result = run_programme(request_, runtime)

    assert result.plan is None
    assert result.completed is False
    assert result.trace.by_agent(AgentName.CURRICULUM)[0].state is TaskState.FAILED


def test_generate_structured_raises_rather_than_returning_garbage():
    llm = ScriptedLLM({"": ['{"topics": []}'] * 5})
    with pytest.raises(StructuredOutputError) as error:
        generate_structured(llm, "prompt", TopicExtraction, repair_attempts=1)

    assert error.value.attempts == 2
    assert "topics" in error.value.last_error
    assert llm.calls == 2


def test_the_repair_prompt_carries_the_validation_error():
    llm = ScriptedLLM({"": ["{}", TOPICS_JSON]})
    generate_structured(llm, "original prompt", TopicExtraction, repair_attempts=1)

    repair = llm.prompts[1]
    assert "--- REPAIR ---" in repair
    assert "original prompt" in repair
    assert "Validation errors" in repair


def test_extract_json_survives_fences_and_chatter():
    assert json.loads(extract_json('Sure!\n```json\n{"a": 1}\n```\nHope that helps'))
    assert json.loads(extract_json('{"a": {"b": "}"}}')) == {"a": {"b": "}"}}


# ── Grounding: the model cannot invent a citation ─────────────────────


def test_a_fabricated_passage_id_is_rejected(request_, retriever):
    fabricated = json.dumps(
        {
            "title": "Invented",
            "segments": [
                {
                    "slide": 1,
                    "heading": "From nowhere",
                    "text": "This narration claims support from a passage that was never shown.",
                    "source_ids": ["S99"],
                }
            ],
        }
    )
    llm = ScriptedLLM(
        {
            "curriculum analyst": [TOPICS_JSON] * 4,
            "lecturer": [fabricated] * 20,
            "assessment author": [QUIZ_JSON] * 8,
        }
    )
    runtime = AgentRuntime(
        llm=llm, tool_context=ToolContext(retriever=retriever), max_attempts=1
    )

    result = run_programme(request_, runtime)

    assert result.lectures == [], "a lecture citing an unseen passage must not ship"
    failed = [
        task
        for task in result.trace.by_agent(AgentName.CONTENT)
        if task.state is TaskState.FAILED
    ]
    assert failed
    assert "S99" in failed[0].error


def test_resolve_citations_rejects_unknown_ids(retriever):
    context = call_tool(
        "retrieve_context",
        RetrieveContextInput(
            query="hash table collisions", user_id=USER_ID, collection_id=COLLECTION_ID
        ),
        ToolContext(retriever=retriever),
    )
    assert isinstance(context, GroundedContext) and context.grounded

    resolved = resolve_citations([context.passages[0].passage_id], context)
    assert resolved == [context.passages[0].citation]

    with pytest.raises(UngroundedCitation, match="S404"):
        resolve_citations(["S404"], context)


def test_a_lecture_cannot_be_built_without_citations():
    with pytest.raises(ValueError):
        LectureDraftLLM.model_validate(
            {
                "title": "T",
                "segments": [
                    {"slide": 1, "heading": "H", "text": "body", "source_ids": []}
                ],
            }
        )


def test_a_passage_id_must_look_like_a_passage_id():
    with pytest.raises(ValueError, match="passage ids"):
        LectureDraftLLM.model_validate(
            {
                "title": "T",
                "segments": [
                    {
                        "slide": 1,
                        "heading": "H",
                        "text": "body",
                        "source_ids": ["Foundations of Algorithms, page 3"],
                    }
                ],
            }
        )


# ── Grounded refusal through the whole graph ──────────────────────────


def test_an_empty_knowledge_base_refuses_instead_of_inventing_a_programme(request_):
    runtime = AgentRuntime(
        llm=scripted_llm(), tool_context=ToolContext(retriever=lambda **kwargs: [])
    )

    result = run_programme(request_, runtime)

    assert result.plan is None
    assert result.lectures == []
    assert result.assessments == []
    assert result.refusals, "a run with no evidence must say why"
    assert result.refusals[0].reason
    assert result.trace.by_agent(AgentName.CURRICULUM)[0].state is TaskState.REFUSED
    assert result.completed is False


def test_a_refusal_costs_no_llm_call(request_):
    runtime = AgentRuntime(
        llm=scripted_llm(), tool_context=ToolContext(retriever=lambda **kwargs: [])
    )
    result = run_programme(request_, runtime)

    assert runtime.llm.calls == 0, "there is nothing to generate from, so do not generate"
    assert result.trace.llm_calls == 0


def test_a_topic_with_no_retrievable_evidence_refuses_its_lecture(request_, retriever):
    """Retrieval works for the plan, then stops working for the drafts."""
    state = {"calls": 0}

    def failing_after_planning(**kwargs):
        state["calls"] += 1
        return retriever(**kwargs) if state["calls"] <= len(SEEDS) else []

    runtime = AgentRuntime(
        llm=scripted_llm(), tool_context=ToolContext(retriever=failing_after_planning)
    )
    result = run_programme(request_, runtime)

    assert result.plan is not None
    assert result.lectures == []
    refused = [
        task
        for task in result.trace.by_agent(AgentName.CONTENT)
        if task.state is TaskState.REFUSED
    ]
    assert refused
    assert refused[0].refusal is not None
    assert refused[0].refusal.reason
    assert result.completed is False


# ── The tool layer is typed on both sides ─────────────────────────────


def test_the_registry_exposes_the_contracted_tools():
    assert {
        "ingest_collection",
        "retrieve_context",
        "create_programme_plan",
        "get_source_location",
    } <= set(TOOL_REGISTRY)

    for spec in TOOL_REGISTRY.values():
        schema = spec.json_schema()
        assert schema["version"] == TOOL_SCHEMA_VERSION
        assert schema["description"]
        assert schema["input_schema"]["type"] == "object"
        assert schema["output_schema"]


def test_manifest_covers_every_registered_tool():
    assert {item["name"] for item in tool_manifest()} == set(TOOL_REGISTRY)


def test_a_malformed_tool_argument_is_rejected_before_the_tool_runs(retriever):
    with pytest.raises(ToolInputError, match="retrieve_context"):
        call_tool(
            "retrieve_context",
            {"query": "", "user_id": USER_ID},
            ToolContext(retriever=retriever),
        )
    assert retriever.calls == [], "an invalid call must never reach the retriever"


def test_an_unknown_tool_is_refused_by_name():
    with pytest.raises(ToolNotFound, match="retrieve_everything"):
        call_tool("retrieve_everything", {})


# ── No live model, ever ───────────────────────────────────────────────


def test_the_suite_only_ever_calls_the_injected_model(request_, runtime):
    run_programme(request_, runtime)

    assert runtime.llm.calls > 0
    assert all(
        "EVIDENCE" in prompt or "--- REPAIR ---" in prompt
        for prompt in runtime.llm.prompts
    ), "every prompt is built from a template with retrieved evidence"


def test_prompts_are_versioned_and_declare_their_variables():
    for name in ("programme_planner", "lecture", "assessment"):
        template = load_prompt(name)
        assert template.version.count(".") == 2
        assert template.variables
        assert "evidence" in template.variables
        assert template.owner
        assert template.output_schema
        assert template.grounding_policy


def test_every_prompt_route_resolves_to_a_declared_template():
    catalog = validate_prompt_catalog()
    assert set(catalog) == set(PromptId)


def test_the_trace_records_prompt_ids_and_versions(request_, runtime):
    result = run_programme(request_, runtime)
    uses = [prompt for task in result.trace.tasks for prompt in task.prompts]
    assert uses
    assert all(prompt.prompt_id and prompt.version for prompt in uses)


@pytest.mark.parametrize(
    ("assessment_type", "operation", "prompt_id"),
    [
        (AssessmentType.DIAGNOSTIC, PromptOperation.ASSESSMENT_DIAGNOSTIC, "assessment/diagnostic"),
        (AssessmentType.PRACTICE, PromptOperation.ASSESSMENT_PRACTICE, "assessment/practice"),
        (AssessmentType.QUIZ, PromptOperation.ASSESSMENT_QUIZ, "assessment/quiz"),
        (AssessmentType.ASSIGNMENT, PromptOperation.ASSESSMENT_ASSIGNMENT, "assessment/assignment"),
        (AssessmentType.MIDTERM, PromptOperation.ASSESSMENT_MIDTERM, "assessment/midterm"),
        (AssessmentType.FINAL, PromptOperation.ASSESSMENT_FINAL, "assessment/final"),
        (AssessmentType.ORAL_EXAM, PromptOperation.ASSESSMENT_ORAL, "assessment/oral_exam"),
    ],
)
def test_each_assessment_type_has_its_own_prompt(assessment_type, operation, prompt_id):
    template = load_prompt_for(operation)
    assert template.name.value == prompt_id
    assert assessment_type.value in template.name.value


def test_a_prompt_refuses_to_render_with_a_missing_variable():
    template = load_prompt("lecture")
    with pytest.raises(KeyError, match="topic_summary"):
        template.render(topic_title="T", slide_count=2, evidence="[S1] …")


def test_prompt_rendering_leaves_the_json_example_intact():
    rendered = load_prompt("lecture").render(
        topic_title="Hashing",
        topic_summary="Buckets and collisions.",
        slide_count=2,
        evidence="[S1] Book — p. 1",
    )
    assert '"source_ids": ["S1"]' in rendered
    assert "Hashing" in rendered
    assert "{topic_title}" not in rendered
