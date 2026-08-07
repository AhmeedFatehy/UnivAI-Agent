"""Section generation: grounded packs, bounded repair, refusals, persistence.

The model is a ``ScriptedLLM`` and retrieval is the in-memory fake from
``conftest``, so the whole contract — grounded packs with resolved citations,
malformed-JSON failure, repair recovery and explicit refusals for unsupported
content — runs without a live model or Qdrant.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agents.schemas import SectionPackV1
from generation.section_gen import (
    SectionGenerationError,
    generate_section_pack,
    publish_section_contract,
    write_section_artifact,
)
from planning.section_planner import SectionBudget, SectionIdentity
from tests.conftest import COLLECTION_ID, USER_ID, ScriptedLLM
from tools.registry import ToolContext

SECTION_MARKER = "post-lecture section pack"

SECTION_JSON = json.dumps(
    {
        "title": "Hashing — Section Practice",
        "objectives": ["Resolve collisions by chaining and probing."],
        "activities": [
            {"title": "Chain a table by hand", "description": "Bucket three keys.", "duration_minutes": 15, "source_ids": ["S1"]},
            {"title": "Trace linear probing", "description": "Record each probe.", "duration_minutes": 20, "source_ids": ["S1"]},
        ],
        "examples": [
            {
                "order": 1,
                "prompt": "Insert three keys into a chained table of size three.",
                "steps": [
                    {"step": "hash to a bucket", "explanation": "the hash chooses a bucket index", "source_ids": ["S1"]},
                    {"step": "append to the list", "explanation": "chaining keeps colliders in one bucket", "source_ids": ["S1"]},
                ],
                "conclusion": "lookups stay near constant time.",
                "source_ids": ["S1"],
            }
        ],
        "todos": [
            {"text": "Rework the probe example from memory.", "time_box_minutes": 10, "source_ids": ["S1"]},
        ],
    }
)

EMPTY_SECTION_JSON = json.dumps(
    {
        "title": "Hashing — Section Practice",
        "objectives": ["Collision handling."],
        "activities": [
            {"title": "Discuss", "description": "Talk through open questions.", "duration_minutes": 30, "source_ids": ["S1"]},
        ],
        "examples": [],
        "todos": [],
    }
)


def identity(**overrides) -> SectionIdentity:
    values = dict(
        programme_title="Computer Science Foundations",
        plan_schema="univai.programme.plan",
        plan_version="1.3.0",
        user_id=USER_ID,
        collection_id=COLLECTION_ID,
        course_id="CS101",
        week_number=3,
        topic_id="T01",
        lecture_title="Hashing and Collisions",
        created_at="2026-08-03T00:00:00+00:00",
    )
    values.update(overrides)
    return SectionIdentity(**values)


def scripted(**overrides) -> ScriptedLLM:
    script = {SECTION_MARKER: [SECTION_JSON] * 8}
    script.update(overrides)
    return ScriptedLLM(script)


def run_section(llm, retriever, **kwargs):
    return generate_section_pack(
        llm=llm,
        identity=identity(),
        tool_context=ToolContext(retriever=retriever),
        **kwargs,
    )


def test_a_section_is_grounded_versioned_and_distinct_from_a_lecture(retriever):
    run = run_section(scripted(), retriever)

    section = run.section
    assert section is not None
    assert section.session_type == "section"
    assert section.schema_name == "univai.section.pack"
    assert section.plan_version == "1.3.0"
    assert section.topic_id == "T01"
    assert section.total_minutes == 35
    assert len(section.examples) == 1
    assert len(section.todos) == 1
    assert all(example.citations for example in section.examples)
    assert all(step.citations for example in section.examples for step in example.steps)
    assert all(todo.citations for todo in section.todos)
    assert run.llm_calls == 1
    assert run.prompt_id == "teaching/section_generation"


def test_section_records_the_versioned_prompt_and_its_evidence(retriever):
    run = run_section(scripted(), retriever)

    assert run.prompt_version.count(".") == 2
    assert "Hashing and Collisions" in run.prompts[0]
    assert "EVIDENCE" in run.prompts[0]
    assert "[S1]" in run.prompts[0]


def test_a_malformed_reply_is_repaired_once_then_accepted(retriever):
    llm = scripted(**{SECTION_MARKER: ["not json at all", SECTION_JSON] * 4})
    run = run_section(llm, retriever)

    assert run.section is not None
    assert run.llm_calls == 2, "one failed call plus exactly one repair"


def test_a_reply_that_stays_malformed_fails_explicitly(retriever):
    llm = ScriptedLLM({SECTION_MARKER: ["still not json"] * 10})
    with pytest.raises(SectionGenerationError, match="JSON"):
        run_section(llm, retriever)


def test_empty_content_produces_a_refusal_not_a_section(retriever):
    llm = scripted(**{SECTION_MARKER: [EMPTY_SECTION_JSON, EMPTY_SECTION_JSON]})
    run = run_section(llm, retriever)

    assert run.section is None
    assert run.refusal is not None
    assert "no worked example" in run.refusal.reason.lower() or "no section" in run.refusal.reason.lower()
    assert run.llm_calls == 2


def test_empty_grounded_draft_gets_one_semantic_repair(retriever):
    llm = scripted(**{SECTION_MARKER: [EMPTY_SECTION_JSON, SECTION_JSON]})
    run = run_section(llm, retriever)

    assert run.section is not None
    assert run.refusal is None
    assert run.llm_calls == 2


def test_a_fabricated_citation_is_refused_not_published(retriever):
    fabricated = SECTION_JSON.replace('"S2"', '"S404"').replace("S1", "S99")
    llm = scripted(**{SECTION_MARKER: [fabricated]})
    run = run_section(llm, retriever)

    assert run.section is None
    assert run.refusal is not None
    assert "ground" in run.refusal.reason.lower() or "source" in run.refusal.reason.lower()


def test_missing_evidence_refuses_without_an_llm_call():
    llm = ScriptedLLM({SECTION_MARKER: [SECTION_JSON]})
    run = run_section(
        llm, lambda **kwargs: [], focus="something entirely absent from the books"
    )

    assert run.section is None
    assert run.refusal is not None
    assert run.llm_calls == 0


def test_a_duration_outside_budget_is_refused(retriever):
    budget = SectionBudget(min_total_minutes=40, max_total_minutes=45)
    run = run_section(scripted(), retriever, budget=budget)

    assert run.section is None
    assert run.refusal is not None


def test_section_artifact_is_persisted_separate_from_lecture_folders(retriever, tmp_path):
    run = run_section(scripted(), retriever)
    assert run.section is not None

    path = write_section_artifact(run.section, tmp_path, tenant_id=USER_ID)

    assert path.is_file()
    assert "sections" in path.parts and USER_ID in str(path)
    assert "week-" not in str(path), "a section must not live inside a lecture folder"
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema_name"] == "univai.section.pack"
    assert parsed["session_type"] == "section"


def test_the_published_fixture_is_versioned_and_self_describing(retriever, tmp_path):
    run = run_section(scripted(), retriever)
    assert run.section is not None

    artifact = publish_section_contract(run.section, tmp_path / "contracts")

    contract = json.loads(artifact.read_text(encoding="utf-8"))
    assert contract["schema_name"] == "univai.section.pack"
    assert contract["schema_version"] == "1.0.0"
    assert artifact.name == "section-T01.v1.0.0.json"
