"""Curriculum agent — turn indexed books into a programme plan.

The division of labour matters: the model is used only to *name* structure it
can see in the retrieved passages. Every consequential decision — what overlaps,
what must come first, how many hours, how many semesters — is computed by
``planning/`` from the evidence the model cited. A model that hallucinates a
topic cites a passage id it was never shown and is rejected before it reaches
the planner.
"""

from __future__ import annotations

from agents.schemas import (
    AgentName,
    Handoff,
    ServingRecord,
    StructuredOutputError,
    TaskRecord,
    ToolCallRecord,
    TopicExtraction,
    UngroundedCitation,
    generate_structured,
    PromptUseRecord,
    topics_from_extraction,
)
from agents.prompts import PromptOperation, load_prompt_for
from planning.programme_planner import ProgrammePlan
from planning.learning_path import (
    ApprovedBook,
    LearningPathV1,
    PrerequisiteProposal,
    generate_learning_path,
)
from tools.registry import (
    CreateProgrammePlanInput,
    GroundedContext,
    Refusal,
    RetrieveContextInput,
    call_tool,
)

DEFAULT_MAX_TOPICS = 8
DEFAULT_EVIDENCE_LIMIT = 12


class CurriculumAgent:
    """Retrieve → extract topics → plan. Refuses rather than inventing a syllabus."""

    name = AgentName.CURRICULUM

    def __init__(self, runtime):
        self.runtime = runtime

    @staticmethod
    def create_learning_path(
        books: list[ApprovedBook],
        proposals: list[PrerequisiteProposal],
        *,
        collection_id: str,
        user_id: str,
    ) -> LearningPathV1:
        """Validate an AI proposal and return the versioned App contract."""
        return generate_learning_path(
            books, proposals, collection_id=collection_id, user_id=user_id
        )

    def run(self, handoff: Handoff, task: TaskRecord) -> ProgrammePlan | None:
        task.start()

        seeds: list[str] = handoff.payload.get("seed_queries") or [handoff.objective]
        max_topics = int(handoff.constraints.get("max_topics", DEFAULT_MAX_TOPICS))
        evidence_limit = int(
            handoff.constraints.get("evidence_limit", DEFAULT_EVIDENCE_LIMIT)
        )

        context = self._gather(handoff, task, seeds, evidence_limit)
        if context is None or not context.grounded:
            refusal = (
                context.refusal
                if context is not None and context.refusal is not None
                else Refusal(
                    reason="No indexed material matched the programme's subject areas.",
                    query="; ".join(seeds),
                    scope={
                        "collection_id": handoff.collection_id,
                        "user_id": handoff.user_id,
                    },
                )
            )
            task.refuse(refusal)
            return None

        template = load_prompt_for(PromptOperation.CURRICULUM_EXTRACT_TOPICS)
        task.prompts.append(
            PromptUseRecord(
                operation=PromptOperation.CURRICULUM_EXTRACT_TOPICS.value,
                prompt_id=template.name.value,
                version=template.version,
            )
        )
        prompt = template.render(
            programme_title=handoff.payload.get("programme_title", handoff.objective),
            max_topics=max_topics,
            evidence=context.as_prompt_block(),
        )

        try:
            extraction = generate_structured(
                self.runtime.llm,
                prompt,
                TopicExtraction,
                repair_attempts=self.runtime.repair_attempts,
                on_call=lambda: setattr(task, "llm_calls", task.llm_calls + 1),
                on_served=lambda serving: task.record_serving(
                    ServingRecord(
                        **serving.model_dump(exclude_none=True),
                        prompt_id=template.name.value,
                        prompt_version=template.version,
                    )
                ),
            )
            topics = topics_from_extraction(extraction, context)
        except (StructuredOutputError, UngroundedCitation, ValueError) as error:
            task.fail(f"topic extraction rejected: {error}")
            return None

        try:
            plan = call_tool(
                "create_programme_plan",
                CreateProgrammePlanInput(
                    topics=topics,
                    programme_title=handoff.payload.get(
                        "programme_title", handoff.objective
                    ),
                    collection_id=handoff.collection_id,
                    capacity_hours=float(
                        handoff.constraints.get("capacity_hours", 120.0)
                    ),
                    max_semesters=int(handoff.constraints.get("max_semesters", 8)),
                ),
                self.runtime.tool_context,
            )
        except Exception as error:  # noqa: BLE001 — recorded on the task, not swallowed
            task.tool_calls.append(
                ToolCallRecord(tool="create_programme_plan", ok=False, detail=str(error))
            )
            task.fail(f"programme planning failed: {error}")
            return None

        assert isinstance(plan, ProgrammePlan)
        task.tool_calls.append(
            ToolCallRecord(
                tool="create_programme_plan",
                ok=True,
                detail=f"{len(plan.semesters)} semester(s), {plan.total_hours} h",
                citations=len(plan.all_citations()),
            )
        )
        task.succeed()
        return plan

    def _gather(
        self,
        handoff: Handoff,
        task: TaskRecord,
        seeds: list[str],
        evidence_limit: int,
    ) -> GroundedContext | None:
        """Retrieve evidence for each seed subject and merge into one context.

        Passage ids are renumbered across the merged set so the model sees one
        continuous, unambiguous evidence block.
        """
        merged: list = []
        seen: set[tuple[str, int | None]] = set()
        last_refusal: Refusal | None = None

        for seed in seeds:
            result = call_tool(
                "retrieve_context",
                RetrieveContextInput(
                    query=seed,
                    user_id=handoff.user_id,
                    collection_id=handoff.collection_id,
                    limit=evidence_limit,
                    use_query_transform=bool(
                        handoff.constraints.get("use_query_transform", False)
                    ),
                ),
                self.runtime.tool_context,
            )
            assert isinstance(result, GroundedContext)
            task.tool_calls.append(
                ToolCallRecord(
                    tool="retrieve_context",
                    ok=True,
                    detail=f"seed '{seed}'",
                    grounded=result.grounded,
                    citations=len(result.passages),
                )
            )
            if not result.grounded:
                last_refusal = result.refusal
                continue
            for passage in result.passages:
                key = (passage.citation.document_id, passage.citation.chunk_index)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(passage)

        if not merged:
            return GroundedContext(
                query="; ".join(seeds),
                grounded=False,
                refusal=last_refusal
                or Refusal(
                    reason="No indexed material matched the programme's subject areas.",
                    query="; ".join(seeds),
                    scope={"collection_id": handoff.collection_id},
                ),
            )

        merged.sort(key=lambda passage: -passage.score)
        renumbered = [
            passage.model_copy(update={"passage_id": f"S{index}", "rank": index})
            for index, passage in enumerate(merged[: evidence_limit * 2], start=1)
        ]
        return GroundedContext(
            query="; ".join(seeds), grounded=True, passages=renumbered
        )


__all__ = ["DEFAULT_EVIDENCE_LIMIT", "DEFAULT_MAX_TOPICS", "CurriculumAgent"]
