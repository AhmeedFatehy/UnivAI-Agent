"""Assessment agent — write cited questions for a planned topic.

Same contract as the content agent: retrieve, generate against a schema, then
resolve the model's passage ids into real citations. A question whose support
cannot be traced to indexed material does not ship.

The output shape matches the exam system's question contract in ``contracts.py``
— four options, a single correct letter, and a ``lecture``/``self_study`` source
— so a generated quiz can cross the repository boundary unchanged.
"""

from __future__ import annotations

from agents.schemas import (
    AgentName,
    Assessment,
    AssessmentDraftLLM,
    AssessmentQuestion,
    Handoff,
    StructuredOutputError,
    TaskRecord,
    ToolCallRecord,
    UngroundedCitation,
    generate_structured,
    PromptUseRecord,
    resolve_citations,
)
from agents.prompts import PromptOperation, load_prompt_for
from tools.registry import GroundedContext, RetrieveContextInput, call_tool

DEFAULT_QUESTION_COUNT = 4
DEFAULT_EVIDENCE_LIMIT = 6


class AssessmentAgent:
    """Turn one planned topic into cited questions, or refuse."""

    name = AgentName.ASSESSMENT

    def __init__(self, runtime):
        self.runtime = runtime

    def run(self, handoff: Handoff, task: TaskRecord) -> Assessment | None:
        task.start()

        topic_id = handoff.payload["topic_id"]
        topic_title = handoff.payload["topic_title"]
        question_count = int(
            handoff.constraints.get("question_count", DEFAULT_QUESTION_COUNT)
        )
        evidence_limit = int(
            handoff.constraints.get("evidence_limit", DEFAULT_EVIDENCE_LIMIT)
        )

        context = call_tool(
            "retrieve_context",
            RetrieveContextInput(
                query=f"{topic_title}. {handoff.payload.get('topic_summary', '')}".strip(),
                user_id=handoff.user_id,
                collection_id=handoff.collection_id,
                document_ids=handoff.payload.get("document_ids") or [],
                limit=evidence_limit,
            ),
            self.runtime.tool_context,
        )
        assert isinstance(context, GroundedContext)
        task.tool_calls.append(
            ToolCallRecord(
                tool="retrieve_context",
                ok=True,
                detail=f"topic '{topic_title}'",
                grounded=context.grounded,
                citations=len(context.passages),
            )
        )

        if not context.grounded:
            assert context.refusal is not None
            task.refuse(context.refusal)
            return None

        template = load_prompt_for(PromptOperation.ASSESSMENT_QUIZ)
        task.prompts.append(
            PromptUseRecord(
                operation=PromptOperation.ASSESSMENT_QUIZ.value,
                prompt_id=template.name.value,
                version=template.version,
            )
        )
        prompt = template.render(
            topic_title=topic_title,
            question_count=question_count,
            evidence=context.as_prompt_block(),
        )

        try:
            draft = generate_structured(
                self.runtime.llm,
                prompt,
                AssessmentDraftLLM,
                repair_attempts=self.runtime.repair_attempts,
                on_call=lambda: setattr(task, "llm_calls", task.llm_calls + 1),
            )
            questions = [
                AssessmentQuestion(
                    prompt=question.prompt,
                    options=question.options,
                    correct_option=question.correct_option,
                    source=question.source,
                    citations=resolve_citations(question.source_ids, context),
                )
                for question in draft.questions
            ]
            assessment = Assessment(
                topic_id=topic_id, title=topic_title, questions=questions
            )
        except (StructuredOutputError, UngroundedCitation, ValueError) as error:
            task.fail(f"assessment draft rejected: {error}")
            return None

        task.succeed()
        return assessment


__all__ = ["DEFAULT_EVIDENCE_LIMIT", "DEFAULT_QUESTION_COUNT", "AssessmentAgent"]
