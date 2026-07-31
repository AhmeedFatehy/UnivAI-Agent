"""Content agent — draft lecture narration for a planned topic.

The agent retrieves evidence for the topic, asks the model for narration that
cites passage ids, and then *resolves* those ids into real book/page/section
citations. A lecture that cannot be attributed to indexed material is never
produced: the task is refused instead.
"""

from __future__ import annotations

from agents.schemas import (
    AgentName,
    Handoff,
    Lecture,
    LectureDraftLLM,
    LectureSegment,
    StructuredOutputError,
    TaskRecord,
    ToolCallRecord,
    UngroundedCitation,
    generate_structured,
    load_prompt,
    resolve_citations,
)
from tools.registry import GroundedContext, RetrieveContextInput, call_tool

DEFAULT_SLIDE_COUNT = 4
DEFAULT_EVIDENCE_LIMIT = 6


class ContentAgent:
    """Turn one planned topic into cited narration, or refuse."""

    name = AgentName.CONTENT

    def __init__(self, runtime):
        self.runtime = runtime

    def run(self, handoff: Handoff, task: TaskRecord) -> Lecture | None:
        task.start()

        topic_id = handoff.payload["topic_id"]
        topic_title = handoff.payload["topic_title"]
        topic_summary = handoff.payload.get("topic_summary", topic_title)
        slide_count = int(handoff.constraints.get("slide_count", DEFAULT_SLIDE_COUNT))
        evidence_limit = int(
            handoff.constraints.get("evidence_limit", DEFAULT_EVIDENCE_LIMIT)
        )

        context = call_tool(
            "retrieve_context",
            RetrieveContextInput(
                query=f"{topic_title}. {topic_summary}",
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

        prompt = load_prompt("lecture").render(
            topic_title=topic_title,
            topic_summary=topic_summary,
            slide_count=slide_count,
            evidence=context.as_prompt_block(),
        )

        try:
            draft = generate_structured(
                self.runtime.llm,
                prompt,
                LectureDraftLLM,
                repair_attempts=self.runtime.repair_attempts,
                on_call=lambda: setattr(task, "llm_calls", task.llm_calls + 1),
            )
            segments = [
                LectureSegment(
                    slide=segment.slide,
                    heading=segment.heading,
                    text=segment.text,
                    citations=resolve_citations(segment.source_ids, context),
                )
                for segment in sorted(draft.segments, key=lambda item: item.slide)
            ]
            lecture = Lecture(topic_id=topic_id, title=draft.title, segments=segments)
        except (StructuredOutputError, UngroundedCitation, ValueError) as error:
            task.fail(f"lecture draft rejected: {error}")
            return None

        task.succeed()
        return lecture


__all__ = ["DEFAULT_EVIDENCE_LIMIT", "DEFAULT_SLIDE_COUNT", "ContentAgent"]
