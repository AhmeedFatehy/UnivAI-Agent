"""Generate a grounded post-lecture section pack, or refuse.

The generator retrieves evidence for the lecture, asks the model for activities,
worked examples and TODOs that cite passage ids, and then *resolves* those ids
into real citations. A section that cannot be attributed to the indexed
material is never published: it is refused instead.

The LLM call is a plain ``str -> str`` callable so tests inject a scripted
responder. A reply that stays malformed raises :class:`SectionGenerationError`
after a single bounded repair (the dedicated ``teaching/section_repair``
prompt); an empty ``examples``/``todos`` draft surfaces as an explicit
:class:`~tools.registry.Refusal`, never generic teaching content.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from agents.prompts import PromptOperation, load_prompt_for
from agents.schemas import (
    SectionDraftLLM,
    SectionOutcome,
    SectionPackV1,
    strict_json_document,
)
from planning.section_planner import (
    DEFAULT_SECTION_BUDGET,
    SectionBudget,
    SectionIdentity,
    SectionPlanError,
    build_section_pack,
    grounded_content_refusal,
)
from tools.registry import (
    GroundedContext,
    Refusal,
    RetrieveContextInput,
    ToolContext,
    call_tool,
)

DEFAULT_EVIDENCE_LIMIT = 8
DEFAULT_FOCUS = "the material covered in the lecture"
# One retrieval per slide would mean ~50 round trips for a single section, most
# of them returning the same passages. A sample spread across the deck covers
# the lecture's span at a fraction of the cost.
MAX_SECTION_SEEDS = 10


class SectionGenerationError(RuntimeError):
    """The model could not produce a valid section draft within the repair budget."""

    def __init__(self, last_error: str):
        super().__init__(f"section draft invalid: {last_error}")
        self.last_error = last_error


class SectionRun(BaseModel):
    """One section request's result plus the observability the caller needs."""

    outcome: SectionOutcome
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    llm_calls: int = Field(ge=0)
    prompts: list[str] = Field(default_factory=list)

    @property
    def section(self) -> SectionPackV1 | None:
        return self.outcome.section

    @property
    def refusal(self) -> Refusal | None:
        return self.outcome.refusal


def section_seed_queries(identity: SectionIdentity, topics: list[str] | None) -> list[str]:
    """The subjects to retrieve on: the lecture's own slide headings.

    Retrieval is scored by ``term_coverage`` — the share of the QUERY's terms
    found in one passage. A single query of ``"<title>. the material covered in
    the lecture"`` therefore demanded that a textbook passage contain the words
    "material", "covered" and "lecture", which no passage does. For a one-word
    title such as "Triggers" the best attainable coverage was 1/4 = 0.25 against
    a 0.34 threshold, so those sections could never ground no matter how good
    the book was.

    Slide headings are what the lecture actually taught, and each is short and
    purely topical, so coverage measures topic overlap and nothing else.
    """
    headings: list[str] = []
    seen: set[str] = set()
    for candidate in topics or []:
        text = " ".join(str(candidate or "").split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            headings.append(text)

    # Sample evenly rather than taking the first N: a lecture's opening slides
    # are introductions, and the practical material sits later in the deck.
    if len(headings) > MAX_SECTION_SEEDS:
        stride = len(headings) / MAX_SECTION_SEEDS
        headings = [headings[int(index * stride)] for index in range(MAX_SECTION_SEEDS)]

    title = " ".join(str(identity.lecture_title or "").split())
    if title and title.casefold() not in {heading.casefold() for heading in headings}:
        headings.append(title)
    return headings


def retrieve_section_evidence(
    tool_context: ToolContext,
    identity: SectionIdentity,
    *,
    focus: str = DEFAULT_FOCUS,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    topics: list[str] | None = None,
) -> GroundedContext:
    """Pull passages on which to ground the section, honouring tenant isolation.

    Each slide heading is retrieved separately and the results are merged, so
    one weak heading cannot starve the whole section. Passage ids are renumbered
    across the merged set so the model sees one continuous evidence block.
    """
    seeds = section_seed_queries(identity, topics)
    merged: list = []
    seen: set[tuple[str, int | None]] = set()
    last_refusal: Refusal | None = None

    for seed in seeds:
        context = call_tool(
            "retrieve_context",
            RetrieveContextInput(
                query=seed,
                user_id=identity.user_id,
                collection_id=identity.collection_id,
                limit=limit,
            ),
            tool_context,
        )
        assert isinstance(context, GroundedContext)
        if not context.grounded:
            last_refusal = context.refusal
            continue
        for passage in context.passages:
            key = (passage.citation.document_id, passage.citation.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            merged.append(passage)

    query = "; ".join(seeds)
    if not merged:
        return GroundedContext(
            query=query,
            grounded=False,
            refusal=last_refusal
            or Refusal(
                reason="No indexed material matched this lecture's slides.",
                query=query,
                scope={"collection_id": identity.collection_id},
            ),
        )

    merged.sort(key=lambda passage: -passage.score)
    renumbered = [
        passage.model_copy(update={"passage_id": f"S{index}", "rank": index})
        for index, passage in enumerate(merged[: limit * 2], start=1)
    ]
    return GroundedContext(query=query, grounded=True, passages=renumbered)


def _generate_section_draft(
    llm: Callable[[str], str],
    prompt: str,
    repair_template,
    *,
    repair_attempts: int,
    on_call: Callable[[], None] | None = None,
    prompts: list[str],
) -> SectionDraftLLM:
    """Call the model and validate, with a bounded repair using the section prompt."""
    current_prompt = prompt
    last_error = ""
    last_raw = ""
    for attempt in range(repair_attempts + 1):
        if on_call is not None:
            on_call()
        last_raw = llm(current_prompt) or ""
        prompts.append(current_prompt)
        try:
            payload = json.loads(strict_json_document(last_raw))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            last_error = f"response is not valid JSON: {error}"
        else:
            try:
                return SectionDraftLLM.model_validate(payload)
            except ValidationError as error:
                last_error = _compact_validation_error(error)
        if attempt >= repair_attempts:
            break
        current_prompt = repair_template.render(
            original_prompt=prompt,
            previous_reply=last_raw[:2000],
            validation_errors=last_error,
            revision_guidance="Resubmit the whole JSON object with the reported errors fixed.",
        )
    raise SectionGenerationError(last_error)


def generate_section_pack(
    *,
    llm: Callable[[str], str],
    identity: SectionIdentity,
    tool_context: ToolContext,
    focus: str = DEFAULT_FOCUS,
    budget: SectionBudget = DEFAULT_SECTION_BUDGET,
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    repair_attempts: int = 1,
    on_call: Callable[[], None] | None = None,
    topics: list[str] | None = None,
) -> SectionRun:
    """Generate a validated section pack, or a grounded refusal.

    ``topics`` are the lecture's slide headings; they decide what the section
    is grounded on, so a section teaches practice for what the lecture taught.
    """
    template = load_prompt_for(PromptOperation.CONTENT_GENERATE_SECTION)
    repair_template = load_prompt_for(PromptOperation.CONTENT_REPAIR_SECTION)
    prompts: list[str] = []

    context = retrieve_section_evidence(
        tool_context, identity, focus=focus, limit=evidence_limit, topics=topics
    )
    if not context.grounded:
        return SectionRun(
            outcome=SectionOutcome(refusal=context.refusal),
            prompt_id=template.name.value,
            prompt_version=template.version,
            llm_calls=0,
            prompts=prompts,
        )

    prompt = template.render(
        section_title=_section_title(identity, focus),
        lecture_title=identity.lecture_title,
        plan_schema=identity.plan_schema,
        plan_version=identity.plan_version,
        schedule=_schedule_text(identity),
        section_budget_minutes=budget.label,
        focus=focus,
        evidence=context.as_prompt_block(),
    )

    calls = [0]

    def _count_call() -> None:
        calls[0] += 1
        if on_call is not None:
            on_call()

    draft = _generate_section_draft(
        llm,
        prompt,
        repair_template,
        repair_attempts=repair_attempts,
        on_call=_count_call,
        prompts=prompts,
    )

    refusal = grounded_content_refusal(draft, reason=focus)
    # A grounded retrieval can still receive an over-cautious first draft with
    # empty practice lists. Spend the same single repair budget on asking the
    # model to re-check the supplied passages before treating that as a durable
    # refusal. This never relaxes citation validation or invents fallback data.
    if refusal is not None and repair_attempts > 0 and calls[0] == 1:
        repair_prompt = repair_template.render(
            original_prompt=prompt,
            previous_reply=draft.model_dump_json()[:2000],
            validation_errors=refusal.reason,
            revision_guidance=(
                "The retrieval is grounded. Re-check whether the evidence supports "
                "an applied reasoning scenario and a concrete learner action. If it "
                "does, include at least one cited worked example and one cited TODO; "
                "if it truly does not, keep the unsupported list empty."
            ),
        )
        try:
            repaired_draft = _generate_section_draft(
                llm,
                repair_prompt,
                repair_template,
                repair_attempts=0,
                on_call=_count_call,
                prompts=prompts,
            )
        except SectionGenerationError:
            # The first response was structurally valid and already gave us a
            # safe refusal. A malformed optional reconsideration must not turn
            # that safe outcome into a failed course-generation job.
            pass
        else:
            draft = repaired_draft
            refusal = grounded_content_refusal(draft, reason=focus)
    if refusal is not None:
        return SectionRun(
            outcome=SectionOutcome(refusal=refusal),
            prompt_id=template.name.value,
            prompt_version=template.version,
            llm_calls=calls[0],
            prompts=prompts,
        )

    try:
        section = build_section_pack(
            draft,
            identity,
            context,
            focus=focus,
            budget=budget,
        )
    except SectionPlanError as error:
        return SectionRun(
            outcome=SectionOutcome(refusal=_plan_refusal(identity, error)),
            prompt_id=template.name.value,
            prompt_version=template.version,
            llm_calls=calls[0],
            prompts=prompts,
        )

    return SectionRun(
        outcome=SectionOutcome(section=section),
        prompt_id=template.name.value,
        prompt_version=template.version,
        llm_calls=calls[0],
        prompts=prompts,
    )


def write_section_artifact(
    section: SectionPackV1,
    root: str | Path,
    *,
    tenant_id: str | None = None,
) -> Path:
    """Persist a validated pack into a tenant-namespaced, lecture-separate folder.

    Sections live under ``sections/<tenant_id>/`` — never inside a lecture's week
    folder — so a live delivery consumer cannot confuse the two session types.
    """
    root = Path(root)
    namespace = tenant_id or section.user_id
    target_dir = root / "sections" / namespace
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact = target_dir / f"section-{safe_topic(section.topic_id)}.json"
    artifact.write_text(
        section.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return artifact


def publish_section_contract(
    section: SectionPackV1, out_dir: str | Path
) -> Path:
    """Emit the versioned fixture UnivAI-live can consume without Agent imports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = section.model_dump(mode="json")
    path = (
        out_dir
        / f"section-{safe_topic(section.topic_id)}.v{section.schema_version}.json"
    )
    path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _schedule_text(identity: SectionIdentity) -> str:
    if identity.week_number is not None and identity.course_id:
        return f"course {identity.course_id}, week {identity.week_number}"
    if identity.week_number is not None:
        return f"week {identity.week_number}"
    return "immediately after the lecture"


def _section_title(identity: SectionIdentity, focus: str = DEFAULT_FOCUS) -> str:
    return f"{identity.lecture_title} — Section Practice"


def _plan_refusal(identity: SectionIdentity, error: SectionPlanError) -> Refusal:
    return Refusal(
        reason=(
            "The section could not be grounded: "
            f"{error}. This lecture's evidence cannot support a valid pack."
        ),
        query=identity.lecture_title,
    )


def _compact_validation_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors()[:6]
    )


def safe_topic(topic_id: str) -> str:
    return (
        "".join(ch for ch in topic_id if ch.isalnum() or ch in "-_") or "section"
    )


__all__ = [
    "DEFAULT_EVIDENCE_LIMIT",
    "SectionGenerationError",
    "SectionRun",
    "generate_section_pack",
    "publish_section_contract",
    "write_section_artifact",
]
