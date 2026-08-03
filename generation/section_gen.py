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
    extract_json,
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


def retrieve_section_evidence(
    tool_context: ToolContext,
    identity: SectionIdentity,
    *,
    focus: str = DEFAULT_FOCUS,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
) -> GroundedContext:
    """Pull passages on which to ground the section, honouring tenant isolation."""
    query = f"{identity.lecture_title}. {focus}"
    context = call_tool(
        "retrieve_context",
        RetrieveContextInput(
            query=query,
            user_id=identity.user_id,
            collection_id=identity.collection_id,
            limit=limit,
        ),
        tool_context,
    )
    assert isinstance(context, GroundedContext)
    return context


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
            payload = json.loads(extract_json(last_raw))
        except (json.JSONDecodeError, TypeError) as error:
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
) -> SectionRun:
    """Generate a validated section pack, or a grounded refusal."""
    template = load_prompt_for(PromptOperation.CONTENT_GENERATE_SECTION)
    repair_template = load_prompt_for(PromptOperation.CONTENT_REPAIR_SECTION)
    prompts: list[str] = []

    context = retrieve_section_evidence(
        tool_context, identity, focus=focus, limit=evidence_limit
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