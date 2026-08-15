"""Strict, schema-validated absence triage. Human administrators decide."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.prompts import PromptOperation, load_prompt_for
from agents.schemas import generate_structured

POLICY_CLAUSES = {
    "P01_DOCUMENTED_EMERGENCY": "A specific emergency outside the learner's control may qualify after human review.",
    "P02_SERIOUS_HEALTH": "A serious health event may qualify; do not diagnose and route supporting documents to a human.",
    "P03_BEREAVEMENT": "Bereavement involving a close relation may qualify after sensitive human review.",
    "P04_OFFICIAL_DUTY": "A compulsory official, legal, military, or university duty may qualify with human-verifiable support.",
    "P05_TECHNICAL_OUTAGE": "A platform or regional outage may qualify when specific and independently reviewable.",
    "P06_ORDINARY_CONFLICT": "Workload, preference, oversleeping, forgetting, travel planning, and ordinary conflicts do not qualify.",
    "P07_INSUFFICIENT_DETAIL": "Vague or internally incomplete claims require one bounded clarification or human review.",
    "P08_ACCESS_ONLY": "Replay access may be recommended when learning continuity is fair but grade relief is unsupported.",
}

QUESTION_CODES = {
    "CAUSE_AND_TIMING": "State the specific event and when it prevented attendance.",
    "DIRECT_IMPACT": "Explain how the event directly prevented this lecture or quiz.",
    "OFFICIAL_DOCUMENT": "Attach an image of the relevant official or legal document for a human administrator.",
    "MEDICAL_DOCUMENT": "Attach an image of available medical documentation for a human administrator; hide unrelated details if possible.",
    "OUTAGE_DETAILS": "State the provider, location, and exact outage period so an administrator can review it.",
}

Recommendation = Literal[
    "recommend_excused", "recommend_access_only", "recommend_unexcused", "human_review"
]
NextAction = Literal["pending_admin", "ask_clarification", "request_evidence"]
QuestionCode = Literal[
    "CAUSE_AND_TIMING", "DIRECT_IMPACT", "OFFICIAL_DOCUMENT", "MEDICAL_DOCUMENT", "OUTAGE_DETAILS"
]
PolicyCode = Literal[
    "P01_DOCUMENTED_EMERGENCY", "P02_SERIOUS_HEALTH", "P03_BEREAVEMENT",
    "P04_OFFICIAL_DUTY", "P05_TECHNICAL_OUTAGE", "P06_ORDINARY_CONFLICT",
    "P07_INSUFFICIENT_DETAIL", "P08_ACCESS_ONLY",
]
Sensitivity = Literal["legal", "medical", "personal_safety", "bereavement"]


class AbsenceTriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: Recommendation
    next_action: NextAction
    question_code: QuestionCode | None = None
    policy_clause_ids: list[PolicyCode] = Field(min_length=1, max_length=4)
    sensitivity_flags: list[Sensitivity] = Field(default_factory=list, max_length=4)
    admin_summary: str = Field(min_length=10, max_length=500)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def question_matches_action(self) -> "AbsenceTriageResult":
        if self.next_action == "pending_admin" and self.question_code is not None:
            raise ValueError("pending_admin cannot include a question_code")
        if self.next_action != "pending_admin" and self.question_code is None:
            raise ValueError("a learner action requires a question_code")
        if self.next_action == "request_evidence" and self.question_code not in {
            "OFFICIAL_DOCUMENT", "MEDICAL_DOCUMENT"
        }:
            raise ValueError("request_evidence requires an evidence question code")
        if self.next_action == "ask_clarification" and self.question_code in {
            "OFFICIAL_DOCUMENT", "MEDICAL_DOCUMENT"
        }:
            raise ValueError("evidence codes cannot be clarification questions")
        return self


def triage_absence(llm, *, case_facts: str, prior_answers: str) -> tuple[AbsenceTriageResult, str, str]:
    if not case_facts.strip() or len(case_facts) > 12_000:
        raise ValueError("case facts must contain 1 to 12000 characters")
    if len(prior_answers) > 8_000:
        raise ValueError("prior answers are too long")
    template = load_prompt_for(PromptOperation.ABSENCE_TRIAGE)
    prompt = template.render(
        policy_clauses="\n".join(f"{key}: {value}" for key, value in POLICY_CLAUSES.items()),
        allowed_question_codes="\n".join(f"{key}: {value}" for key, value in QUESTION_CODES.items()),
        case_facts=case_facts,
        prior_answers=prior_answers or "None",
    )
    result = generate_structured(llm, prompt, AbsenceTriageResult, repair_attempts=1)
    return result, template.name.value, template.version


__all__ = ["AbsenceTriageResult", "POLICY_CLAUSES", "QUESTION_CODES", "triage_absence"]
