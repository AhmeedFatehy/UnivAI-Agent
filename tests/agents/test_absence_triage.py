"""Strict absence triage remains advisory and schema bounded."""

import json

import pytest
from pydantic import ValidationError

from agents.absence import AbsenceTriageResult, triage_absence


def test_absence_triage_returns_prompt_provenance_and_bounded_evidence_code():
    def llm(_prompt: str) -> str:
        return json.dumps(
            {
                "recommendation": "human_review",
                "next_action": "request_evidence",
                "question_code": "OFFICIAL_DOCUMENT",
                "policy_clause_ids": ["P04_OFFICIAL_DUTY"],
                "sensitivity_flags": ["legal"],
                "admin_summary": "A human must review the claimed official duty and any supplied document.",
                "confidence": 0.7,
            }
        )

    result, prompt_id, version = triage_absence(
        llm,
        case_facts='{"learner_statement":"I was summoned to court."}',
        prior_answers="None",
    )

    assert result.question_code == "OFFICIAL_DOCUMENT"
    assert result.recommendation == "human_review"
    assert prompt_id == "absence/triage"
    assert version.count(".") == 2


def test_absence_schema_rejects_ai_claiming_a_final_decision():
    with pytest.raises(ValidationError):
        AbsenceTriageResult.model_validate(
            {
                "recommendation": "approved",
                "next_action": "pending_admin",
                "question_code": None,
                "policy_clause_ids": ["P01_DOCUMENTED_EMERGENCY"],
                "sensitivity_flags": [],
                "admin_summary": "The model attempted to make the final decision.",
                "confidence": 1,
            }
        )
