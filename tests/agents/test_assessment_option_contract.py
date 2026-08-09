from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.schemas import AssessmentDraftLLM


def question(option_count: int, correct: str = "A") -> dict:
    return {
        "prompt": "Which grounded answer is correct?",
        "options": [f"{letter}) option {letter}" for letter in "ABCDEF"[:option_count]],
        "correct_option": correct,
        "source": "lecture",
        "source_ids": ["S1"],
    }


@pytest.mark.parametrize("assessment_type", ["midterm", "final"])
def test_midterm_and_final_require_six_options(assessment_type: str):
    draft = AssessmentDraftLLM.model_validate(
        {"assessment_type": assessment_type, "questions": [question(6, "F")]}
    )

    assert len(draft.questions[0].options) == 6
    assert draft.questions[0].correct_option == "F"

    with pytest.raises(ValidationError, match="exactly 6"):
        AssessmentDraftLLM.model_validate(
            {"assessment_type": assessment_type, "questions": [question(4)]}
        )


def test_quiz_requires_four_options_and_an_a_to_d_answer():
    draft = AssessmentDraftLLM.model_validate(
        {"assessment_type": "quiz", "questions": [question(4, "D")]}
    )
    assert len(draft.questions[0].options) == 4

    with pytest.raises(ValidationError, match="exactly 4"):
        AssessmentDraftLLM.model_validate(
            {"assessment_type": "quiz", "questions": [question(6)]}
        )

    with pytest.raises(ValidationError, match="correct_option"):
        AssessmentDraftLLM.model_validate(
            {"assessment_type": "quiz", "questions": [question(4, "F")]}
        )


def test_duplicate_options_are_rejected_before_publication():
    malformed = question(4)
    malformed["options"][3] = malformed["options"][0]

    with pytest.raises(ValidationError, match="unique"):
        AssessmentDraftLLM.model_validate(
            {"assessment_type": "quiz", "questions": [malformed]}
        )
