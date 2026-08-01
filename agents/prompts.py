"""Typed, versioned system-prompt catalog and operation router."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptId(str, Enum):
    PROGRAMME_PLANNER = "curriculum/programme_planner"
    BOOK_STRUCTURE_ANALYSIS = "curriculum/book_structure_analysis"
    SEMESTER_GENERATION = "curriculum/semester_generation"
    SEMESTER_PLAN_REPAIR = "curriculum/semester_plan_repair"
    LECTURE_GENERATION = "teaching/lecture_generation"
    LECTURE_SUMMARY = "teaching/lecture_summary"
    LEARNING_OBJECTIVES = "teaching/learning_objectives"
    DIAGNOSTIC = "assessment/diagnostic"
    PRACTICE = "assessment/practice"
    QUIZ = "assessment/quiz"
    ASSIGNMENT = "assessment/assignment"
    MIDTERM = "assessment/midterm"
    FINAL = "assessment/final"
    ORAL_EXAM = "assessment/oral_exam"
    QUESTION_REPAIR = "assessment/question_repair"
    ANSWER_EXPLANATION = "assessment/answer_explanation"
    QUERY_DECOMPOSITION = "retrieval/query_decomposition"
    QUERY_EXPANSION = "retrieval/query_expansion"
    GROUNDED_ANSWER = "retrieval/grounded_answer"
    GROUNDED_REFUSAL = "retrieval/grounded_refusal"
    RETRIEVAL_RELEVANCE = "evaluation/retrieval_relevance"
    ANSWER_GROUNDEDNESS = "evaluation/answer_groundedness"
    ASSESSMENT_QUALITY = "evaluation/assessment_quality"
    STRUCTURED_OUTPUT_REPAIR = "shared/structured_output_repair"


class PromptOperation(str, Enum):
    CURRICULUM_EXTRACT_TOPICS = "curriculum.extract_topics"
    CURRICULUM_ANALYZE_BOOK = "curriculum.analyze_book_structure"
    CURRICULUM_GENERATE_SEMESTER = "curriculum.generate_semester"
    CURRICULUM_REPAIR_SEMESTER = "curriculum.repair_semester"
    CONTENT_GENERATE_LECTURE = "content.generate_lecture"
    CONTENT_SUMMARIZE_LECTURE = "content.summarize_lecture"
    CONTENT_GENERATE_OBJECTIVES = "content.generate_learning_objectives"
    ASSESSMENT_DIAGNOSTIC = "assessment.generate:diagnostic"
    ASSESSMENT_PRACTICE = "assessment.generate:practice"
    ASSESSMENT_QUIZ = "assessment.generate:quiz"
    ASSESSMENT_ASSIGNMENT = "assessment.generate:assignment"
    ASSESSMENT_MIDTERM = "assessment.generate:midterm"
    ASSESSMENT_FINAL = "assessment.generate:final"
    ASSESSMENT_ORAL = "assessment.generate:oral_exam"
    ASSESSMENT_REPAIR = "assessment.repair_question"
    ASSESSMENT_EXPLAIN = "assessment.explain_answer"
    RETRIEVAL_DECOMPOSE = "retrieval.decompose_query"
    RETRIEVAL_EXPAND = "retrieval.expand_query"
    RETRIEVAL_ANSWER = "retrieval.answer"
    RETRIEVAL_REFUSE = "retrieval.refuse"
    EVALUATION_RETRIEVAL = "evaluation.retrieval_relevance"
    EVALUATION_GROUNDEDNESS = "evaluation.answer_groundedness"
    EVALUATION_ASSESSMENT = "evaluation.assessment_quality"
    SHARED_REPAIR_JSON = "shared.repair_structured_output"


class PromptTemplate(BaseModel):
    """One authored prompt plus the metadata required to route it safely."""

    name: PromptId
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    operations: list[PromptOperation] = Field(min_length=1)
    variables: list[str] = Field(default_factory=list)
    output_schema: str = Field(min_length=1)
    grounding_policy: str = Field(min_length=1)
    capabilities: list[str] = Field(default_factory=list)
    generation_defaults: dict[str, Any] = Field(default_factory=dict)
    safety: list[str] = Field(default_factory=list)
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)

    def render_user(self, **values: Any) -> str:
        missing = [name for name in self.variables if name not in values]
        if missing:
            raise KeyError(
                f"prompt '{self.name.value}' v{self.version} needs {missing}, "
                "which were not supplied"
            )

        body = self.user
        for name in self.variables:
            body = body.replace("{" + name + "}", str(values[name]))

        leftover = [
            name for name in self.variables if "{" + name + "}" in body
        ]
        if leftover:
            raise ValueError(
                f"prompt '{self.name.value}' still contains {leftover} after render"
            )
        return body.strip()

    def render(self, **values: Any) -> str:
        return f"{self.system.strip()}\n\n{self.render_user(**values)}"


class PromptRegistry(BaseModel):
    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    routes: dict[PromptOperation, PromptId]
    aliases: dict[str, PromptId] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _aliases_do_not_shadow_operations(self) -> "PromptRegistry":
        overlap = set(self.aliases) & {operation.value for operation in self.routes}
        if overlap:
            raise ValueError(f"prompt aliases shadow operations: {sorted(overlap)}")
        return self


def _directory(prompts_dir: str | Path | None) -> Path:
    return Path(prompts_dir) if prompts_dir else PROMPTS_DIR


@lru_cache(maxsize=8)
def load_prompt_registry(prompts_dir: str | Path | None = None) -> PromptRegistry:
    path = _directory(prompts_dir) / "registry.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"prompt registry not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"prompt registry {path} must be a YAML mapping")
    return PromptRegistry.model_validate(payload)


@lru_cache(maxsize=64)
def _load_prompt_id(
    prompt_id: PromptId, prompts_dir: str | Path | None = None
) -> PromptTemplate:
    path = _directory(prompts_dir) / f"{prompt_id.value}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"prompt template not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"prompt template {path} must be a YAML mapping")
    payload.setdefault("name", prompt_id.value)
    prompt = PromptTemplate.model_validate(payload)
    if prompt.name is not prompt_id:
        raise ValueError(
            f"prompt file {path} declares '{prompt.name.value}', expected '{prompt_id.value}'"
        )
    return prompt


def load_prompt(
    name: str | PromptId, prompts_dir: str | Path | None = None
) -> PromptTemplate:
    """Load by stable ID, with registry aliases for older callers."""
    registry = load_prompt_registry(prompts_dir)
    if isinstance(name, PromptId):
        prompt_id = name
    else:
        prompt_id = registry.aliases.get(name)
        if prompt_id is None:
            prompt_id = PromptId(name)
    return _load_prompt_id(prompt_id, prompts_dir)


def load_prompt_for(
    operation: str | PromptOperation, prompts_dir: str | Path | None = None
) -> PromptTemplate:
    registry = load_prompt_registry(prompts_dir)
    typed_operation = (
        operation if isinstance(operation, PromptOperation) else PromptOperation(operation)
    )
    try:
        prompt_id = registry.routes[typed_operation]
    except KeyError as error:
        raise KeyError(f"no system prompt is routed for '{typed_operation.value}'") from error
    prompt = _load_prompt_id(prompt_id, prompts_dir)
    if typed_operation not in prompt.operations:
        raise ValueError(
            f"registry routes '{typed_operation.value}' to '{prompt_id.value}', "
            "but the prompt does not declare that operation"
        )
    return prompt


def validate_prompt_catalog(
    prompts_dir: str | Path | None = None,
) -> dict[PromptId, PromptTemplate]:
    """Load every routed prompt and reject missing or duplicate declarations."""
    registry = load_prompt_registry(prompts_dir)
    directory = _directory(prompts_dir)
    missing_operations = set(PromptOperation) - set(registry.routes)
    if missing_operations:
        raise ValueError(
            "prompt operations without routes: "
            f"{sorted(operation.value for operation in missing_operations)}"
        )
    unrouted_prompts = set(PromptId) - set(registry.routes.values())
    if unrouted_prompts:
        raise ValueError(
            "prompt IDs without routes: "
            f"{sorted(prompt_id.value for prompt_id in unrouted_prompts)}"
        )
    files = {
        path.relative_to(directory).with_suffix("").as_posix()
        for path in directory.rglob("*.yaml")
        if path.name != "registry.yaml"
    }
    expected_files = {prompt_id.value for prompt_id in PromptId}
    if files != expected_files:
        raise ValueError(
            "prompt files do not match declared prompt IDs; "
            f"missing={sorted(expected_files - files)}, extra={sorted(files - expected_files)}"
        )
    loaded: dict[PromptId, PromptTemplate] = {}
    operation_owners: dict[PromptOperation, PromptId] = {}
    for operation, prompt_id in registry.routes.items():
        prompt = _load_prompt_id(prompt_id, prompts_dir)
        loaded[prompt_id] = prompt
        for declared in prompt.operations:
            previous = operation_owners.setdefault(declared, prompt_id)
            if previous is not prompt_id:
                raise ValueError(
                    f"operation '{declared.value}' is declared by both "
                    f"'{previous.value}' and '{prompt_id.value}'"
                )
        if operation not in prompt.operations:
            raise ValueError(
                f"prompt '{prompt_id.value}' does not declare routed operation "
                f"'{operation.value}'"
            )
    return loaded


__all__ = [
    "PROMPTS_DIR",
    "PromptId",
    "PromptOperation",
    "PromptRegistry",
    "PromptTemplate",
    "load_prompt",
    "load_prompt_for",
    "load_prompt_registry",
    "validate_prompt_catalog",
]
