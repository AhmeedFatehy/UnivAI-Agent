"""Prompt-injection guardrails for user queries and retrieved source text.

The guardrail is a deterministic classifier, not an LLM. It never invents a
verdict and it never broadens into a content filter: the same decision is
reproduced for the same input on every run, and normal academic text passes
through unflagged.
"""

from guardrails.input import (
    GuardrailDecision,
    GuardrailKind,
    classify_source_text,
    classify_user_input,
    screen_passages,
    screen_query,
)

__all__ = [
    "GuardrailDecision",
    "GuardrailKind",
    "classify_source_text",
    "classify_user_input",
    "screen_passages",
    "screen_query",
]
