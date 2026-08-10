"""Layered prompt-injection guardrails and prompt/data boundaries.

The deterministic classifier is an entry-point signal, not the security
boundary. Dynamic values are also structurally isolated as untrusted data,
provider adapters use real system/user roles, tools are least-privilege, and
model outputs cross strict schemas.
"""

from guardrails.input import (
    GuardrailDecision,
    GuardrailKind,
    classify_source_text,
    classify_user_input,
    screen_passages,
    screen_query,
)
from guardrails.prompt_boundary import (
    PROMPT_BOUNDARY_POLICY_VERSION,
    PromptBoundaryError,
    quote_untrusted_data,
)

__all__ = [
    "GuardrailDecision",
    "GuardrailKind",
    "PROMPT_BOUNDARY_POLICY_VERSION",
    "PromptBoundaryError",
    "classify_source_text",
    "classify_user_input",
    "screen_passages",
    "screen_query",
    "quote_untrusted_data",
]
