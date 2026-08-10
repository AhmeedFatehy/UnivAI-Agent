"""Deterministic prompt-injection controls for user queries and source text.

Two distinct surfaces are screened:

* **Direct injection** — a user query that tries to rewrite the model's role,
  drop its instructions, or exfiltrate the system prompt. This is screened with
  :func:`classify_user_input`.
* **Indirect injection** — retrieved/source text that carries embedded
  instructions (role tags, ``ignore the previous …``, exfiltration prompts).
  Source text is *quoted data and never instruction authority*: it is screened
  with :func:`classify_source_text` so a caller can observe the risk, but the
  text itself is still returned as data for the deterministic grounding gate to
  adjudicate.

The classifier is deliberately **rule-based and deterministic**: no model call,
no probability, no tuning threshold that silently turns into a content filter.
Each rule is a compound pattern (a verb plus a target), so ordinary academic
prose containing words like *system*, *instructions* or *ignore* does not trip
it.

The one place the classifier is conservative is on **role tags**: an actual
``system:``/``user:``/``<|system|>`` marker in retrieved text is a strong signal
of an injection attempt and is always reported. Everything else is a compound
match.

Severity is a hint for operators, not a gating value; ``safe`` is the decision a
caller must act on. A ``safe=False`` user query should be refused or answered
from grounded sources only; ``safe=False`` source text must still be treated as
data, never as instructions.
"""

from __future__ import annotations

import re
import unicodedata
from html import unescape
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

GUARDRAIL_SCHEMA = "univai.agent.guardrail"
GUARDRAIL_SCHEMA_VERSION = "1.1.0"

#: Canonical names of the compound rules. Exposed so operators and tests can
#: reference a specific reason instead of pattern-matching prose.
RULE_IGNORE_PREVIOUS = "ignore_previous_instructions"
RULE_ROLE_REASSIGNMENT = "role_reassignment"
RULE_NEW_INSTRUCTIONS = "new_instructions"
RULE_EXFILTRATE_PROMPT = "prompt_exfiltration"
RULE_UNRESTRICTED_MODE = "unrestricted_mode"
RULE_ROLE_TAG = "role_tag_marker"
RULE_EMBEDDED_INSTRUCTION = "embedded_instruction"
RULE_DELIMITER_ATTACK = "delimiter_attack"
RULE_OVERSIZED_INPUT = "oversized_input"

MAX_USER_QUERY_CHARS = 4_000

_INVISIBLE_FORMATTING = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")

_RULE_DIRECT: list[tuple[str, re.Pattern]] = [
    (
        RULE_IGNORE_PREVIOUS,
        re.compile(
            r"\b(?:ignore|disregard|forget|discard|skip|overlook)\s+"
            r"(?:all\s+|any\s+|every\s+|the\s+|above\s+|previous\s+|prior\s+|earlier\s+)*"
            r"(?:instructions|prompt|context|rules?|messages|system)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_IGNORE_PREVIOUS,
        re.compile(
            r"\bignore\s+everything\s+(?:above|before|previously|written)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_ROLE_REASSIGNMENT,
        re.compile(
            r"\b(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|that\s+you\s+are)|"
            r"from\s+now\s+on\s+you\s+(?:are|will)|your\s+new\s+(?:role|identity))\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_NEW_INSTRUCTIONS,
        re.compile(
            r"\b(?:new|updated|revised|replacement)\s+instructions\b|"
            r"\b(?:override|replace)\s+(?:the|your)\s+(?:system|original|previous)\s+"
            r"(?:prompt|instructions|rules)\b|"
            r"\bdisregard\s+your\s+(?:system|programming|original)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_EXFILTRATE_PROMPT,
        re.compile(
            r"\b(?:repeat|print|show|reveal|output|display|echo|copy|dump)\s+"
            r"(?:me\s+)?"
            r"(?:your|the|your\s+(?:full|original|initial|base)|"
            r"the\s+(?:full|complete|original|initial|base))\s+"
            r"(?:system\s+prompt|initial\s+prompt|original\s+prompt|original\s+instructions|"
            r"first\s+message|underlying\s+instructions|base\s+prompt|prompt\s+below)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_UNRESTRICTED_MODE,
        re.compile(
            r"\b(?:developer\s+mode|jailbreak|unrestricted\s+mode|no\s+restrictions|"
            r"ignore\s+(?:all\s+)?(?:your\s+)?(?:safety|rules|guidelines)|"
            r"do\s+anything\s+now|there\s+are\s+no\s+rules|anti\s+restriction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_ROLE_TAG,
        re.compile(
            r"<\|(?:system|user|assistant)\|>|"
            r"^###\s*system\b|^##\s*system\b|"
            r"^\s*(?:system|user|assistant):\s*(?:\n|$)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
]

_RULE_INDIRECT: list[tuple[str, re.Pattern]] = [
    (
        RULE_ROLE_TAG,
        re.compile(
            r"<\|(?:system|user|assistant)\|>|"
            r"###\s*(?:system|user|assistant)\b|"
            r"##\s*(?:system|user|assistant)\b|"
            r"(?:^|\n)\s*(?:system|user|assistant):\s*(?:\n|$)|"
            r"\b(?:system|user|assistant)\s*:\s*"
            r"(?:you\s+(?:are|must|will|should)|ignore|disregard|forget|pretend|"
            r"act\s+as|from\s+now\s+on)|"
            r"\*\*(?:system|user|assistant)\s*:?\s*\*\*\s*:?\s+"
            r"(?:you\s+(?:are|must|will|should)|ignore|disregard|forget|pretend|"
            r"act\s+as|from\s+now\s+on)",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_IGNORE_PREVIOUS,
        re.compile(
            r"\b(?:ignore|disregard|forget|forget\s+all)\s+"
            r"(?:all\s+|any\s+|the\s+|above\s+|previous\s+|prior\s+|earlier\s+)*"
            r"(?:instructions|prompt|context|rules)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_ROLE_REASSIGNMENT,
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on|your\s+new\s+(?:task|role|mission)|"
            r"you\s+must\s+now|instead\s+you\s+should)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_EMBEDDED_INSTRUCTION,
        re.compile(
            r"(?:^|\n)\s*(?:important|instruction|note|remember|warning|rule)\s*:\s*"
            r"(?:you\s+(?:must|are|should|will)|ignore|disregard|forget|pretend|act\s+as)",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_EXFILTRATE_PROMPT,
        re.compile(
            r"\b(?:repeat|print|show|reveal|output|display)\s+(?:your|the)\s+"
            r"(?:system\s+prompt|initial\s+prompt|original\s+instructions)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RULE_DELIMITER_ATTACK,
        re.compile(
            r"\b(?:end\s+of\s+(?:input|prompt|previous)|"
            r"everything\s+above\s+is|start\s+of\s+new\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
]


class GuardrailKind(str, Enum):
    USER_QUERY = "user_query"
    SOURCE_TEXT = "source_text"


class GuardrailDecision(BaseModel):
    """The deterministic verdict for one screened input."""

    schema_name: str = GUARDRAIL_SCHEMA
    schema_version: str = GUARDRAIL_SCHEMA_VERSION
    kind: GuardrailKind
    safe: bool
    matched_rules: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


def _evaluate(
    text: str, kind: GuardrailKind, rules: list[tuple[str, re.Pattern]]
) -> GuardrailDecision:
    original = text or ""
    if kind is GuardrailKind.USER_QUERY and len(original) > MAX_USER_QUERY_CHARS:
        return GuardrailDecision(
            kind=kind,
            safe=False,
            matched_rules=[RULE_OVERSIZED_INPUT],
            reasons=_describe([RULE_OVERSIZED_INPUT]),
        )

    # Normalise compatibility characters, HTML entities and invisible Unicode
    # formatting before applying compound rules. This catches visually disguised
    # role markers without broadening the rules into a topical content filter.
    text = unicodedata.normalize("NFKC", unescape(original))
    text = _INVISIBLE_FORMATTING.sub("", text)
    matched: list[str] = []
    for name, pattern in rules:
        if pattern.search(text or ""):
            matched.append(name)

    if not matched:
        return GuardrailDecision(kind=kind, safe=True, matched_rules=[], reasons=[])
    return GuardrailDecision(
        kind=kind,
        safe=False,
        matched_rules=matched,
        reasons=_describe(matched),
    )


def _describe(matched: list[str]) -> list[str]:
    descriptions = {
        RULE_IGNORE_PREVIOUS: "asks to ignore previous instructions or context",
        RULE_ROLE_REASSIGNMENT: "reassigns the assistant's role or task",
        RULE_NEW_INSTRUCTIONS: "requests new or replacement instructions",
        RULE_EXFILTRATE_PROMPT: "attempts to extract the system prompt",
        RULE_UNRESTRICTED_MODE: "asks for an unrestricted or jailbroken mode",
        RULE_ROLE_TAG: "contains an injected system/user/assistant role marker",
        RULE_EMBEDDED_INSTRUCTION: "embeds an instruction addressed to the model",
        RULE_DELIMITER_ATTACK: "tries to delimit the start of a new prompt",
        RULE_OVERSIZED_INPUT: "exceeds the maximum accepted user-query length",
    }
    return [descriptions[name] for name in matched]


def classify_user_input(text: str) -> GuardrailDecision:
    """Screen a **user query** for direct prompt injection.

    ``safe=False`` means the query attempts to override the assistant's role,
    drop its instructions, or exfiltrate the system prompt. The caller must
    refuse or answer from grounded sources only.
    """
    return _evaluate(text, GuardrailKind.USER_QUERY, _RULE_DIRECT)


def classify_source_text(text: str) -> GuardrailDecision:
    """Screen **retrieved/source text** for indirect injection markers.

    Source text is quoted data and never instruction authority. ``safe=False``
    flags embedded instruction material so an operator can observe the risk;
    the text is still data and must be routed through the normal grounding gate.
    """
    return _evaluate(text, GuardrailKind.SOURCE_TEXT, _RULE_INDIRECT)


def screen_query(text: str) -> GuardrailDecision:
    """Alias for :func:`classify_user_input` used at entry points."""
    return classify_user_input(text)


def screen_passages(passages: list[str]) -> list[GuardrailDecision]:
    """Screen every passage in a retrieved set, in order."""
    return [classify_source_text(passage) for passage in passages]


__all__ = [
    "GUARDRAIL_SCHEMA",
    "GUARDRAIL_SCHEMA_VERSION",
    "RULE_DELIMITER_ATTACK",
    "RULE_EMBEDDED_INSTRUCTION",
    "RULE_EXFILTRATE_PROMPT",
    "RULE_IGNORE_PREVIOUS",
    "RULE_NEW_INSTRUCTIONS",
    "RULE_OVERSIZED_INPUT",
    "RULE_ROLE_REASSIGNMENT",
    "RULE_ROLE_TAG",
    "RULE_UNRESTRICTED_MODE",
    "GuardrailDecision",
    "GuardrailKind",
    "MAX_USER_QUERY_CHARS",
    "classify_source_text",
    "classify_user_input",
    "screen_passages",
    "screen_query",
]
