"""Bounded primary/fallback model execution shared by the Agent operations.

A :class:`ResilientLLM` is a drop-in ``str -> str`` callable (the same shape the
agent graph's ``AgentRuntime.llm`` expects), so every operation that already
uses a plain model callable can fail over without changing its validation,
grounding, citation or refusal contracts. Schema validation lives downstream in
:func:`agents.schemas.generate_structured`; this module only decides *which*
provider/model serves the prompt and records *why*.

Rules:

* Fallback is **bounded**: the primary plus each configured fallback is tried
  exactly once, in order.
* Exhaustion raises :class:`FallbackExhausted` explicitly. There is no
  permissive "return anything" path here — a caller that reaches exhaustion
  either surfaces the error or converts it into the existing grounded-refusal
  contract.
* Every served reply records which provider/model handled it, whether a
  fallback was used and why, and the real latency. Token and cost fields are
  only filled in when the provider actually reports them; otherwise they stay
  ``None`` and are rendered as *unknown* — they are never estimated.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

RESILIENCE_SCHEMA = "univai.agent.resilience"
RESILIENCE_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ModelSpec:
    """A named model on a named provider."""

    provider: str
    model: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider is required")
        if not self.model.strip():
            raise ValueError("model is required")

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class ServedResult:
    """One completed model call, with the metadata that is actually known."""

    text: str
    provider: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def spec_label(self) -> str:
        if self.provider and self.model:
            return f"{self.provider}:{self.model}"
        return "unknown"

    def known_metadata(self) -> dict[str, Any]:
        """Only the fields the provider actually supplied. Unknown stays unknown."""
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "attempts": self.attempts,
        }


class ModelBackend(Protocol):
    """A callable model backend. ``complete`` may raise on transient failure."""

    spec: ModelSpec

    def complete(self, prompt: str) -> ServedResult:
        ...


@dataclass
class _CallableBackend:
    """Concrete backend wrapper for a dynamically constructed callable."""

    spec: ModelSpec
    callback: Callable[[str], ServedResult]

    def complete(self, prompt: str) -> ServedResult:
        return self.callback(prompt)


class FallbackExhausted(RuntimeError):
    """Every configured model failed; nothing permissive was returned."""

    def __init__(
        self,
        spec_labels: Sequence[str],
        reasons: Sequence[str | None],
    ):
        self.spec_labels = list(spec_labels)
        self.reasons = list(reasons)
        summary = "; ".join(
            f"{label}: {reason or 'no error reported'}"
            for label, reason in zip(spec_labels, reasons)
        )
        super().__init__(f"all configured models failed: {summary}")


def execute_with_fallback(
    prompt: str,
    primary: ModelBackend,
    fallbacks: Sequence[ModelBackend] = (),
    *,
    on_fallback: Callable[[ModelBackend, Exception], None] | None = None,
    on_error: Callable[[ModelBackend, Exception], None] | None = None,
) -> ServedResult:
    """Run ``primary`` and then each ``fallbacks`` until one replies.

    Each backend is attempted once, so a run of ``n`` backends costs at most
    ``n`` calls. The first non-raising reply wins. If every backend fails,
    :class:`FallbackExhausted` is raised.

    ``on_fallback`` fires when a backend fails *and* another backend remains;
    ``on_error`` fires on every failure, including the last one.
    """
    candidates: list[ModelBackend] = [primary, *fallbacks]
    if not candidates or any(candidate is None for candidate in candidates):
        raise ValueError("at least one model backend is required")

    reasons: list[str | None] = []
    for index, backend in enumerate(candidates):
        started = time.perf_counter()
        try:
            result = backend.complete(prompt)
        except Exception as error:  # noqa: BLE001 - a backend failure is the point
            # Provider exceptions can include request bodies, prompts, URLs or
            # credentials. Keep the operational error type, but never persist
            # the backend-supplied message in serving metadata or logs.
            reasons.append(type(error).__name__)
            if on_error is not None:
                on_error(backend, error)
            if on_fallback is not None and index < len(candidates) - 1:
                on_fallback(backend, error)
            continue

        result.latency_ms = round((time.perf_counter() - started) * 1000, 4)
        result.fallback_used = index > 0
        result.attempts = index + 1
        result.fallback_reason = reasons[-1] if index > 0 and reasons else None
        if result.fallback_used:
            logger.warning(
                "model fallback: %s served after %s failed (%s)",
                result.spec_label,
                candidates[index - 1].spec.label,
                result.fallback_reason,
            )
        return result

    raise FallbackExhausted(
        [candidate.spec.label for candidate in candidates],
        reasons,
    )


class ResilientLLM:
    """A ``str -> str`` model callable that fails over and records servings.

    Wraps :func:`execute_with_fallback` so existing agent code that takes a
    plain callable keeps working unchanged, while every call is recorded on
    :attr:`history` and :attr:`last_served` for the trace and the reports.
    """

    def __init__(
        self,
        primary: ModelBackend,
        fallbacks: Sequence[ModelBackend] = (),
        *,
        on_fallback: Callable[[ModelBackend, Exception], None] | None = None,
    ):
        self.primary = primary
        self.fallbacks = list(fallbacks)
        self.on_fallback = on_fallback
        self.history: list[ServedResult] = []
        self.last_served: ServedResult | None = None
        self._prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self._prompts.append(prompt)
        result = execute_with_fallback(
            prompt,
            self.primary,
            self.fallbacks,
            on_fallback=self.on_fallback,
        )
        self.history.append(result)
        self.last_served = result
        return result.text

    @property
    def calls(self) -> int:
        """Number of executed calls — the same shape ``ScriptedLLM.calls`` has."""
        return len(self.history)

    @property
    def prompts(self) -> list[str]:
        return self._prompts

    @property
    def fallback_calls(self) -> int:
        return sum(1 for result in self.history if result.fallback_used)


def _ollama_backend(
    spec: ModelSpec,
    base_url: str,
    temperature: float = 0,
) -> ModelBackend:
    """Backend over ``langchain_ollama`` reading real token metadata when offered."""

    from langchain_ollama import ChatOllama
    from guardrails.prompt_boundary import split_prompt_roles

    client = ChatOllama(model=spec.model, base_url=base_url, temperature=temperature)

    def complete(prompt: str) -> ServedResult:
        roles = split_prompt_roles(prompt)
        request = (
            [("system", roles[0]), ("human", roles[1])]
            if roles is not None
            else prompt
        )
        response = client.invoke(request)
        content = getattr(response, "content", response)
        text = content if isinstance(content, str) else str(content)

        raw = getattr(response, "response_metadata", None) or {}
        input_tokens = raw.get("prompt_eval_count")
        output_tokens = raw.get("eval_count")
        # Ollama does not report a cost for local inference; keep it unknown.
        return ServedResult(
            text=text,
            provider=spec.provider,
            model=spec.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={"raw": raw},
        )

    return _CallableBackend(spec=spec, callback=complete)


class OllamaBackend:
    """Small wrapper so the backend exposes ``.spec`` and ``.complete``."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        temperature: float = 0,
    ):
        self.spec = ModelSpec(provider=provider, model=model)
        self._impl = _ollama_backend(self.spec, base_url, temperature)

    def complete(self, prompt: str) -> ServedResult:
        return self._impl.complete(prompt)


def build_resilient_llm() -> ResilientLLM:
    """Build the integrated resilient model from environment configuration.

    Primary is ``LLM_MODEL`` at ``LLM_BASE_URL``. When ``LLM_FALLBACK_MODEL`` is
    set, it becomes the single fallback (optionally at ``LLM_FALLBACK_BASE_URL``).
    Tests never call this factory; they inject their own backends.
    """
    import os

    from config import LLM_BASE_URL, LLM_MODEL

    primary = OllamaBackend(
        provider="ollama", model=LLM_MODEL, base_url=LLM_BASE_URL, temperature=0
    )
    fallback_model = os.getenv("LLM_FALLBACK_MODEL", "").strip()
    fallbacks: list[ModelBackend] = []
    if fallback_model:
        fallback_base = os.getenv("LLM_FALLBACK_BASE_URL", LLM_BASE_URL).strip()
        fallbacks.append(
            OllamaBackend(
                provider="ollama",
                model=fallback_model,
                base_url=fallback_base or LLM_BASE_URL,
                temperature=0,
            )
        )
    return ResilientLLM(primary, fallbacks)


__all__ = [
    "RESILIENCE_SCHEMA",
    "RESILIENCE_SCHEMA_VERSION",
    "FallbackExhausted",
    "ModelBackend",
    "ModelSpec",
    "OllamaBackend",
    "ResilientLLM",
    "ServedResult",
    "build_resilient_llm",
    "execute_with_fallback",
]
