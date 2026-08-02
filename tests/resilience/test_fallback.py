"""Bounded primary/fallback model execution with serving metadata.

A fallback must preserve the caller's schema, grounding, citation and refusal
contracts: this module only decides *which* backend serves a prompt. The tests
below verify the fail-over is bounded, visible, and that exhaustion fails
explicitly rather than returning permissive garbage.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from resilience.fallback import (
    FallbackExhausted,
    ModelBackend,
    ModelSpec,
    OllamaBackend,
    ResilientLLM,
    ServedResult,
    execute_with_fallback,
)


@dataclass
class FakeBackend:
    """A scripted model backend. ``fail_with`` makes it raise instead of reply."""

    spec: ModelSpec
    reply: str = "primary reply"
    fail_with: Exception | None = None
    calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None

    def complete(self, prompt: str) -> ServedResult:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return ServedResult(
            text=self.reply,
            provider=self.spec.provider,
            model=self.spec.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost=self.cost,
        )


def spec(provider: str = "ollama", model: str = "qwen3:4b-instruct") -> ModelSpec:
    return ModelSpec(provider=provider, model=model)


def test_primary_success_needs_no_fallback():
    primary = FakeBackend(spec(), reply="ok")
    result = execute_with_fallback("prompt", primary)

    assert result.text == "ok"
    assert result.fallback_used is False
    assert result.attempts == 1
    assert result.provider == "ollama"
    assert result.model == "qwen3:4b-instruct"
    assert result.latency_ms is not None
    assert primary.calls == 1


def test_fallback_serves_when_the_primary_fails():
    primary = FakeBackend(spec(), fail_with=RuntimeError("connection refused"))
    fallback = FakeBackend(spec(provider="ollama", model="fallback-model"), reply="fallback ok")
    on_fallback: list[tuple[ModelBackend, Exception]] = []

    result = execute_with_fallback(
        "prompt", primary, [fallback], on_fallback=lambda b, e: on_fallback.append((b, e))
    )

    assert result.text == "fallback ok"
    assert result.fallback_used is True
    assert result.fallback_reason is not None
    assert result.fallback_reason == "RuntimeError"
    assert "connection refused" not in result.fallback_reason
    assert result.attempts == 2
    assert result.model == "fallback-model"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert len(on_fallback) == 1


def test_on_fallback_receives_the_failed_backend_and_error():
    primary = FakeBackend(spec(), fail_with=RuntimeError("boom"))
    fallback = FakeBackend(spec(model="f2"), reply="ok")
    seen: list[tuple[str, str]] = []

    def record(backend, error):
        seen.append((backend.spec.label, str(error)))

    execute_with_fallback("prompt", primary, [fallback], on_fallback=record)

    assert seen == [("ollama:qwen3:4b-instruct", "boom")]


def test_exhaustion_fails_explicitly_and_bounded():
    primary = FakeBackend(spec(), fail_with=RuntimeError("primary down"))
    fallback = FakeBackend(spec(model="f2"), fail_with=RuntimeError("fallback down"))
    on_error: list[Exception] = []

    with pytest.raises(FallbackExhausted) as error:
        execute_with_fallback(
            "prompt", primary, [fallback], on_error=lambda b, e: on_error.append(e)
        )

    assert "all configured models failed" in str(error.value)
    assert len(error.value.spec_labels) == 2
    assert primary.calls == 1
    assert fallback.calls == 1, "each backend is attempted exactly once"
    assert len(on_error) == 2


def test_backend_errors_do_not_leak_messages_into_exhaustion_metadata():
    secret = "Bearer secret-token prompt=private"
    primary = FakeBackend(spec(), fail_with=RuntimeError(secret))

    with pytest.raises(FallbackExhausted) as error:
        execute_with_fallback("prompt", primary)

    assert error.value.reasons == ["RuntimeError"]
    assert secret not in str(error.value)


def test_multiple_fallbacks_are_tried_in_order():
    first = FakeBackend(spec(model="f1"), fail_with=RuntimeError("f1 down"))
    second = FakeBackend(spec(model="f2"), reply="second wins")
    third = FakeBackend(spec(model="f3"), reply="never reached")

    result = execute_with_fallback("prompt", first, [second, third])

    assert result.model == "f2"
    assert result.attempts == 2
    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 0


def test_at_least_one_backend_is_required():
    with pytest.raises(ValueError, match="at least one model backend"):
        execute_with_fallback("prompt", None)  # type: ignore[arg-type]
def test_served_result_keeps_token_and_cost_unknown_when_not_reported():
    primary = FakeBackend(spec(), reply="ok")
    result = execute_with_fallback("prompt", primary)

    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.cost is None
    known = result.known_metadata()
    assert known["input_tokens"] is None


def test_served_result_reports_real_token_metadata_when_supplied():
    primary = FakeBackend(
        spec(), reply="ok", input_tokens=12, output_tokens=34, cost=0.0002
    )
    result = execute_with_fallback("prompt", primary)

    assert result.input_tokens == 12
    assert result.output_tokens == 34
    assert result.cost == 0.0002


# ── ResilientLLM: the str -> str contract the agent graph uses ────────


def test_resilient_llm_is_a_drop_in_str_to_str_callable():
    llm = ResilientLLM(FakeBackend(spec(), reply='{"ok": true}'))

    assert llm("prompt") == '{"ok": true}'
    assert llm.calls == 1
    assert llm.history[0].fallback_used is False


def test_resilient_llm_records_fallback_servings():
    primary = FakeBackend(spec(), fail_with=RuntimeError("primary down"))
    fallback = FakeBackend(spec(model="f2"), reply="served by fallback")
    llm = ResilientLLM(primary, [fallback])

    assert llm("prompt") == "served by fallback"
    assert llm.calls == 1
    assert llm.fallback_calls == 1
    assert llm.last_served is not None
    assert llm.last_served.fallback_used is True
    assert llm.last_served.model == "f2"


def test_resilient_llm_exhaustion_raises_without_an_answer():
    llm = ResilientLLM(
        FakeBackend(spec(), fail_with=RuntimeError("a")),
        [FakeBackend(spec(model="f2"), fail_with=RuntimeError("b"))],
    )

    with pytest.raises(FallbackExhausted):
        llm("prompt")
    assert llm.calls == 0, "a raised call is not recorded as a serving"


def test_primary_spec_label_is_well_formed():
    assert spec().label == "ollama:qwen3:4b-instruct"


def test_model_spec_requires_provider_and_model():
    with pytest.raises(ValueError):
        ModelSpec(provider="", model="x")
    with pytest.raises(ValueError):
        ModelSpec(provider="ollama", model="")


def test_ollama_backend_factory_exposes_the_requested_spec(monkeypatch):
    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        SimpleNamespace(ChatOllama=FakeChatOllama),
    )

    backend = OllamaBackend(
        provider="ollama",
        model="primary-model",
        base_url="http://localhost:11434",
    )

    assert backend.spec == ModelSpec(provider="ollama", model="primary-model")
