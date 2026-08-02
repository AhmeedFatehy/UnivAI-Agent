"""Resilient model execution — bounded primary/fallback with serving metadata."""

from resilience.fallback import (
    FallbackExhausted,
    ModelBackend,
    ModelSpec,
    OllamaBackend,
    ResilientLLM,
    ServedResult,
    build_resilient_llm,
    execute_with_fallback,
)

__all__ = [
    "FallbackExhausted",
    "ModelBackend",
    "ModelSpec",
    "OllamaBackend",
    "ResilientLLM",
    "ServedResult",
    "build_resilient_llm",
    "execute_with_fallback",
]
