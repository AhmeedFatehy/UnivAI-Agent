"""Explicit runtime selection shared by the Agent entry points."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class RuntimeMode(str, Enum):
    STANDALONE = "standalone"
    INTEGRATED = "integrated"


REPOSITORY_ROOT = Path(__file__).resolve().parent


def runtime_mode() -> RuntimeMode:
    raw = os.getenv("UNIVAI_MODE", RuntimeMode.INTEGRATED.value).strip().lower()
    try:
        mode = RuntimeMode(raw)
    except ValueError as exc:
        raise RuntimeError(
            "UNIVAI_MODE must be 'standalone' or 'integrated'"
        ) from exc

    if mode is RuntimeMode.STANDALONE and os.getenv("UNIVAI_ENV", "").lower() in {
        "production",
        "prod",
    }:
        raise RuntimeError("Standalone fixture providers are disabled in production")
    return mode


def standalone_root() -> Path:
    configured = os.getenv("UNIVAI_STANDALONE_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else (REPOSITORY_ROOT / ".standalone").resolve()
    )
    return root


def ensure_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{label} must stay inside {resolved_root}")
    return resolved
