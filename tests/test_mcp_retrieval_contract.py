from __future__ import annotations

import pytest

import mcp_server
from runtime import RuntimeMode


def test_legacy_retrieval_reports_operational_failure_through_mcp(monkeypatch) -> None:
    def unavailable(**_kwargs):
        raise ConnectionError("qdrant is down")

    monkeypatch.setattr(mcp_server, "runtime_mode", lambda: RuntimeMode.INTEGRATED)
    monkeypatch.setattr(mcp_server, "retrieve_formatted", unavailable)

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        mcp_server.retrieve_context(
            query="What is virtual memory?",
            user_id="S-2026-000001",
        )


def test_grounded_retrieval_reports_operational_failure_through_mcp(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise ConnectionError("grant store is down")

    monkeypatch.setattr(mcp_server, "call_tool", unavailable)

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        mcp_server.retrieve_grounded_context(
            query="What is virtual memory?",
            user_id="S-2026-000001",
        )
