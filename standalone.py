"""Cross-platform standalone commands for the Agent repository."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from contracts import validate_course
from runtime import REPOSITORY_ROOT, runtime_mode, standalone_root
from standalone_generation import generate_course
from standalone_store import reset

FIXTURE = REPOSITORY_ROOT / "fixtures" / "sample_course.md"
FIXTURE_USER = "standalone-student"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _tool_text(result: object) -> str:
    return "\n".join(
        getattr(block, "text", "") for block in getattr(result, "content", [])
    )


async def _exercise_mcp(port: int) -> None:
    url = f"http://127.0.0.1:{port}/mcp"
    deadline = time.monotonic() + 30
    last_error = "no connection attempt completed"
    while time.monotonic() < deadline:
        try:
            async with streamable_http_client(url) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    info = _tool_text(await session.call_tool("server_info", {}))
                    assert "Mode: standalone" in info
                    ingested = _tool_text(
                        await session.call_tool(
                            "ingest_file",
                            {"file_path": str(FIXTURE), "user_id": FIXTURE_USER},
                        )
                    )
                    match = re.search(r"Document ID: ([a-f0-9-]+)", ingested)
                    assert match, ingested
                    document_id = match.group(1)
                    listed = _tool_text(
                        await session.call_tool(
                            "list_documents", {"user_id": FIXTURE_USER}
                        )
                    )
                    assert "sample_course.md" in listed
                    known = _tool_text(
                        await session.call_tool(
                            "retrieve_context",
                            {
                                "query": "How does tenant isolation protect learners?",
                                "user_id": FIXTURE_USER,
                            },
                        )
                    )
                    assert "Tenant Isolation" in known
                    isolated = _tool_text(
                        await session.call_tool(
                            "retrieve_context",
                            {
                                "query": "tenant isolation",
                                "user_id": "another-student",
                            },
                        )
                    )
                    assert isolated == "No relevant documents found."
                    removed = _tool_text(
                        await session.call_tool(
                            "remove_document",
                            {
                                "user_id": FIXTURE_USER,
                                "document_id": document_id,
                            },
                        )
                    )
                    assert "Removed" in removed
                    return
        except Exception as error:
            last_error = "".join(traceback.format_exception(error))
            await asyncio.sleep(0.2)
    raise RuntimeError(f"Standalone MCP did not pass within 30 seconds:\n{last_error}")


def smoke() -> int:
    if runtime_mode().value != "standalone":
        raise RuntimeError("Smoke requires UNIVAI_MODE=standalone")
    reset()
    port = _free_port()
    environment = {
        **os.environ,
        "UNIVAI_MODE": "standalone",
        "FASTMCP_HOST": "127.0.0.1",
        "FASTMCP_PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        asyncio.run(_exercise_mcp(port))
        with tempfile.TemporaryDirectory(prefix="univai-agent-smoke-") as directory:
            output = generate_course(FIXTURE, Path(directory))
            validate_course(output)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reset()
    print(json.dumps({"ok": True, "mode": "standalone", "weeks": 4}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "generate", "reset", "status"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        return smoke()
    if args.command == "reset":
        reset()
        print("Standalone store reset.")
        return 0
    if args.command == "status":
        print(
            json.dumps(
                {
                    "mode": runtime_mode().value,
                    "root": str(standalone_root()),
                    "fixture": str(FIXTURE),
                },
                indent=2,
            )
        )
        return 0
    output = args.output or standalone_root() / "output"
    generate_course(FIXTURE, output.resolve())
    print(json.dumps({"ok": True, "output": str(output.resolve()), "weeks": 4}))
    return 0


if __name__ == "__main__":
    os.environ["UNIVAI_MODE"] = "standalone"
    raise SystemExit(main())
