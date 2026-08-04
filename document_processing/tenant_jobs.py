"""Tenant-fair execution for blocking ingestion work.

FastMCP can serve async tools concurrently, but PDF parsing and FastEmbed are
blocking functions.  Running them directly in a tool handler freezes the MCP
event loop and makes every learner wait behind the largest active textbook.

Each tenant gets an independent lock: one learner's own uploads stay ordered,
while different learners run concurrently in worker threads.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar


ResultT = TypeVar("ResultT")


class TenantJobCoordinator:
    """Run blocking work concurrently across tenants and serially within one."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, tenant_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(tenant_id, asyncio.Lock())

    async def run(
        self,
        tenant_id: str,
        work: Callable[..., ResultT],
        /,
        *args: object,
    ) -> ResultT:
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required")
        tenant_lock = await self._lock_for(tenant_id)
        async with tenant_lock:
            return await asyncio.to_thread(work, *args)


INGESTION_JOBS = TenantJobCoordinator()


__all__ = ["INGESTION_JOBS", "TenantJobCoordinator"]
