"""Concurrency contract for tenant-isolated ingestion execution."""

from __future__ import annotations

import asyncio
import threading

from document_processing.tenant_jobs import TenantJobCoordinator


def test_different_tenants_run_without_waiting_for_each_other():
    coordinator = TenantJobCoordinator()
    first_started = threading.Event()
    second_started = threading.Event()
    release = threading.Event()

    def blocking_job(started: threading.Event, result: str) -> str:
        started.set()
        assert release.wait(timeout=2)
        return result

    async def scenario() -> list[str]:
        first = asyncio.create_task(
            coordinator.run("student-a", blocking_job, first_started, "a")
        )
        assert await asyncio.to_thread(first_started.wait, 1)
        second = asyncio.create_task(
            coordinator.run("student-b", blocking_job, second_started, "b")
        )
        assert await asyncio.to_thread(second_started.wait, 1), (
            "student-b waited behind student-a"
        )
        release.set()
        return list(await asyncio.gather(first, second))

    assert asyncio.run(scenario()) == ["a", "b"]


def test_one_tenants_jobs_remain_ordered():
    coordinator = TenantJobCoordinator()
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    def first_job() -> str:
        first_started.set()
        assert release_first.wait(timeout=2)
        return "first"

    def second_job() -> str:
        second_started.set()
        return "second"

    async def scenario() -> list[str]:
        first = asyncio.create_task(coordinator.run("student-a", first_job))
        assert await asyncio.to_thread(first_started.wait, 1)
        second = asyncio.create_task(coordinator.run("student-a", second_job))
        await asyncio.sleep(0.05)
        assert not second_started.is_set(), "same-tenant jobs ran concurrently"
        release_first.set()
        return list(await asyncio.gather(first, second))

    assert asyncio.run(scenario()) == ["first", "second"]
