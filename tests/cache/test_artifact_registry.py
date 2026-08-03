"""Artifact registry: one-writer builds, verification, and grant ref-counting.

The cache must be *safe* to share: concurrent identical uploads build once, a
crashed build is recovered, a corrupt artifact is never reused, and deleting
one tenant's book never touches another's until the final reference is gone.
These tests pin exactly that.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cache.artifact_registry import (
    ARTIFACT_SCHEMA,
    ArtifactBuildTimeout,
    ArtifactChunk,
    ArtifactRegistry,
    ArtifactState,
    BuildClaim,
    ContentArtifact,
)
from cache.authorization import GrantDenied, GrantStore, revoke_document
from cache.content_identity import artifact_key

KEY_HASH = "ab" * 32
KEY_FINGERPRINT = "cd" * 32
KEY = artifact_key(KEY_HASH, KEY_FINGERPRINT)


def _artifact(
    key: str = KEY,
    texts: list[str] | None = None,
    pages: list[int | None] | None = None,
    *,
    book_title: str = "Book",
    document_type: str = "md",
    content_hash: str = KEY_HASH,
    byte_size: int = 6,
    state: ArtifactState = ArtifactState.BUILDING,
) -> ContentArtifact:
    texts = ["body"] if texts is None else texts
    pages = pages or [None] * len(texts)
    return ContentArtifact(
        artifact_key=key,
        content_hash=content_hash,
        byte_size=byte_size,
        pipeline_fingerprint=KEY_FINGERPRINT,
        document_type=document_type,
        book_title=book_title,
        chunks=[ArtifactChunk(text=text, page=page) for text, page in zip(texts, pages)],
        state=state,
    )


def _registry(tmp_path: Path, **kwargs) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "cache", **kwargs)


# ── One-writer, atomic state machine ──────────────────────────────────


def test_claim_then_publish_makes_the_artifact_ready(tmp_path):
    registry = _registry(tmp_path)
    assert registry.claim_build(KEY) is BuildClaim.CLAIMED
    registry.publish(_artifact())

    artifact = registry.get(KEY)
    assert artifact is not None
    assert artifact.state is ArtifactState.READY
    assert artifact.chunk_count == 1
    assert artifact.artifact_key == KEY


def test_a_second_claim_sees_the_ready_artifact_without_building(tmp_path):
    registry = _registry(tmp_path)
    registry.claim_build(KEY)
    registry.publish(_artifact())
    assert registry.claim_build(KEY) is BuildClaim.READY


def test_publish_rejects_an_artifact_with_no_chunks(tmp_path):
    registry = _registry(tmp_path)
    registry.claim_build(KEY)
    with pytest.raises(ValueError, match="no chunks"):
        registry.publish(_artifact(texts=[]))


def test_publishing_without_a_claim_is_rejected(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(Exception):
        registry.publish(_artifact())


def test_a_failed_build_is_retryable(tmp_path):
    registry = _registry(tmp_path)
    assert registry.claim_build(KEY) is BuildClaim.CLAIMED
    registry.mark_failed(KEY, "embedding download failed")
    assert registry.claim_build(KEY) is BuildClaim.CLAIMED


def test_an_abandoned_build_is_recovered(tmp_path):
    registry = _registry(tmp_path)
    assert registry.claim_build(KEY) is BuildClaim.CLAIMED

    stale = registry.record(KEY)
    stale["claimed_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=1)
    ).isoformat()
    (tmp_path / "cache" / "state.json").write_text(
        json.dumps({KEY: stale}, indent=2) + "\n", encoding="utf-8"
    )

    assert registry.claim_build(KEY, builder_id="alive-process") is BuildClaim.RECOVERED
    record = registry.record(KEY)
    assert record["builder_id"] == "alive-process"


def test_ensure_ready_builds_once_and_then_hits(tmp_path):
    registry = _registry(tmp_path)
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return _artifact()

    first, outcome = registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=build
    )
    assert outcome == "build"
    assert calls["n"] == 1

    second, outcome = registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=build
    )
    assert outcome == "hit"
    assert calls["n"] == 1
    assert first.artifact_key == second.artifact_key


def test_concurrent_writers_run_exactly_one_build(tmp_path):
    registry = _registry(tmp_path, build_timeout_seconds=10.0)
    calls = {"n": 0}
    outcomes: dict[str, str] = {}

    def build():
        calls["n"] += 1
        time.sleep(0.05)
        return _artifact()

    def worker(name: str) -> None:
        _, outcome = registry.ensure_ready(
            KEY,
            content_hash=KEY_HASH,
            byte_size=6,
            build=build,
            builder_id=name,
        )
        outcomes[name] = outcome

    threads = [threading.Thread(target=worker, args=(f"worker-{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls["n"] == 1, "identical concurrent uploads must build exactly once"
    assert list(outcomes.values()).count("build") == 1
    assert set(outcomes.values()) == {"build", "hit"}
    assert registry.get(KEY) is not None


def test_ensure_ready_waits_for_a_concurrent_build(tmp_path):
    registry = _registry(tmp_path, build_timeout_seconds=5.0, poll_interval_seconds=0.01)
    started = threading.Event()

    def builder() -> None:
        assert registry.claim_build(KEY) is BuildClaim.CLAIMED
        started.set()
        time.sleep(0.1)
        registry.publish(_artifact())

    thread = threading.Thread(target=builder)
    thread.start()
    started.wait(timeout=2.0)

    start = time.monotonic()
    artifact, outcome = registry.ensure_ready(
        KEY,
        content_hash=KEY_HASH,
        byte_size=6,
        build=lambda: _artifact(),
        builder_id="the-waiter",
    )
    elapsed = time.monotonic() - start
    thread.join()

    assert outcome == "hit"
    assert elapsed >= 0.05, "the waiter must poll until the writer publishes"
    assert artifact.chunk_count == 1


def test_ensure_ready_raises_after_the_bounded_wait(tmp_path):
    registry = _registry(
        tmp_path, build_timeout_seconds=0.4, poll_interval_seconds=0.02
    )
    registry.claim_build(KEY)

    with pytest.raises(ArtifactBuildTimeout):
        registry.ensure_ready(
            KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
        )


# ── Verification rejects corruption ───────────────────────────────────


def test_verify_reusable_returns_the_verified_artifact(tmp_path):
    registry = _registry(tmp_path)
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )
    artifact = registry.verify_reusable(KEY, content_hash=KEY_HASH, byte_size=6)
    assert artifact is not None
    assert artifact.chunk_count == 1


def test_a_length_mismatch_marks_the_artifact_corrupt(tmp_path):
    registry = _registry(tmp_path)
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )

    assert registry.verify_reusable(KEY, content_hash=KEY_HASH, byte_size=999) is None
    assert registry.record(KEY)["state"] == ArtifactState.CORRUPT.value
    assert registry.get(KEY) is None
    assert "artifact_corrupt" in [event["event"] for event in registry.events()]


def test_a_hash_mismatch_marks_the_artifact_corrupt(tmp_path):
    registry = _registry(tmp_path)
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )
    assert registry.verify_reusable(KEY, content_hash="ff" * 32, byte_size=6) is None
    assert registry.record(KEY)["state"] == ArtifactState.CORRUPT.value


def test_tampered_payload_identity_is_rejected(tmp_path):
    registry = _registry(tmp_path)
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )
    payload = json.loads(registry.payload_path(KEY).read_text(encoding="utf-8"))
    payload["content_hash"] = "ff" * 32
    registry.payload_path(KEY).write_text(json.dumps(payload), encoding="utf-8")

    assert registry.verify_reusable(KEY, content_hash=KEY_HASH, byte_size=6) is None
    assert registry.record(KEY)["state"] == ArtifactState.CORRUPT.value


def test_abandoned_registry_lock_is_recovered(tmp_path):
    registry = _registry(tmp_path, lock_timeout_seconds=0.05)
    registry.lock_file.write_text("abandoned", encoding="utf-8")
    old = time.time() - 60
    os.utime(registry.lock_file, (old, old))

    assert registry.claim_build(KEY) is BuildClaim.CLAIMED


def test_a_corrupt_artifact_is_rebuilt_not_reused(tmp_path):
    registry = _registry(tmp_path)
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return _artifact()

    registry.ensure_ready(KEY, content_hash=KEY_HASH, byte_size=6, build=build)
    assert registry.verify_reusable(KEY, content_hash="ff" * 32, byte_size=6) is None

    artifact, outcome = registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=build
    )
    assert outcome == "build"
    assert calls["n"] == 2
    assert artifact.chunk_count == 1


# ── Cleanup is idempotent and audited ────────────────────────────────


def test_remove_is_idempotent_and_audited(tmp_path):
    registry = _registry(tmp_path)
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )

    assert registry.remove(KEY) is True
    assert registry.remove(KEY) is False
    assert registry.get(KEY) is None
    cleanup = [event for event in registry.events() if event["event"] == "artifact_cleanup"]
    assert len(cleanup) == 1
    assert cleanup[0]["artifact_key"] == KEY


def test_events_never_carry_content_or_tenant_identity(tmp_path):
    registry = _registry(tmp_path)
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact(texts=["secret body"])
    )
    registry.remove(KEY)

    assert registry.events()
    for event in registry.events():
        serialised = json.dumps(event)
        assert "secret body" not in serialised
        for forbidden in ("user_id", "student-", "source_filename", "query"):
            assert forbidden not in serialised, f"event leaked '{forbidden}'"


def test_pipeline_fingerprint_is_recorded_in_telemetry(tmp_path):
    registry = _registry(tmp_path)
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )
    built = [event for event in registry.events() if event["event"] == "artifact_built"]
    assert built and built[0]["pipeline_fingerprint"] == KEY_FINGERPRINT


# ── Tenant grants and reference counting ──────────────────────────────


def _grant(grants: GrantStore, user: str, document: str, key: str = KEY) -> object:
    return grants.grant(
        artifact_key=key,
        user_id=user,
        collection_id="programme-a",
        document_id=document,
        source_filename="book.md",
        document_type="md",
        book_title="Book",
    )


def test_two_tenants_get_independent_grants_on_one_artifact(tmp_path):
    grants = GrantStore(tmp_path / "cache")
    first = _grant(grants, "student-a", "doc-a")
    second = _grant(grants, "student-b", "doc-b")

    assert first.grant_id != second.grant_id
    assert first.artifact_key == second.artifact_key == KEY
    assert grants.active_refcount(KEY) == 2


def test_grant_is_idempotent_per_tenant_and_document(tmp_path):
    grants = GrantStore(tmp_path / "cache")
    first = _grant(grants, "student-a", "doc-a")
    again = _grant(grants, "student-a", "doc-a")

    assert again.grant_id == first.grant_id
    assert grants.active_refcount(KEY) == 1


def test_grant_replaces_a_stale_artifact_mapping_atomically(tmp_path):
    grants = GrantStore(tmp_path / "cache")
    first = _grant(grants, "student-a", "doc-a")
    replacement_key = artifact_key(KEY_HASH, "ef" * 32)

    replacement = _grant(grants, "student-a", "doc-a", replacement_key)

    assert replacement.grant_id != first.grant_id
    assert replacement.artifact_key == replacement_key
    assert grants.active_refcount(KEY) == 0
    assert grants.active_refcount(replacement_key) == 1


def test_require_grant_denies_an_unrelated_tenant(tmp_path):
    grants = GrantStore(tmp_path / "cache")
    _grant(grants, "student-a", "doc-a")

    assert grants.is_granted("student-a", "doc-a") is True
    assert grants.is_granted("student-c", "doc-a") is False
    with pytest.raises(GrantDenied):
        grants.require_grant("student-c", "doc-a")


def test_active_refcount_counts_only_active_grants(tmp_path):
    grants = GrantStore(tmp_path / "cache")
    _grant(grants, "student-a", "doc-a")
    _grant(grants, "student-b", "doc-b")
    grants.revoke("student-a", "doc-a")

    assert grants.active_refcount(KEY) == 1


def test_authorized_document_ids_are_tenant_and_scope_filtered(tmp_path):
    grants = GrantStore(tmp_path / "cache")
    _grant(grants, "student-a", "doc-a")
    grants.grant(
        artifact_key=KEY,
        user_id="student-a",
        collection_id="programme-b",
        document_id="doc-b",
        source_filename="other.md",
        document_type="md",
        book_title="Other Book",
    )
    _grant(grants, "student-b", "doc-c")

    assert grants.authorized_document_ids("student-a") == ["doc-a", "doc-b"]
    assert grants.authorized_document_ids(
        "student-a", collection_id="programme-a"
    ) == ["doc-a"]
    assert grants.authorized_document_ids(
        "student-a", book_titles=["Other Book"]
    ) == ["doc-b"]
    assert grants.authorized_document_ids("student-c") == []


def test_revoking_one_tenant_does_not_clean_up_the_shared_artifact(tmp_path):
    registry = _registry(tmp_path)
    grants = GrantStore(tmp_path / "cache")
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )
    _grant(grants, "student-a", "doc-a")
    _grant(grants, "student-b", "doc-b")

    report = revoke_document(grants, registry, user_id="student-a", document_id="doc-a")

    assert report.revoked is True
    assert report.last_reference is False
    assert report.artifact_removed is False
    assert registry.get(KEY) is not None, "tenant B still needs the artifact"
    assert grants.is_granted("student-b", "doc-b") is True


def test_revoking_the_last_grant_makes_cleanup_eligible(tmp_path):
    registry = _registry(tmp_path)
    grants = GrantStore(tmp_path / "cache")
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )
    _grant(grants, "student-a", "doc-a")
    _grant(grants, "student-b", "doc-b")

    revoke_document(grants, registry, user_id="student-a", document_id="doc-a")
    assert registry.is_cleanup_eligible(KEY, grants.active_refcount(KEY)) is False

    report = revoke_document(grants, registry, user_id="student-b", document_id="doc-b")

    assert report.last_reference is True
    assert report.artifact_removed is True
    assert registry.get(KEY) is None
    assert registry.record(KEY) is None
    assert registry.is_cleanup_eligible(KEY, 0) is False


def test_last_reference_cleanup_retries_after_an_interrupted_delete(
    tmp_path, monkeypatch
):
    registry = _registry(tmp_path)
    grants = GrantStore(tmp_path / "cache")
    registry.ensure_ready(
        KEY, content_hash=KEY_HASH, byte_size=6, build=lambda: _artifact()
    )
    _grant(grants, "student-a", "doc-a")
    real_remove = registry.remove
    attempts = 0

    def interrupted_once(artifact_key: str) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated interrupted cleanup")
        return real_remove(artifact_key)

    monkeypatch.setattr(registry, "remove", interrupted_once)
    with pytest.raises(OSError, match="interrupted cleanup"):
        revoke_document(grants, registry, user_id="student-a", document_id="doc-a")

    retry = revoke_document(grants, registry, user_id="student-a", document_id="doc-a")
    assert retry.revoked is False
    assert attempts == 2
    assert registry.get(KEY) is None


def test_revoking_an_absent_grant_is_a_noop(tmp_path):
    registry = _registry(tmp_path)
    grants = GrantStore(tmp_path / "cache")
    report = revoke_document(grants, registry, user_id="student-a", document_id="none")

    assert report.revoked is False
    assert report.artifact_removed is False


def test_revoking_twice_is_a_noop(tmp_path):
    grants = GrantStore(tmp_path / "cache")
    _grant(grants, "student-a", "doc-a")

    assert grants.revoke("student-a", "doc-a") is not None
    assert grants.revoke("student-a", "doc-a") is None
