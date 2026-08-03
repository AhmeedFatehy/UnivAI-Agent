"""Atomic, one-writer registry of immutable content-addressed artifacts.

Two tenants uploading identical bytes must not both parse/chunk/embed the book.
:class:`ArtifactRegistry` gives every artifact a single record with an explicit
state machine — ``building → ready | failed | corrupt`` — so that:

* exactly one writer owns a build (:meth:`claim_build` uses an exclusive lock);
* a crashed build is detected by its age and taken over (:meth:`claim_build`);
* concurrent uploads with the same bytes wait a bounded time for the one writer
  and then reuse its result (:meth:`ensure_ready`);
* a READY artifact is only reused after its recorded byte length and SHA-256 are
  re-verified against the current source bytes (:meth:`verify_reusable`); a
  mismatch or a missing/empty payload is rejected and the artifact is marked
  ``corrupt`` — never trusted on filename, title or a client hash;
* cleanup is idempotent and audited through an append-only ``events.jsonl``
  (:meth:`remove`).

The registry never stores tenant identity, filenames or raw book text: chunk
payloads live per-artifact beside the record, and events carry only the key,
fingerprint, size and outcome — private telemetry without raw book content.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from runtime import REPOSITORY_ROOT

ARTIFACT_SCHEMA = "univai.agent.content_artifact"
ARTIFACT_SCHEMA_VERSION = "1.0.0"

#: Environment variable naming the on-disk cache root.
CACHE_ROOT_ENV = "UNIVAI_ARTIFACT_CACHE_ROOT"
DEFAULT_BUILD_TIMEOUT_SECONDS = 60.0
DEFAULT_LOCK_TIMEOUT_SECONDS = 5.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.05


class ArtifactState(str, Enum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    CORRUPT = "corrupt"


class BuildClaim(str, Enum):
    CLAIMED = "claimed"      # this writer must run the build
    RECOVERED = "recovered"  # a stale/abandoned build was taken over
    READY = "ready"          # a verified artifact already exists
    WAIT = "wait"            # another writer owns the build; poll later


class ArtifactChunk(BaseModel):
    """One reusable chunk: the parsed text, the loader page, and the vectors."""

    text: str
    page: int | None = Field(default=None, ge=1)
    dense_vector: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None


class ContentArtifact(BaseModel):
    """The immutable artifact: identity, pipeline, chunks and build state."""

    schema_name: str = ARTIFACT_SCHEMA
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    artifact_key: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    pipeline_fingerprint: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    # Only a title extracted from the immutable bytes belongs here. A
    # filename-derived title is tenant metadata and must stay on the grant.
    book_title: str | None = None
    chunks: list[ArtifactChunk] = Field(default_factory=list)
    state: ArtifactState = ArtifactState.BUILDING
    build_id: str = ""
    builder_id: str = ""
    claimed_at: str = ""
    ready_at: str | None = None
    failed_reason: str | None = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def chunk_texts(self) -> list[str]:
        return [chunk.text for chunk in self.chunks]

    @property
    def pages(self) -> list[int | None]:
        return [chunk.page for chunk in self.chunks]

    def embeddings(self) -> list[tuple[list[float] | None, list[int] | None, list[float] | None]]:
        """Per-chunk ``(dense, sparse_indices, sparse_values)`` triples."""
        return [
            (chunk.dense_vector, chunk.sparse_indices, chunk.sparse_values)
            for chunk in self.chunks
        ]


class ArtifactBuildTimeout(RuntimeError):
    """A concurrent build did not finish within the bounded wait."""


class ArtifactNotClaimed(RuntimeError):
    """Publishing requires an active building claim by the same key."""


class _FileLock:
    """Advisory exclusive lock backed by an ``O_CREAT | O_EXCL`` file."""

    def __init__(self, path: Path, timeout_seconds: float):
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._held = False

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
                os.close(descriptor)
                self._held = True
                return
            except FileExistsError:
                # An abruptly terminated process cannot remove its lock file.
                # Registry critical sections are tiny, so a lock older than a
                # conservative floor is abandoned and may be recovered.
                try:
                    stale_after = max(30.0, self.timeout_seconds * 2)
                    if time.time() - self.path.stat().st_mtime > stale_after:
                        os.unlink(self.path)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise ArtifactBuildTimeout(
                        f"timed out waiting for registry lock {self.path}"
                    )
                time.sleep(0.01)

    def release(self) -> None:
        if self._held:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self._held = False


def default_cache_root() -> Path:
    """The on-disk cache root: ``UNIVAI_ARTIFACT_CACHE_ROOT`` or ``.cache/artifacts``."""
    configured = os.getenv(CACHE_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPOSITORY_ROOT / ".cache" / "artifacts").resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _builder_id() -> str:
    return f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex[:8]}"


class ArtifactRegistry:
    """File-backed registry with atomic claim/ready/failed/corrupt transitions.

    The lightweight ``state.json`` holds one small record per artifact; the
    heavy chunk payload lives in ``artifacts/<key>.json`` and is only written
    *before* the state flips to READY, so READY never dangles on a missing
    payload. ``events.jsonl`` is an append-only telemetry/audit log.
    """

    def __init__(
        self,
        root: Path,
        *,
        build_timeout_seconds: float = DEFAULT_BUILD_TIMEOUT_SECONDS,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self.root = Path(root).expanduser().resolve()
        self.state_file = self.root / "state.json"
        self.payload_dir = self.root / "artifacts"
        self.events_file = self.root / "events.jsonl"
        self.lock_file = self.root / ".registry.lock"
        self.build_timeout_seconds = build_timeout_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._thread_lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.payload_dir.mkdir(parents=True, exist_ok=True)

    # ── private telemetry / audit ──────────────────────────────────────

    def record_event(self, event: str, **fields: object) -> None:
        """Append one private telemetry event. Never content, user or filename."""
        row: dict[str, object] = {"ts": _now(), "event": event}
        row.update(fields)
        with self._thread_lock:
            with open(self.events_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def events(self) -> list[dict]:
        """Every recorded event, oldest first."""
        if not self.events_file.exists():
            return []
        return [
            json.loads(line)
            for line in self.events_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def cache_events(self) -> list[dict]:
        return [event for event in self.events() if event.get("event", "").startswith("cache_")]

    # ── locking ────────────────────────────────────────────────────────

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            lock = _FileLock(self.lock_file, self.lock_timeout_seconds)
            lock.acquire()
            try:
                yield
            finally:
                lock.release()

    # ── storage ────────────────────────────────────────────────────────

    def _read_state(self) -> dict[str, dict]:
        if not self.state_file.exists():
            return {}
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, dict]) -> None:
        temporary = self.state_file.with_name("state.json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_file)

    def _read_payload(self, artifact_key: str) -> ContentArtifact | None:
        path = self.payload_dir / f"{artifact_key}.json"
        if not path.exists():
            return None
        try:
            return ContentArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def _write_payload(self, artifact: ContentArtifact) -> None:
        path = self.payload_dir / f"{artifact.artifact_key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(artifact.model_dump_json(), encoding="utf-8")
        os.replace(temporary, path)

    def payload_path(self, artifact_key: str) -> Path:
        return self.payload_dir / f"{artifact_key}.json"

    # ── state machine ──────────────────────────────────────────────────

    def record(self, artifact_key: str) -> dict | None:
        """The current state record for an artifact, or ``None``."""
        with self._locked():
            return self._read_state().get(artifact_key)

    def get(self, artifact_key: str) -> ContentArtifact | None:
        """A READY artifact's payload, or ``None``."""
        record = self.record(artifact_key)
        if record is None or record.get("state") != ArtifactState.READY.value:
            return None
        return self._read_payload(artifact_key)

    def claim_build(
        self,
        artifact_key: str,
        *,
        builder_id: str | None = None,
    ) -> BuildClaim:
        """Claim the right to build an artifact — exactly one writer wins.

        Returns :attr:`BuildClaim.CLAIMED` for a fresh build, ``RECOVERED`` when
        a stale (abandoned) build is taken over, ``READY`` when a verified
        artifact already exists, or ``WAIT`` when another writer is building.
        """
        builder_id = builder_id or _builder_id()
        now = _now()
        with self._locked():
            state = self._read_state()
            record = state.get(artifact_key)

            if record is None:
                state[artifact_key] = self._new_record(
                    artifact_key, builder_id, now
                )
                self._write_state(state)
                self.record_event(
                    "cache_build_started", artifact_key=artifact_key, builder_id=builder_id
                )
                return BuildClaim.CLAIMED

            current = record.get("state")
            if current == ArtifactState.READY.value:
                return BuildClaim.READY

            if current == ArtifactState.BUILDING.value:
                if self._is_stale(record):
                    previous_builder = record.get("builder_id")
                    record["build_id"] = str(uuid.uuid4())
                    record["builder_id"] = builder_id
                    record["claimed_at"] = now
                    record["ready_at"] = None
                    self._write_state(state)
                    self.record_event(
                        "cache_build_recovered",
                        artifact_key=artifact_key,
                        builder_id=builder_id,
                        previous_builder=previous_builder,
                    )
                    return BuildClaim.RECOVERED
                return BuildClaim.WAIT

            # failed or corrupt → the build may be retried, atomically.
            previous = current
            record["state"] = ArtifactState.BUILDING.value
            record["build_id"] = str(uuid.uuid4())
            record["builder_id"] = builder_id
            record["claimed_at"] = now
            record["ready_at"] = None
            record["failed_reason"] = None
            self._write_state(state)
            self.record_event(
                "cache_build_retried",
                artifact_key=artifact_key,
                builder_id=builder_id,
                previous_state=previous,
            )
            return BuildClaim.CLAIMED

    @staticmethod
    def _new_record(artifact_key: str, builder_id: str, now: str) -> dict:
        return {
            "artifact_key": artifact_key,
            "state": ArtifactState.BUILDING.value,
            "content_hash": "",
            "byte_size": -1,
            "pipeline_fingerprint": "",
            "build_id": str(uuid.uuid4()),
            "builder_id": builder_id,
            "claimed_at": now,
            "ready_at": None,
            "failed_reason": None,
        }

    def _is_stale(self, record: dict) -> bool:
        claimed_at = record.get("claimed_at")
        if not claimed_at:
            return True
        try:
            claimed = datetime.fromisoformat(claimed_at)
        except ValueError:
            return True
        age = (datetime.now(timezone.utc) - claimed).total_seconds()
        return age > self.build_timeout_seconds

    def publish(self, artifact: ContentArtifact) -> None:
        """Atomically transition a claimed build to READY.

        The payload is written first and the state flips second, so a READY
        record always points at a complete payload. Rejects an empty artifact.
        """
        if not artifact.chunks:
            raise ValueError("an artifact with no chunks cannot be published")
        with self._locked():
            state = self._read_state()
            record = state.get(artifact.artifact_key)
            if record is None or record.get("state") != ArtifactState.BUILDING.value:
                raise ArtifactNotClaimed(
                    f"artifact '{artifact.artifact_key}' is not claimed as building"
                )

            now = _now()
            payload = artifact.model_copy(
                update={
                    "state": ArtifactState.READY,
                    "build_id": record["build_id"],
                    "builder_id": record["builder_id"],
                    "claimed_at": record["claimed_at"],
                    "ready_at": now,
                }
            )
            self._write_payload(payload)
            record["state"] = ArtifactState.READY.value
            record["content_hash"] = artifact.content_hash
            record["byte_size"] = artifact.byte_size
            record["pipeline_fingerprint"] = artifact.pipeline_fingerprint
            record["ready_at"] = now
            record["failed_reason"] = None
            self._write_state(state)
        self.record_event(
            "artifact_built",
            artifact_key=artifact.artifact_key,
            pipeline_fingerprint=artifact.pipeline_fingerprint,
            byte_size=artifact.byte_size,
            chunk_count=artifact.chunk_count,
        )

    def mark_failed(self, artifact_key: str, reason: str) -> None:
        """Record a failed build so a later claim may retry it atomically."""
        with self._locked():
            state = self._read_state()
            record = state.get(artifact_key)
            if record is None or record.get("state") != ArtifactState.BUILDING.value:
                return
            record["state"] = ArtifactState.FAILED.value
            record["failed_reason"] = reason
            self._write_state(state)
        self.record_event("artifact_failed", artifact_key=artifact_key, reason=reason)

    def verify_reusable(
        self,
        artifact_key: str,
        *,
        content_hash: str,
        byte_size: int,
    ) -> ContentArtifact | None:
        """Re-verify a READY artifact against the current source bytes.

        Returns the payload only when the recorded length and SHA-256 still
        match the source file; a mismatch or a missing/empty payload marks the
        artifact ``corrupt`` and returns ``None``.
        """
        with self._locked():
            state = self._read_state()
            record = state.get(artifact_key)
            if record is None or record.get("state") != ArtifactState.READY.value:
                return None
            if record.get("content_hash") != content_hash or record.get("byte_size") != byte_size:
                self._mark_corrupt_locked(
                    state, record, artifact_key, "content length or hash mismatch on reuse"
                )
                return None
            payload = self._read_payload(artifact_key)
            if payload is None or payload.artifact_key != artifact_key:
                self._mark_corrupt_locked(
                    state, record, artifact_key, "artifact payload missing or mismatched"
                )
                return None
            if (
                payload.content_hash != content_hash
                or payload.byte_size != byte_size
                or payload.content_hash != record.get("content_hash")
                or payload.byte_size != record.get("byte_size")
                or payload.pipeline_fingerprint != record.get("pipeline_fingerprint")
            ):
                self._mark_corrupt_locked(
                    state, record, artifact_key, "artifact payload identity mismatch"
                )
                return None
            if payload.chunk_count == 0:
                self._mark_corrupt_locked(
                    state, record, artifact_key, "artifact published with no chunks"
                )
                return None
            return payload

    def _mark_corrupt_locked(
        self, state: dict, record: dict, artifact_key: str, reason: str
    ) -> None:
        record["state"] = ArtifactState.CORRUPT.value
        record["failed_reason"] = reason
        self._write_state(state)
        self.record_event("artifact_corrupt", artifact_key=artifact_key, reason=reason)

    def remove(self, artifact_key: str) -> bool:
        """Remove the artifact record and payload. Idempotent and audited."""
        with self._locked():
            state = self._read_state()
            record = state.pop(artifact_key, None)
            self._write_state(state)
            payload = self.payload_dir / f"{artifact_key}.json"
            removed = record is not None or payload.exists()
            if payload.exists():
                payload.unlink()
        if removed:
            self.record_event(
                "artifact_cleanup",
                artifact_key=artifact_key,
                state=(record or {}).get("state"),
                pipeline_fingerprint=(record or {}).get("pipeline_fingerprint"),
            )
        return removed

    def is_cleanup_eligible(self, artifact_key: str, active_refcount: int) -> bool:
        """True when no tenant references the artifact and its record still exists."""
        return active_refcount == 0 and self.record(artifact_key) is not None

    # ── build-or-reuse orchestration ───────────────────────────────────

    def ensure_ready(
        self,
        artifact_key: str,
        *,
        content_hash: str,
        byte_size: int,
        build: Callable[[], ContentArtifact],
        builder_id: str | None = None,
    ) -> tuple[ContentArtifact, str]:
        """Return a verified artifact and its cache outcome.

        ``build`` runs exactly once across concurrent callers with the same
        bytes: one claims the build, the others wait a bounded time and reuse
        the result. Outcomes are ``"build"``, ``"recovered"`` or ``"hit"``.
        """
        deadline = time.monotonic() + self.build_timeout_seconds
        while True:
            claim = self.claim_build(artifact_key, builder_id=builder_id)

            if claim in (BuildClaim.CLAIMED, BuildClaim.RECOVERED):
                outcome = "build" if claim is BuildClaim.CLAIMED else "recovered"
                try:
                    artifact = build()
                    if artifact.artifact_key != artifact_key:
                        raise ValueError(
                            "built artifact key does not match the claimed key"
                        )
                    self.publish(artifact)
                except Exception as error:  # noqa: BLE001 — recorded, not swallowed
                    self.mark_failed(artifact_key, str(error))
                    raise
                verified = self.verify_reusable(
                    artifact_key, content_hash=content_hash, byte_size=byte_size
                )
                if verified is None:
                    raise ArtifactBuildTimeout(
                        f"artifact '{artifact_key}' failed verification right after publish"
                    )
                return verified, outcome

            if claim is BuildClaim.READY:
                artifact = self.verify_reusable(
                    artifact_key, content_hash=content_hash, byte_size=byte_size
                )
                if artifact is not None:
                    self.record_event("cache_hit", artifact_key=artifact_key)
                    return artifact, "hit"
                # verify marked it corrupt → the loop reclaims and rebuilds.
                continue

            # BuildClaim.WAIT — another writer owns the build; poll its result.
            while time.monotonic() < deadline:
                time.sleep(self.poll_interval_seconds)
                artifact = self.verify_reusable(
                    artifact_key, content_hash=content_hash, byte_size=byte_size
                )
                if artifact is not None:
                    self.record_event("cache_hit", artifact_key=artifact_key)
                    return artifact, "hit"
                record = self.record(artifact_key)
                if record is not None and record.get("state") in {
                    ArtifactState.FAILED.value,
                    ArtifactState.CORRUPT.value,
                }:
                    break
            else:
                raise ArtifactBuildTimeout(
                    f"artifact '{artifact_key}' was not built within "
                    f"{self.build_timeout_seconds}s"
                )


__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactBuildTimeout",
    "ArtifactChunk",
    "ArtifactNotClaimed",
    "ArtifactRegistry",
    "ArtifactState",
    "BuildClaim",
    "CACHE_ROOT_ENV",
    "ContentArtifact",
    "DEFAULT_BUILD_TIMEOUT_SECONDS",
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "default_cache_root",
]
