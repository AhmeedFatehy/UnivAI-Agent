"""Tenant-owned grants over shared immutable artifacts.

The immutable :class:`~cache.artifact_registry.ContentArtifact` is shared, but
access is not: a tenant can only read a document's chunks through an active
:class:`DocumentGrant`. This module owns that mapping and the reference count.

* :meth:`GrantStore.grant` is idempotent per ``(user_id, document_id)`` — a
  re-ingest never duplicates access, and two tenants uploading identical bytes
  get two independent grants on one artifact.
* retrieval and citation resolution must call :meth:`GrantStore.require_grant`
  first; an absent grant is an explicit :class:`GrantDenied`, never a silent
  empty result.
* :meth:`GrantStore.revoke` removes only the caller's grant. When it is the
  *last* active grant on an artifact, :func:`revoke_document` reports that the
  artifact is eligible for cleanup; the grant store never deletes data itself —
  the caller asks the registry to remove the artifact, so cleanup is auditable
  and retry-safe.
* nothing here records filenames, query history, upload order, or any other
  tenant's grants.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from cache.artifact_registry import ArtifactRegistry, _FileLock

GRANT_SCHEMA = "univai.agent.document_grant"
GRANT_SCHEMA_VERSION = "1.0.0"


class GrantState(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


class DocumentGrant(BaseModel):
    """One tenant's authorisation to read one artifact's chunks."""

    schema_name: str = GRANT_SCHEMA
    schema_version: str = GRANT_SCHEMA_VERSION
    grant_id: str = Field(min_length=1)
    artifact_key: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    state: GrantState = GrantState.ACTIVE
    granted_at: str


class GrantDenied(PermissionError):
    """Raised when a tenant lacks an active grant for a document."""


class RevokeReport(BaseModel):
    """Outcome of removing one tenant's grant (and possibly the artifact)."""

    revoked: bool
    artifact_key: str | None = None
    last_reference: bool = False
    artifact_removed: bool = False
    reason: str | None = None


class GrantStore:
    """File-backed store of tenant grants, serialized under an exclusive lock."""

    def __init__(self, root: Path, *, lock_timeout_seconds: float = 5.0):
        self.root = Path(root).expanduser().resolve()
        self.grants_file = self.root / "grants.json"
        self.lock_file = self.root / ".grants.lock"
        self.lock_timeout_seconds = lock_timeout_seconds
        self._thread_lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── storage ────────────────────────────────────────────────────────

    def _read(self) -> dict:
        if not self.grants_file.exists():
            return {"grants": []}
        return json.loads(self.grants_file.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        temporary = self.grants_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.grants_file)

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            lock = _FileLock(self.lock_file, self.lock_timeout_seconds)
            lock.acquire()
            try:
                yield
            finally:
                lock.release()

    # ── grants ─────────────────────────────────────────────────────────

    def grant(
        self,
        *,
        artifact_key: str,
        user_id: str,
        collection_id: str,
        document_id: str,
        source_filename: str,
        document_type: str,
        book_title: str,
    ) -> DocumentGrant:
        """Create or replace the active grant for ``(user_id, document_id)``.

        Re-ingesting against the same artifact is idempotent. A pipeline or
        content change revokes the old mapping and creates a new active grant
        atomically, so authorization never remains bound to stale vectors.
        """
        if not user_id.strip():
            raise ValueError("user_id is required")
        if not document_id.strip():
            raise ValueError("document_id is required")
        with self._locked():
            data = self._read()
            grants = data["grants"]
            existing = next(
                (
                    grant
                    for grant in grants
                    if grant["user_id"] == user_id
                    and grant["document_id"] == document_id
                    and grant["state"] == GrantState.ACTIVE.value
                ),
                None,
            )
            if existing is not None:
                if existing["artifact_key"] == artifact_key:
                    return DocumentGrant.model_validate(existing)
                existing["state"] = GrantState.REVOKED.value

            grant = DocumentGrant(
                grant_id=str(uuid.uuid4()),
                artifact_key=artifact_key,
                user_id=user_id,
                collection_id=collection_id,
                document_id=document_id,
                source_filename=source_filename,
                document_type=document_type,
                book_title=book_title,
                state=GrantState.ACTIVE,
                granted_at=datetime.now(timezone.utc).isoformat(),
            )
            grants.append(grant.model_dump(mode="json"))
            self._write(data)
            return grant

    def is_granted(self, user_id: str, document_id: str) -> bool:
        try:
            self.require_grant(user_id, document_id)
            return True
        except GrantDenied:
            return False

    def require_grant(self, user_id: str, document_id: str) -> DocumentGrant:
        """Return the active grant or raise :class:`GrantDenied`."""
        with self._locked():
            grants = self._read()["grants"]
            grant = next(
                (
                    item
                    for item in grants
                    if item["user_id"] == user_id
                    and item["document_id"] == document_id
                    and item["state"] == GrantState.ACTIVE.value
                ),
                None,
            )
            if grant is None:
                raise GrantDenied(
                    f"no active grant for document '{document_id}' owned by '{user_id}'"
                )
            return DocumentGrant.model_validate(grant)

    def active_grants(self, artifact_key: str) -> list[DocumentGrant]:
        """Every active grant on one artifact."""
        with self._locked():
            return [
                DocumentGrant.model_validate(item)
                for item in self._read()["grants"]
                if item["artifact_key"] == artifact_key
                and item["state"] == GrantState.ACTIVE.value
            ]

    def active_refcount(self, artifact_key: str) -> int:
        """Number of tenants still authorised on one artifact."""
        return len(self.active_grants(artifact_key))

    def authorized_document_ids(
        self,
        user_id: str,
        *,
        collection_id: str | None = None,
        book_titles: list[str] | None = None,
    ) -> list[str]:
        """Active document IDs the tenant may search within an optional scope.

        Retrieval uses this to turn an otherwise broad tenant query into an
        explicit allow-list before Qdrant is called. The list contains only
        the requesting tenant's document IDs and reveals no other grant,
        filename, uploader, or cache outcome.
        """
        wanted_titles = set(book_titles or [])
        with self._locked():
            document_ids = {
                item["document_id"]
                for item in self._read()["grants"]
                if item["user_id"] == user_id
                and item["state"] == GrantState.ACTIVE.value
                and (collection_id is None or item["collection_id"] == collection_id)
                and (not wanted_titles or item["book_title"] in wanted_titles)
            }
        return sorted(document_ids)

    def unreferenced_artifact_keys(self) -> list[str]:
        """Previously granted artifacts that no active grant still references."""
        with self._locked():
            records = self._read()["grants"]
            previously_granted = {item["artifact_key"] for item in records}
            active = {
                item["artifact_key"]
                for item in records
                if item["state"] == GrantState.ACTIVE.value
            }
        return sorted(previously_granted - active)

    def revoke(self, user_id: str, document_id: str) -> DocumentGrant | None:
        """Revoke the tenant's grant, keeping it as a revoked audit record.

        Returns the revoked grant, or ``None`` when the tenant had none.
        """
        with self._locked():
            data = self._read()
            grants = data["grants"]
            revoked: DocumentGrant | None = None
            kept: list[dict] = []
            for grant in grants:
                if (
                    grant["user_id"] == user_id
                    and grant["document_id"] == document_id
                    and grant["state"] == GrantState.ACTIVE.value
                ):
                    grant = dict(grant)
                    grant["state"] = GrantState.REVOKED.value
                    revoked = DocumentGrant.model_validate(grant)
                kept.append(grant)
            if revoked is None:
                return None
            data["grants"] = kept
            self._write(data)
            return revoked


def cleanup_unreferenced_artifacts(
    grants: GrantStore, registry: ArtifactRegistry
) -> list[str]:
    """Retry auditable cleanup for every artifact whose refcount reached zero."""
    removed: list[str] = []
    for artifact_key in grants.unreferenced_artifact_keys():
        if registry.remove(artifact_key):
            removed.append(artifact_key)
    return removed


def revoke_document(
    grants: GrantStore,
    registry: ArtifactRegistry,
    *,
    user_id: str,
    document_id: str,
) -> RevokeReport:
    """Revoke one tenant's grant and clean up the shared artifact when it was the last.

    Removing tenant A's book never touches tenant B's access: the shared
    artifact is only removed once no active grant references it, and the removal
    goes through the audited, idempotent :meth:`ArtifactRegistry.remove`.
    """
    cleanup_unreferenced_artifacts(grants, registry)
    grant = grants.revoke(user_id, document_id)
    if grant is None:
        return RevokeReport(
            revoked=False,
            reason=f"no active grant for document '{document_id}' owned by '{user_id}'",
        )
    last_reference = grants.active_refcount(grant.artifact_key) == 0
    removed = False
    if last_reference:
        removed = registry.remove(grant.artifact_key)
    return RevokeReport(
        revoked=True,
        artifact_key=grant.artifact_key,
        last_reference=last_reference,
        artifact_removed=removed,
    )


__all__ = [
    "GRANT_SCHEMA",
    "GRANT_SCHEMA_VERSION",
    "DocumentGrant",
    "GrantDenied",
    "GrantState",
    "GrantStore",
    "RevokeReport",
    "cleanup_unreferenced_artifacts",
    "revoke_document",
]
