"""Multi-book ingestion: index a whole collection, one book at a time.

A programme is built from several books, and a single unreadable file must not
cost the other books their index. So this module ingests each document
independently and reports per-document outcomes:

* one book failing leaves the rest indexed — :attr:`BatchIngestionReport.failed`
  records why, and ``partial_success`` says the batch was mixed;
* transient failures (a Qdrant timeout, a dropped connection) are retried up to
  ``max_attempts``; permanent ones (missing file, unsupported format) are not,
  because retrying them only wastes time.

Loading, chunking and indexing are injected through :class:`IngestionBackend`
so tests exercise the real batching, metadata and retry logic without a running
Qdrant or an embedding model download.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from cache.artifact_registry import (
    ArtifactChunk,
    ArtifactRegistry,
    ArtifactState,
    ContentArtifact,
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    default_cache_root,
)
from cache.authorization import (
    DocumentGrant,
    GrantStore,
    RevokeReport,
    cleanup_unreferenced_artifacts,
    revoke_document,
)
from cache.content_identity import ContentIdentity, default_pipeline_components
from document_processing.chunking import chunk_documents
from document_processing.loaders import load_document
from document_processing.metadata import (
    SCHEMA_VERSION,
    ChunkMetadata,
    apply_chunk_metadata,
    book_title_from,
    normalise_page,
    rebuild_chunks,
    stable_document_id,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3

#: Failures that will never succeed on a retry.
#: ImportError covers ModuleNotFoundError — a loader whose optional dependency is
#: missing fails identically every time, so retrying it just triples the wait
#: before the same error is reported.
PERMANENT_ERRORS: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
    ImportError,
    ValueError,
    TypeError,
    KeyError,
)


class DocumentIngestionResult(BaseModel):
    """One book that made it into the index."""

    path: str
    document_id: str
    book_title: str
    document_type: str
    chunks_indexed: int = Field(ge=1)
    pages: int = Field(ge=0)
    attempts: int = Field(ge=1)
    collection_name: str | None = None


class DocumentIngestionFailure(BaseModel):
    """One book that did not make it, and why."""

    path: str
    error_type: str
    error: str
    attempts: int = Field(ge=1)
    retryable: bool


class BatchIngestionReport(BaseModel):
    """Outcome of ingesting one collection of books."""

    schema_version: str = SCHEMA_VERSION
    collection_id: str
    user_id: str
    collection_name: str | None = None
    requested: int = Field(ge=0)
    succeeded: list[DocumentIngestionResult] = Field(default_factory=list)
    failed: list[DocumentIngestionFailure] = Field(default_factory=list)
    started_at: str
    finished_at: str

    @property
    def documents_indexed(self) -> int:
        return len(self.succeeded)

    @property
    def chunks_indexed(self) -> int:
        return sum(item.chunks_indexed for item in self.succeeded)

    @property
    def complete_success(self) -> bool:
        return bool(self.succeeded) and not self.failed

    @property
    def partial_success(self) -> bool:
        return bool(self.succeeded) and bool(self.failed)

    @property
    def total_failure(self) -> bool:
        return not self.succeeded and bool(self.failed)

    def summary(self) -> str:
        return (
            f"collection '{self.collection_id}': {self.documents_indexed}/{self.requested} "
            f"books indexed, {self.chunks_indexed} chunks, {len(self.failed)} failed"
        )


@dataclass
class IngestionBackend:
    """The side-effecting steps of ingestion, injectable for tests.

    Defaults are the real pipeline: LangChain loaders, the project's chunkers
    and Qdrant indexing.
    """

    load: Callable[[str], list] = field(default=None)  # type: ignore[assignment]
    chunk: Callable[[list, str], list] = field(default=None)  # type: ignore[assignment]
    index: Callable[..., dict] = field(default=None)  # type: ignore[assignment]
    purge: Callable[..., int] = field(default=None)  # type: ignore[assignment]
    embed: Callable[[list[str]], tuple] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.load is None:
            self.load = load_document
        if self.chunk is None:
            self.chunk = chunk_documents
        if self.index is None:
            self.index = _default_index
        if self.purge is None:
            self.purge = _default_purge
        if self.embed is None:
            self.embed = _default_embed

    def is_retryable(self, error: BaseException) -> bool:
        return not isinstance(error, PERMANENT_ERRORS)


class IngestionError(RuntimeError):
    """A book that could not be ingested, with the attempts actually spent."""

    def __init__(self, path: str, cause: BaseException, attempts: int, retryable: bool):
        super().__init__(f"{path}: {cause}")
        self.path = path
        self.cause = cause
        self.attempts = attempts
        self.retryable = retryable


def _default_index(**kwargs) -> dict:
    """Import Qdrant indexing lazily so a fake backend needs no vector store."""
    from vector_store.indexing import index_chunks

    return index_chunks(**kwargs)


def _default_embed(texts: list[str]) -> tuple[list, list]:
    """Embed text with the production dense and sparse models.

    Used only when the content-addressed cache builds a fresh artifact, so the
    vectors can be stored once and reused for every tenant with identical bytes.
    """
    from vector_store.qdrant_client import get_dense_embedder, get_sparse_embedder

    dense = list(get_dense_embedder().embed(texts))
    sparse = list(get_sparse_embedder().embed(texts))
    return dense, sparse


def _as_list(vector) -> list[float] | None:
    """Coerce an embedding vector to a plain float list, or ``None``."""
    if vector is None:
        return None
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


def _usable_embeddings(
    artifact: ContentArtifact,
) -> list[tuple[list[float] | None, list[int] | None, list[float] | None]] | None:
    """Per-chunk embedding triples from a cached artifact, or ``None``.

    Returns ``None`` unless *every* chunk carries a dense vector and sparse
    indices, so the indexer either reuses a complete, verified vector set or
    falls back to embedding everything fresh. A partial vector set is never
    trusted.
    """
    embeddings = artifact.embeddings()
    if not embeddings or len(embeddings) != artifact.chunk_count:
        return None
    if any(
        dense is None or indices is None for dense, indices, _values in embeddings
    ):
        return None
    return embeddings


def _default_purge(
    *,
    user_id: str,
    document_id: str,
    collection_name: str | None,
    preserve_ingestion_id: str,
) -> int:
    """Drop superseded chunks only after their replacement is safely indexed.

    A collection that does not exist yet has nothing to purge, which is the
    normal first-ingest case rather than an error.
    """
    from vector_store.collection_manager import delete_document_versions
    from vector_store.qdrant_client import get_qdrant_client
    from config import COLLECTION_NAME

    name = collection_name or COLLECTION_NAME
    if not get_qdrant_client().collection_exists(collection_name=name):
        return 0
    return delete_document_versions(
        user_id=user_id,
        document_id=document_id,
        preserve_ingestion_id=preserve_ingestion_id,
        collection_name=name,
    )


# ── Content-addressed RAG cache ───────────────────────────────────────


@dataclass
class ArtifactCache:
    """Content-addressed cache + tenant authorization facade for ingestion.

    Wraps an :class:`~cache.artifact_registry.ArtifactRegistry` and a
    :class:`~cache.authorization.GrantStore` so identical bytes are parsed,
    chunked and embedded once per pipeline fingerprint, and every tenant keeps
    an independent, revocable grant on the shared immutable artifact.
    """

    registry: ArtifactRegistry
    grants: GrantStore

    @classmethod
    def at(cls, root: Path, **registry_kwargs) -> "ArtifactCache":
        """A cache backed by a directory under ``root``."""
        root = Path(root)
        lock_timeout = registry_kwargs.get(
            "lock_timeout_seconds", DEFAULT_LOCK_TIMEOUT_SECONDS
        )
        return cls(
            registry=ArtifactRegistry(root, **registry_kwargs),
            grants=GrantStore(root, lock_timeout_seconds=lock_timeout),
        )

    def require_grant(self, user_id: str, document_id: str) -> DocumentGrant:
        return self.grants.require_grant(user_id, document_id)

    def is_granted(self, user_id: str, document_id: str) -> bool:
        return self.grants.is_granted(user_id, document_id)

    def revoke(self, user_id: str, document_id: str) -> RevokeReport:
        return revoke_document(
            self.grants, self.registry, user_id=user_id, document_id=document_id
        )


def default_artifact_cache() -> ArtifactCache:
    """A cache at the configured root — for callers that do not inject one."""
    return ArtifactCache.at(default_cache_root())


def _prepare_ingestion(
    source: Path,
    filename: str,
    *,
    backend: IngestionBackend,
    cache: ArtifactCache | None,
) -> tuple[list, str, str, dict | None]:
    """Return ``(chunks, book_title, document_type, cache_outcome)``.

    With a cache, identical bytes are loaded, chunked and embedded once per
    pipeline fingerprint and every tenant reuses the stored artifact; without
    one the classic load/chunk path runs every time. The outcome carries only
    the caller's own artifact key, fingerprint and reusable embeddings — never
    another tenant's prior upload.
    """
    if cache is None:
        documents = backend.load(str(source))
        if not documents:
            raise ValueError(f"'{filename}' loaded as an empty document")
        document_type = (
            documents[0].metadata.get("document_type")
            or source.suffix.lstrip(".")
            or "unknown"
        )
        book_title = book_title_from(documents, source)
        chunks = backend.chunk(documents, document_type)
        return chunks, book_title, document_type, None

    identity = ContentIdentity.from_file(source, pipeline=default_pipeline_components())

    def build() -> ContentArtifact:
        documents = backend.load(str(source))
        if not documents:
            raise ValueError(f"'{filename}' loaded as an empty document")
        document_type = (
            documents[0].metadata.get("document_type")
            or source.suffix.lstrip(".")
            or "unknown"
        )
        book_title = book_title_from(documents, source)
        chunks = backend.chunk(documents, document_type)
        texts = [getattr(chunk, "page_content", "") or "" for chunk in chunks]
        dense, sparse = backend.embed(texts)

        artifact_chunks: list[ArtifactChunk] = []
        for index, chunk in enumerate(chunks):
            metadata = getattr(chunk, "metadata", None) or {}
            sparse_vec = sparse[index] if sparse and index < len(sparse) else None
            artifact_chunks.append(
                ArtifactChunk(
                    text=texts[index],
                    page=normalise_page(metadata.get("page", metadata.get("page_number"))),
                    dense_vector=(
                        _as_list(dense[index]) if dense and index < len(dense) else None
                    ),
                    sparse_indices=(
                        sparse_vec.indices.tolist()
                        if sparse_vec is not None and hasattr(sparse_vec, "indices")
                        else None
                    ),
                    sparse_values=(
                        sparse_vec.values.tolist()
                        if sparse_vec is not None and hasattr(sparse_vec, "values")
                        else None
                    ),
                )
            )

        return ContentArtifact(
            artifact_key=identity.artifact_key,
            content_hash=identity.content_hash,
            byte_size=identity.byte_size,
            pipeline_fingerprint=identity.pipeline_fingerprint,
            document_type=document_type,
            book_title=book_title,
            chunks=artifact_chunks,
            state=ArtifactState.BUILDING,
        )

    artifact, outcome = cache.registry.ensure_ready(
        identity.artifact_key,
        content_hash=identity.content_hash,
        byte_size=identity.byte_size,
        build=build,
    )
    chunks = rebuild_chunks(artifact.chunk_texts, pages=artifact.pages)
    return (
        chunks,
        artifact.book_title,
        artifact.document_type,
        {
            "outcome": outcome,
            "artifact_key": artifact.artifact_key,
            "content_hash": artifact.content_hash,
            "pipeline_fingerprint": artifact.pipeline_fingerprint,
            "embeddings": _usable_embeddings(artifact),
        },
    )


def ingest_document(
    path: str | Path,
    *,
    collection_id: str,
    user_id: str,
    backend: IngestionBackend | None = None,
    collection_name: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay: float = 0.0,
    ingested_at: str | None = None,
    cache: ArtifactCache | None = None,
) -> tuple[DocumentIngestionResult, list[ChunkMetadata]]:
    """Ingest one book, retrying transient failures.

    Raises :class:`IngestionError` if every attempt fails; the caller decides
    whether that ends the batch (it does not).

    When ``cache`` is given, identical bytes are parsed/chunked/embedded once
    and every tenant gets an independent grant on the shared artifact. The
    outcome (build vs hit) is recorded only in private telemetry, never in the
    report a tenant sees.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    backend = backend or IngestionBackend()
    source = Path(path)
    filename = source.name
    document_id = stable_document_id(collection_id, filename)

    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            ingestion_id = str(uuid.uuid4())
            chunks, book_title, document_type, cache_outcome = _prepare_ingestion(
                source, filename, backend=backend, cache=cache
            )
            records = apply_chunk_metadata(
                chunks,
                collection_id=collection_id,
                document_id=document_id,
                user_id=user_id,
                book_title=book_title,
                source_filename=filename,
                document_type=document_type,
                ingestion_id=ingestion_id,
                ingested_at=ingested_at,
            )

            index_kwargs = {
                "chunks": chunks,
                "user_id": user_id,
                "document_id": document_id,
                "source_filename": filename,
                "document_type": document_type,
                "collection_name": collection_name,
            }
            if cache_outcome is not None:
                index_kwargs["artifact_key"] = cache_outcome["artifact_key"]
                index_kwargs["content_hash"] = cache_outcome["content_hash"]
                if cache_outcome["embeddings"] is not None:
                    index_kwargs["embeddings"] = cache_outcome["embeddings"]

            indexed = backend.index(**index_kwargs)
            chunks_indexed = int(indexed.get("chunks_indexed", len(chunks)))

            # The grant is what authorizes this tenant to retrieve the artifact's
            # chunks. It is created only after the chunks are actually indexed,
            # and it is idempotent for a re-ingest by the same tenant/artifact.
            # A changed pipeline atomically replaces the grant, then cleans up
            # the old artifact only when no other tenant still references it.
            if cache is not None:
                cache.grants.grant(
                    artifact_key=cache_outcome["artifact_key"],
                    user_id=user_id,
                    collection_id=collection_id,
                    document_id=document_id,
                    source_filename=filename,
                    document_type=document_type,
                    book_title=book_title,
                )
                cleanup_unreferenced_artifacts(cache.grants, cache.registry)

            # Index first, then remove every older generation. Purging before
            # upload makes a transient embedding/Qdrant failure erase the last
            # known-good copy. A retry also cleans up a partial earlier upload.
            replaced = backend.purge(
                user_id=user_id,
                document_id=document_id,
                collection_name=collection_name,
                preserve_ingestion_id=ingestion_id,
            )
            if replaced:
                logger.info("Replaced %d superseded chunk(s) for %s", replaced, filename)

            pages = len({record.page for record in records if record.page is not None})

            result = DocumentIngestionResult(
                path=str(source),
                document_id=document_id,
                book_title=book_title,
                document_type=document_type,
                chunks_indexed=chunks_indexed,
                pages=pages,
                attempts=attempt,
                collection_name=indexed.get("collection_name", collection_name),
            )
            return result, records

        except Exception as error:  # noqa: BLE001 — wrapped and re-raised below
            last_error = error
            retryable = backend.is_retryable(error)
            if not retryable or attempt == max_attempts:
                raise IngestionError(str(source), error, attempt, retryable) from error
            logger.warning(
                "Ingest attempt %d/%d failed for %s (%s); retrying",
                attempt,
                max_attempts,
                filename,
                error,
            )
            if retry_delay:
                time.sleep(retry_delay)

    raise IngestionError(str(source), last_error or RuntimeError("unknown"), max_attempts, True)


def ingest_collection(
    paths: Iterable[str | Path],
    *,
    collection_id: str,
    user_id: str,
    backend: IngestionBackend | None = None,
    collection_name: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay: float = 0.0,
    ingested_at: str | None = None,
    cache: ArtifactCache | None = None,
) -> tuple[BatchIngestionReport, dict[str, list[ChunkMetadata]]]:
    """Ingest several books into one collection, tolerating per-book failure.

    Returns the report plus the chunk records keyed by ``document_id`` — the
    records are what a caller needs to resolve a citation without another round
    trip to the vector store.
    """
    if not collection_id or not collection_id.strip():
        raise ValueError("collection_id is required")
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")

    backend = backend or IngestionBackend()
    ordered: Sequence[str | Path] = list(paths)
    started_at = datetime.now(timezone.utc).isoformat()

    succeeded: list[DocumentIngestionResult] = []
    failed: list[DocumentIngestionFailure] = []
    records_by_document: dict[str, list[ChunkMetadata]] = {}

    for path in ordered:
        try:
            result, records = ingest_document(
                path,
                collection_id=collection_id,
                user_id=user_id,
                backend=backend,
                collection_name=collection_name,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                ingested_at=ingested_at,
                cache=cache,
            )
        except IngestionError as error:
            cause = error.cause
            failed.append(
                DocumentIngestionFailure(
                    path=str(path),
                    error_type=type(cause).__name__,
                    error=str(cause) or repr(cause),
                    attempts=error.attempts,
                    retryable=error.retryable,
                )
            )
            logger.error(
                "Ingest failed for %s after %d attempt(s): %s", path, error.attempts, cause
            )
            continue

        succeeded.append(result)
        records_by_document[result.document_id] = records

    report = BatchIngestionReport(
        collection_id=collection_id,
        user_id=user_id,
        collection_name=collection_name,
        requested=len(ordered),
        succeeded=succeeded,
        failed=failed,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info("%s", report.summary())
    return report, records_by_document


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "PERMANENT_ERRORS",
    "ArtifactCache",
    "BatchIngestionReport",
    "DocumentIngestionFailure",
    "DocumentIngestionResult",
    "IngestionBackend",
    "IngestionError",
    "default_artifact_cache",
    "ingest_collection",
    "ingest_document",
]
