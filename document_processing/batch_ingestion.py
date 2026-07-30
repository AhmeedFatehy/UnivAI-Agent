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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from document_processing.chunking import chunk_documents
from document_processing.loaders import load_document
from document_processing.metadata import (
    SCHEMA_VERSION,
    ChunkMetadata,
    apply_chunk_metadata,
    book_title_from,
    stable_document_id,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3

#: Failures that will never succeed on a retry.
PERMANENT_ERRORS: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
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
    """The three side-effecting steps of ingestion, injectable for tests.

    Defaults are the real pipeline: LangChain loaders, the project's chunkers
    and Qdrant indexing.
    """

    load: Callable[[str], list] = field(default=None)  # type: ignore[assignment]
    chunk: Callable[[list, str], list] = field(default=None)  # type: ignore[assignment]
    index: Callable[..., dict] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.load is None:
            self.load = load_document
        if self.chunk is None:
            self.chunk = chunk_documents
        if self.index is None:
            self.index = _default_index

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
) -> tuple[DocumentIngestionResult, list[ChunkMetadata]]:
    """Ingest one book, retrying transient failures.

    Raises :class:`IngestionError` if every attempt fails; the caller decides
    whether that ends the batch (it does not).
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
            records = apply_chunk_metadata(
                chunks,
                collection_id=collection_id,
                document_id=document_id,
                user_id=user_id,
                book_title=book_title,
                source_filename=filename,
                document_type=document_type,
                ingested_at=ingested_at,
            )

            indexed = backend.index(
                chunks=chunks,
                user_id=user_id,
                document_id=document_id,
                source_filename=filename,
                document_type=document_type,
                collection_name=collection_name,
            )
            chunks_indexed = int(indexed.get("chunks_indexed", len(chunks)))
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

        except BaseException as error:  # noqa: BLE001 — wrapped and re-raised below
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
    "BatchIngestionReport",
    "DocumentIngestionFailure",
    "DocumentIngestionResult",
    "IngestionBackend",
    "IngestionError",
    "ingest_collection",
    "ingest_document",
]
