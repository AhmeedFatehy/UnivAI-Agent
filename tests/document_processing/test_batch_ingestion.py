"""Multi-book ingestion: identity on every chunk, partial success, bounded retry.

No Qdrant and no embedding model — the indexer is a fake, but the loading,
chunking, metadata and batching logic under test are the real ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from document_processing.batch_ingestion import (
    IngestionBackend,
    ingest_collection,
    ingest_document,
)
from document_processing.chunking import chunk_documents
from document_processing.metadata import (
    ChunkMetadata,
    assign_sections,
    book_title_from,
    citation_from_payload,
    headings_in,
    normalise_page,
    stable_document_id,
)
from tests.conftest import COLLECTION_ID, USER_ID, markdown_loader


# ── Identity on every chunk ───────────────────────────────────────────


def test_three_books_index_with_full_identity(book_paths, fake_index):
    report, records = ingest_collection(
        book_paths, collection_id=COLLECTION_ID, user_id=USER_ID, backend=fake_index.backend()
    )

    assert report.complete_success
    assert report.documents_indexed == 3
    assert report.requested == 3
    assert report.chunks_indexed == len(fake_index.rows)

    for chunk_records in records.values():
        for record in chunk_records:
            assert isinstance(record, ChunkMetadata)
            assert record.collection_id == COLLECTION_ID
            assert record.user_id == USER_ID
            assert record.document_id
            assert record.book_title
            assert record.page is not None and record.page >= 1
            assert record.section
            assert record.chunk_index >= 0
            assert record.total_chunks >= 1


def test_book_titles_come_from_the_documents_not_the_filenames(book_paths, fake_index):
    report, _ = ingest_collection(
        book_paths, collection_id=COLLECTION_ID, user_id=USER_ID, backend=fake_index.backend()
    )
    titles = {result.book_title for result in report.succeeded}
    assert titles == {
        "Foundations of Algorithms",
        "Database Systems",
        "Machine Learning Basics",
    }


def test_documents_stay_source_isolated(indexed_books):
    """Each book keeps its own document id, and no chunk is shared between them."""
    index, records = indexed_books

    by_document: dict[str, set[str]] = {}
    for record in records:
        by_document.setdefault(record.document_id, set()).add(record.book_title)

    assert len(by_document) == 3
    for titles in by_document.values():
        assert len(titles) == 1, "a document id must belong to exactly one book"

    chunk_keys = [(row["document_id"], row["chunk_index"]) for row in index.rows]
    assert len(chunk_keys) == len(set(chunk_keys))


def test_document_ids_are_stable_across_reingestion(book_paths, fake_index):
    first, _ = ingest_collection(
        book_paths, collection_id=COLLECTION_ID, user_id=USER_ID, backend=fake_index.backend()
    )
    second, _ = ingest_collection(
        book_paths, collection_id=COLLECTION_ID, user_id=USER_ID, backend=fake_index.backend()
    )

    assert [item.document_id for item in first.succeeded] == [
        item.document_id for item in second.succeeded
    ]
    assert first.succeeded[0].document_id == stable_document_id(
        COLLECTION_ID, Path(first.succeeded[0].path).name
    )


def test_reingesting_replaces_chunks_instead_of_duplicating_them(book_paths, fake_index):
    """A stable document id is only useful if the old copy actually goes away."""
    ingest_collection(
        book_paths, collection_id=COLLECTION_ID, user_id=USER_ID, backend=fake_index.backend()
    )
    after_first = len(fake_index.rows)

    ingest_collection(
        book_paths, collection_id=COLLECTION_ID, user_id=USER_ID, backend=fake_index.backend()
    )

    assert len(fake_index.rows) == after_first, "re-ingesting must replace, not append"
    keys = [(row["document_id"], row["chunk_index"]) for row in fake_index.rows]
    assert len(keys) == len(set(keys)), "no chunk may appear twice"


def test_reingestion_does_not_disturb_another_users_copy(book_paths, fake_index):
    ingest_collection(
        book_paths[:1], collection_id=COLLECTION_ID, user_id="student-a",
        backend=fake_index.backend(),
    )
    ingest_collection(
        book_paths[:1], collection_id=COLLECTION_ID, user_id="student-b",
        backend=fake_index.backend(),
    )
    ingest_collection(
        book_paths[:1], collection_id=COLLECTION_ID, user_id="student-a",
        backend=fake_index.backend(),
    )

    owners = {row["user_id"] for row in fake_index.rows}
    assert owners == {"student-a", "student-b"}
    per_owner = {
        owner: len([row for row in fake_index.rows if row["user_id"] == owner])
        for owner in owners
    }
    assert per_owner["student-a"] == per_owner["student-b"]


def test_failed_reingestion_keeps_the_last_good_copy(book_paths, fake_index):
    ingest_collection(
        book_paths[:1],
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        backend=fake_index.backend(),
    )
    previous = list(fake_index.rows)

    def fail_index(**kwargs):
        raise ConnectionError("replacement upload failed")

    backend = fake_index.backend()
    backend.index = fail_index
    report, _ = ingest_collection(
        book_paths[:1],
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        backend=backend,
        max_attempts=1,
    )

    assert report.total_failure
    assert fake_index.rows == previous


def test_batch_does_not_swallow_process_cancellation(book_paths, fake_index):
    def cancel(**kwargs):
        raise KeyboardInterrupt()

    backend = fake_index.backend()
    backend.index = cancel

    with pytest.raises(KeyboardInterrupt):
        ingest_collection(
            book_paths[:1],
            collection_id=COLLECTION_ID,
            user_id=USER_ID,
            backend=backend,
        )


def test_the_same_book_in_two_collections_gets_two_identities(book_paths, fake_index):
    one, _ = ingest_collection(
        book_paths[:1], collection_id="programme-a", user_id=USER_ID, backend=fake_index.backend()
    )
    two, _ = ingest_collection(
        book_paths[:1], collection_id="programme-b", user_id=USER_ID, backend=fake_index.backend()
    )
    assert one.succeeded[0].document_id != two.succeeded[0].document_id


def test_indexed_chunks_round_trip_into_citations(indexed_books):
    index, _ = indexed_books
    for row in index.rows:
        citation = citation_from_payload(row)
        assert citation is not None
        assert citation.collection_id == COLLECTION_ID
        assert citation.book_title
        assert citation.page >= 1
        assert citation.section
        assert citation.label()


def test_chunks_without_identity_are_not_citable():
    """A pre-schema chunk must not be dressed up as a citation."""
    assert citation_from_payload({"content": "text", "chunk_index": 0}) is None


# ── Partial success ───────────────────────────────────────────────────


def test_one_missing_book_does_not_cost_the_others(book_paths, fake_index, tmp_path):
    missing = tmp_path / "not_a_book.md"
    paths = [book_paths[0], missing, book_paths[1]]

    report, records = ingest_collection(
        paths, collection_id=COLLECTION_ID, user_id=USER_ID, backend=fake_index.backend()
    )

    assert report.partial_success
    assert report.documents_indexed == 2
    assert len(records) == 2
    assert len(report.failed) == 1
    failure = report.failed[0]
    assert failure.path == str(missing)
    assert failure.error_type == "FileNotFoundError"
    assert failure.retryable is False


def test_every_book_failing_is_reported_as_total_failure(tmp_path, fake_index):
    report, records = ingest_collection(
        [tmp_path / "a.md", tmp_path / "b.md"],
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        backend=fake_index.backend(),
    )
    assert report.total_failure
    assert not report.complete_success
    assert records == {}
    assert len(report.failed) == 2


def test_an_empty_book_is_a_failure_not_an_empty_index(tmp_path, fake_index):
    empty = tmp_path / "empty.md"
    empty.write_text("   \n\n  ", encoding="utf-8")

    report, _ = ingest_collection(
        [empty], collection_id=COLLECTION_ID, user_id=USER_ID, backend=fake_index.backend()
    )
    assert report.total_failure
    assert fake_index.rows == []


# ── Bounded retry ─────────────────────────────────────────────────────


class FlakyIndexer:
    """Fails with a transient error for the first ``failures`` calls."""

    def __init__(self, failures: int, inner):
        self.failures = failures
        self.inner = inner
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("qdrant connection reset")
        return self.inner(**kwargs)


def test_a_transient_failure_is_retried_and_succeeds(book_paths, fake_index):
    flaky = FlakyIndexer(1, fake_index.index)
    backend = IngestionBackend(
        load=markdown_loader, chunk=chunk_documents, index=flaky, purge=fake_index.purge
    )

    result, records = ingest_document(
        book_paths[0],
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        backend=backend,
        max_attempts=3,
    )

    assert flaky.calls == 2
    assert result.attempts == 2
    assert result.chunks_indexed == len(records)


def test_retries_stop_at_max_attempts(book_paths, fake_index):
    flaky = FlakyIndexer(99, fake_index.index)
    backend = IngestionBackend(
        load=markdown_loader, chunk=chunk_documents, index=flaky, purge=fake_index.purge
    )

    report, _ = ingest_collection(
        [book_paths[0]],
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        backend=backend,
        max_attempts=3,
    )

    assert flaky.calls == 3, "a transient failure must not be retried forever"
    assert report.total_failure
    assert report.failed[0].attempts == 3
    assert report.failed[0].retryable is True
    assert report.failed[0].error_type == "ConnectionError"


def test_a_missing_optional_dependency_is_not_retried(book_paths, fake_index):
    """A loader whose import fails will fail identically every time."""

    class LoaderMissingDependency:
        def __init__(self):
            self.calls = 0

        def __call__(self, file_path):
            self.calls += 1
            raise ModuleNotFoundError("No module named 'markdown'")

    loader = LoaderMissingDependency()
    backend = IngestionBackend(
        load=loader, chunk=chunk_documents, index=fake_index.index, purge=fake_index.purge
    )

    report, _ = ingest_collection(
        [book_paths[0]],
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        backend=backend,
        max_attempts=3,
    )

    assert loader.calls == 1, "a missing module will not appear on a retry"
    assert report.failed[0].retryable is False
    assert report.failed[0].error_type == "ModuleNotFoundError"


def test_a_permanent_failure_is_not_retried(book_paths, fake_index):
    class AlwaysUnsupported:
        def __init__(self):
            self.calls = 0

        def __call__(self, file_path):
            self.calls += 1
            raise ValueError("Unsupported format: '.xyz'")

    loader = AlwaysUnsupported()
    backend = IngestionBackend(
        load=loader, chunk=chunk_documents, index=fake_index.index, purge=fake_index.purge
    )

    report, _ = ingest_collection(
        [book_paths[0]],
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        backend=backend,
        max_attempts=3,
    )

    assert loader.calls == 1, "retrying an unsupported format only wastes time"
    assert report.failed[0].attempts == 1
    assert report.failed[0].retryable is False


# ── Metadata units ────────────────────────────────────────────────────


def test_loader_pages_are_normalised_to_one_based():
    assert normalise_page(0) == 1
    assert normalise_page(11) == 12
    assert normalise_page("4") == 5
    assert normalise_page(None) is None
    assert normalise_page("front matter") is None


def test_headings_lose_their_markdown_emphasis():
    """PyMuPDF4LLM renders bold PDF headings as '**Day 1**'; a citation must not."""
    assert headings_in("## **Day 1 Contents**") == ["Day 1 Contents"]
    assert headings_in("# *Intro* and `code`") == ["Intro and code"]
    assert headings_in("### ~~Struck~~ Heading") == ["Struck Heading"]


def test_heading_cleanup_keeps_underscores_in_identifiers():
    """An underscore in a heading is far more often an identifier than emphasis."""
    assert headings_in("## The chunk_index field") == ["The chunk_index field"]


def test_a_book_title_is_cleaned_too(tmp_path):
    book = tmp_path / "notes.md"
    book.write_text("# **Bold Book**\n\nbody text\n", encoding="utf-8")
    assert book_title_from(markdown_loader(str(book)), book) == "Bold Book"


def test_a_chunk_belongs_to_the_section_it_starts_in():
    sections = assign_sections(
        [
            "# Book\n\nintro text\n\n## First\n\nbody of first",
            "more body of first\n\n## Second\n\nbody of second",
            "still inside second",
        ]
    )
    assert sections == ["Book", "First", "Second"]


def test_pages_are_flagged_when_they_are_estimated(indexed_books):
    """Markdown has no pages, so the ordinal stands in — and says so."""
    _, records = indexed_books
    assert all(record.page_is_estimated for record in records)
    assert all(record.page == record.chunk_index + 1 for record in records)


def test_collection_id_is_required():
    with pytest.raises(ValueError, match="collection_id"):
        ingest_collection([], collection_id="  ", user_id=USER_ID)
