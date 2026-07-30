"""Shared test fixtures: three books, a fake vector store, a scripted LLM.

Nothing here reaches the network. Qdrant, the embedding models and the LLM are
all replaced by deterministic doubles, so the suite exercises the real batching,
metadata, grounding, planning and graph logic without a live service or a paid
call.

The three books under ``tests/fixtures/books/`` are project-authored and short,
so a reviewer can open one and check a citation's page and section by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# The agent's runtime picker defaults to "integrated"; the tests never touch a
# real service, so the mode is pinned to keep imports deterministic.
os.environ.setdefault("UNIVAI_MODE", "standalone")

from document_processing.batch_ingestion import IngestionBackend, ingest_collection
from document_processing.chunking import chunk_documents
from document_processing.metadata import ChunkMetadata

FIXTURE_BOOKS = Path(__file__).resolve().parent / "fixtures" / "books"
COLLECTION_ID = "cs-programme-2026"
USER_ID = "student-fixture"


class FakeDocument:
    """Stand-in for a LangChain ``Document`` — the two attributes we rely on."""

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = dict(metadata or {})


def markdown_loader(file_path: str) -> list[FakeDocument]:
    """Load a Markdown fixture without ``unstructured``'s NLTK downloads.

    ``document_processing.loaders`` routes ``.md`` through
    ``UnstructuredMarkdownLoader``, which wants model data at import time. The
    fixture books are plain UTF-8, so reading them directly keeps the suite
    offline while still feeding the real chunker real Markdown.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return [
        FakeDocument(
            path.read_text(encoding="utf-8"),
            {"source": str(path), "document_type": path.suffix.lstrip(".")},
        )
    ]


@dataclass
class FakeIndex:
    """In-memory stand-in for Qdrant that keeps whole chunks and their payload."""

    rows: list[dict] = field(default_factory=list)
    calls: int = 0

    def index(self, **kwargs) -> dict:
        self.calls += 1
        chunks = kwargs["chunks"]
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            self.rows.append(
                {
                    "id": f"{metadata['document_id']}:{metadata['chunk_index']}",
                    "content": chunk.page_content,
                    "page_content": chunk.page_content,
                    "user_id": kwargs["user_id"],
                    "document_id": metadata["document_id"],
                    "source_filename": kwargs["source_filename"],
                    "document_type": kwargs["document_type"],
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata["chunk_index"],
                    "total_chunks": metadata["total_chunks"],
                    # Mirrors how vector_store.indexing nests and stringifies
                    # everything it does not promote to the top level.
                    "metadata": {
                        key: str(value) for key, value in metadata.items()
                    },
                }
            )
        return {
            "document_id": kwargs["document_id"],
            "collection_name": kwargs.get("collection_name") or "test_collection",
            "chunks_indexed": len(chunks),
            "user_id": kwargs["user_id"],
        }

    def purge(self, *, user_id: str, document_id: str, collection_name: str | None) -> int:
        before = len(self.rows)
        self.rows = [
            row
            for row in self.rows
            if not (row["user_id"] == user_id and row["document_id"] == document_id)
        ]
        return before - len(self.rows)

    def backend(self) -> IngestionBackend:
        return IngestionBackend(
            load=markdown_loader,
            chunk=chunk_documents,
            index=self.index,
            purge=self.purge,
        )


@dataclass
class FakeRetriever:
    """Lexical retrieval over the fake index, honouring the same filters.

    Scores by content-term overlap, which is enough to make relevant passages
    win and irrelevant ones lose deterministically.
    """

    rows: list[dict]
    calls: list[dict] = field(default_factory=list)

    def __call__(self, **kwargs) -> list[dict]:
        from planning.overlap import content_terms

        self.calls.append(kwargs)
        wanted = content_terms(kwargs.get("query", ""))
        user_id = kwargs.get("user_id")
        collection_id = kwargs.get("collection_id")
        document_ids = kwargs.get("document_ids")
        book_titles = kwargs.get("book_titles")
        limit = kwargs.get("limit") or 5

        scored: list[tuple[float, dict]] = []
        for row in self.rows:
            if user_id and row.get("user_id") != user_id:
                continue
            nested = row.get("metadata") or {}
            if collection_id and nested.get("collection_id") != collection_id:
                continue
            if document_ids and row.get("document_id") not in document_ids:
                continue
            if book_titles and nested.get("book_title") not in book_titles:
                continue
            overlap = wanted & content_terms(row["content"])
            if not overlap:
                continue
            scored.append((len(overlap) / max(1, len(wanted)), row))

        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        hits = []
        for score, row in scored[:limit]:
            hit = dict(row)
            hit["score"] = round(score, 4)
            nested = hit.get("metadata") or {}
            hit["collection_id"] = nested.get("collection_id")
            hit["book_title"] = nested.get("book_title")
            hit["section"] = nested.get("section")
            hits.append(hit)
        return hits


class ScriptedLLM:
    """A deterministic ``str -> str`` model.

    ``responses`` maps a marker that appears in the prompt to the replies to
    give, in order. Each call records the prompt so a test can assert on what
    the model was actually shown.
    """

    def __init__(self, responses: dict[str, list[str]], default: str = "{}"):
        self.responses = {key: list(value) for key, value in responses.items()}
        self.default = default
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for marker, replies in self.responses.items():
            if marker in prompt and replies:
                return replies.pop(0)
        return self.default

    @property
    def calls(self) -> int:
        return len(self.prompts)


@pytest.fixture(scope="session")
def book_paths() -> list[Path]:
    paths = sorted(FIXTURE_BOOKS.glob("*.md"))
    assert len(paths) == 3, f"expected three fixture books, found {len(paths)}"
    return paths


@pytest.fixture
def fake_index() -> FakeIndex:
    return FakeIndex()


@pytest.fixture
def indexed_books(book_paths, fake_index) -> tuple[FakeIndex, list[ChunkMetadata]]:
    """The three books, ingested into the fake index under one collection."""
    report, records = ingest_collection(
        book_paths,
        collection_id=COLLECTION_ID,
        user_id=USER_ID,
        backend=fake_index.backend(),
        ingested_at="2026-07-30T00:00:00+00:00",
    )
    assert report.complete_success, report.failed
    flat = [record for values in records.values() for record in values]
    return fake_index, flat


@pytest.fixture
def retriever(indexed_books) -> FakeRetriever:
    index, _ = indexed_books
    return FakeRetriever(index.rows)
