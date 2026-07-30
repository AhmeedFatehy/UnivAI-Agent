"""Chunk identity and citation metadata for multi-book ingestion.

Every chunk that reaches the vector store has to be able to answer one
question: *which book, which page, which section did this come from?* Without
that, an answer cannot be cited and therefore cannot be trusted.

This module owns that identity:

* :class:`SourceLocation` — the citation itself (collection, document, book,
  page, section). It is the single shape every grounded output cites with.
* :class:`ChunkMetadata` — the full per-chunk record written onto the LangChain
  ``Document.metadata`` before indexing, so ``vector_store.indexing`` carries it
  into the Qdrant payload unchanged.

Pages come from the loader when the format has them (PDF). Text and Markdown
books have no pages, so the chunk's 1-based ordinal is used instead and flagged
with ``page_is_estimated`` — the number is never silently invented.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# Bumped whenever the payload shape below changes in a way consumers can see.
CHUNK_METADATA_SCHEMA = "univai.agent.chunk_metadata"
SCHEMA_VERSION = "1.0.0"

# ── Qdrant payload keys ───────────────────────────────────────────────
# ``vector_store.indexing.index_chunks`` promotes a fixed set of keys to the top
# level of the payload and nests everything else (stringified) under
# ``original_metadata``. Filters and citation reads must use these exact paths.
PAYLOAD_USER_ID = "user_id"
PAYLOAD_DOCUMENT_ID = "document_id"
PAYLOAD_SOURCE_FILENAME = "source_filename"
PAYLOAD_DOCUMENT_TYPE = "document_type"
PAYLOAD_PAGE_NUMBER = "page_number"
PAYLOAD_CHUNK_INDEX = "chunk_index"
PAYLOAD_COLLECTION_ID = "original_metadata.collection_id"
PAYLOAD_BOOK_TITLE = "original_metadata.book_title"
PAYLOAD_SECTION = "original_metadata.section"

#: Payload keys worth a Qdrant keyword index — the ones we filter on.
INDEXED_PAYLOAD_KEYS = (
    PAYLOAD_USER_ID,
    PAYLOAD_DOCUMENT_ID,
    PAYLOAD_COLLECTION_ID,
    PAYLOAD_BOOK_TITLE,
)

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_SETEXT_RE = re.compile(r"^(?!\s*$)(.+)\n[=-]{3,}\s*$", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# Emphasis markers to strip out of a heading before it becomes a citation.
# PyMuPDF4LLM renders bold PDF headings as "**Day 1**", and a citation reading
# "**Day 1** — p. 33 § **Viewing File Content**" is noise.
# Underscores are deliberately left alone: inside a heading an underscore is far
# more likely part of an identifier (chunk_index) than markdown emphasis.
_EMPHASIS_RE = re.compile(r"\*+|`+|~~")


def clean_heading(text: str) -> str:
    """Strip markdown emphasis and stray hashes from a heading."""
    cleaned = _EMPHASIS_RE.sub("", text or "")
    return cleaned.strip().strip("#").strip()


class SourceLocation(BaseModel):
    """Where a piece of evidence physically lives. This *is* the citation."""

    model_config = {"frozen": True}

    collection_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    chunk_index: int | None = Field(default=None, ge=0)
    source_filename: str | None = None
    page_is_estimated: bool = False

    @field_validator("section")
    @classmethod
    def _blank_section_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def locator(self) -> str:
        """Short human locator, e.g. ``p. 12 (est.) § Sorting``."""
        parts: list[str] = []
        if self.page is not None:
            parts.append(f"p. {self.page}{' (est.)' if self.page_is_estimated else ''}")
        if self.section:
            parts.append(f"§ {self.section}")
        if not parts and self.chunk_index is not None:
            parts.append(f"chunk {self.chunk_index}")
        return " ".join(parts) or "location unknown"

    def label(self) -> str:
        """Full citation label used in prompts and rendered answers."""
        return f"{self.book_title} — {self.locator()}"


class ChunkMetadata(BaseModel):
    """Everything written onto a chunk before it is embedded and indexed."""

    schema_name: str = CHUNK_METADATA_SCHEMA
    schema_version: str = SCHEMA_VERSION

    collection_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    document_type: str = Field(min_length=1)

    page: int | None = Field(default=None, ge=1)
    page_is_estimated: bool = False
    section: str | None = None
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=1)
    ingested_at: str

    def as_source_location(self) -> SourceLocation:
        return SourceLocation(
            collection_id=self.collection_id,
            document_id=self.document_id,
            book_title=self.book_title,
            page=self.page,
            section=self.section,
            chunk_index=self.chunk_index,
            source_filename=self.source_filename,
            page_is_estimated=self.page_is_estimated,
        )

    def as_document_metadata(self) -> dict:
        """Flat dict merged into ``Document.metadata`` before indexing.

        ``page`` and ``page_number`` are both set because
        ``vector_store.indexing`` reads ``page`` first and falls back to
        ``page_number``; keeping them in sync keeps the top-level payload
        field correct regardless of which one it picks up.
        """
        data = self.model_dump()
        if self.page is not None:
            data["page"] = self.page
            data["page_number"] = self.page
        else:
            data.pop("page", None)
        return data


def stable_document_id(collection_id: str, source_filename: str) -> str:
    """Deterministic document id, so re-ingesting a book keeps its identity."""
    if not collection_id.strip():
        raise ValueError("collection_id is required")
    if not source_filename.strip():
        raise ValueError("source_filename is required")
    seed = f"univai:collection:{collection_id}:document:{source_filename}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def normalise_page(raw: object) -> int | None:
    """Coerce a loader page value to a 1-based page number.

    PyMuPDF4LLM reports 0-based page indices, so a ``0`` becomes page 1.
    Anything non-numeric returns ``None`` rather than guessing.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw + 1 if raw >= 0 else None
    if isinstance(raw, float):
        return int(raw) + 1 if raw >= 0 else None
    if isinstance(raw, str):
        text = raw.strip()
        if text.isdigit():
            return int(text) + 1
    return None


def headings_in(text: str) -> list[str]:
    """Every Markdown heading in ``text``, in order of appearance."""
    found: list[tuple[int, str]] = [
        (match.start(), clean_heading(match.group(2)))
        for match in _HEADING_RE.finditer(text)
    ]
    found.extend(
        (match.start(), clean_heading(match.group(1)))
        for match in _SETEXT_RE.finditer(text)
    )
    found.sort(key=lambda item: item[0])
    return [title for _, title in found if title]


def assign_sections(texts: list[str], *, initial: str | None = None) -> list[str | None]:
    """Attach the governing section heading to each chunk.

    Chunks arrive in document order. A chunk belongs to the section it *starts*
    in: its own leading heading if it has one, otherwise the last heading seen
    in an earlier chunk. A heading further down the chunk governs the chunks
    after it, not the text above it.
    """
    current = initial
    sections: list[str | None] = []
    for text in texts:
        found = headings_in(text)
        opening = _leading_heading(text)
        sections.append(opening or current)
        if found:
            current = found[-1]
    return sections


def _leading_heading(text: str) -> str | None:
    """The heading a chunk opens with, if the chunk opens with one."""
    stripped = _FRONT_MATTER_RE.sub("", text).lstrip()
    if not stripped:
        return None
    match = _HEADING_RE.match(stripped)
    if match:
        return clean_heading(match.group(2)) or None
    setext = _SETEXT_RE.match(stripped)
    if setext:
        return clean_heading(setext.group(1)) or None
    return None


def book_title_from(documents: list, path: Path | str) -> str:
    """Best available title for a book: metadata title, first H1, then filename."""
    source = Path(path)
    for document in documents or []:
        metadata = getattr(document, "metadata", None) or {}
        for key in ("title", "book_title", "Title"):
            candidate = metadata.get(key)
            if isinstance(candidate, str) and clean_heading(candidate):
                return clean_heading(candidate)

    if documents:
        head = _FRONT_MATTER_RE.sub("", getattr(documents[0], "page_content", "") or "")
        found = headings_in(head)
        if found:
            return found[0]

    return source.stem.replace("_", " ").replace("-", " ").strip() or source.name


def apply_chunk_metadata(
    chunks: list,
    *,
    collection_id: str,
    document_id: str,
    user_id: str,
    book_title: str,
    source_filename: str,
    document_type: str,
    ingested_at: str | None = None,
) -> list[ChunkMetadata]:
    """Stamp identity onto every chunk of one book, in place.

    Returns the records that were written, in chunk order.
    """
    if not chunks:
        raise ValueError(f"'{source_filename}' produced no chunks to index")

    stamped_at = ingested_at or datetime.now(timezone.utc).isoformat()
    texts = [getattr(chunk, "page_content", "") or "" for chunk in chunks]
    sections = assign_sections(texts)
    total = len(chunks)

    records: list[ChunkMetadata] = []
    for index, (chunk, section) in enumerate(zip(chunks, sections)):
        existing = getattr(chunk, "metadata", None)
        if existing is None:
            existing = {}
            chunk.metadata = existing

        loader_page = normalise_page(existing.get("page", existing.get("page_number")))
        record = ChunkMetadata(
            collection_id=collection_id,
            document_id=document_id,
            user_id=user_id,
            book_title=book_title,
            source_filename=source_filename,
            document_type=document_type,
            page=loader_page if loader_page is not None else index + 1,
            page_is_estimated=loader_page is None,
            section=section,
            chunk_index=index,
            total_chunks=total,
            ingested_at=stamped_at,
        )
        existing.update(record.as_document_metadata())
        records.append(record)

    return records


def citation_from_payload(payload: dict) -> SourceLocation | None:
    """Rebuild a :class:`SourceLocation` from a retrieval hit.

    Accepts either a raw Qdrant payload or the flattened dict produced by
    ``retrieval.hybrid_search``. Returns ``None`` when the hit predates this
    schema and carries no collection/book identity — an uncitable hit must not
    be silently promoted into a citation.
    """
    nested = payload.get("original_metadata") or payload.get("metadata") or {}
    if not isinstance(nested, dict):
        nested = {}

    def pick(key: str) -> object:
        value = payload.get(key)
        if value in (None, ""):
            value = nested.get(key)
        return value

    collection_id = pick("collection_id")
    document_id = pick("document_id")
    book_title = pick("book_title") or pick("source_filename")
    if not collection_id or not document_id or not book_title:
        return None

    page = pick("page_number")
    if page in (None, ""):
        page = pick("page")
    page_number: int | None
    if isinstance(page, bool):
        page_number = None
    elif isinstance(page, int):
        page_number = page if page >= 1 else None
    elif isinstance(page, str) and page.strip().isdigit():
        page_number = int(page) or None
    else:
        page_number = None

    chunk_index = pick("chunk_index")
    if isinstance(chunk_index, str) and chunk_index.strip().lstrip("-").isdigit():
        chunk_index = int(chunk_index)
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
        chunk_index = None

    estimated = pick("page_is_estimated")
    if isinstance(estimated, str):
        estimated = estimated.strip().lower() == "true"

    section = pick("section")
    filename = pick("source_filename")

    return SourceLocation(
        collection_id=str(collection_id),
        document_id=str(document_id),
        book_title=str(book_title),
        page=page_number,
        section=str(section) if section else None,
        chunk_index=chunk_index,
        source_filename=str(filename) if filename else None,
        page_is_estimated=bool(estimated),
    )


__all__ = [
    "CHUNK_METADATA_SCHEMA",
    "SCHEMA_VERSION",
    "INDEXED_PAYLOAD_KEYS",
    "PAYLOAD_BOOK_TITLE",
    "PAYLOAD_CHUNK_INDEX",
    "PAYLOAD_COLLECTION_ID",
    "PAYLOAD_DOCUMENT_ID",
    "PAYLOAD_DOCUMENT_TYPE",
    "PAYLOAD_PAGE_NUMBER",
    "PAYLOAD_SECTION",
    "PAYLOAD_SOURCE_FILENAME",
    "PAYLOAD_USER_ID",
    "ChunkMetadata",
    "SourceLocation",
    "apply_chunk_metadata",
    "assign_sections",
    "book_title_from",
    "citation_from_payload",
    "headings_in",
    "normalise_page",
    "stable_document_id",
]
