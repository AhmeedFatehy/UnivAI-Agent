"""Content identity: what makes a book the same book for the RAG cache.

A content-addressed cache can only be reused *safely* if its key names the
bytes and the pipeline that produced the vectors, never the filename or a
client-supplied hash. This module owns that key:

* ``content_hash`` — SHA-256 of the file's original bytes, computed server-side.
  :func:`file_content_hash` streams the file, so a large PDF is never read
  whole into memory, and :func:`file_size` is the byte length. Together they
  are what :class:`cache.artifact_registry.ArtifactRegistry.verify_reusable`
  checks again before any reuse — a corrupt or incomplete artifact is rejected,
  never trusted because of its name.
* ``pipeline_fingerprint`` — a SHA-256 over every component that shapes the
  vectors: the parser, the chunker (size/overlap), the dense and sparse
  embedding models and the chunk-metadata schema. Change any of them and the
  fingerprint changes, so an artifact built by an older pipeline can never be
  reused for a newer one.
* :func:`artifact_key` — one content-addressed key binding the two.

:class:`ContentIdentity` is the immutable, validated bundle that ingestion
reads off the disk before it touches the cache. Everything here is untrusted-input
safe: a filename or title is never hashed, only the raw bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

import config
from document_processing.metadata import SCHEMA_VERSION as CHUNK_METADATA_SCHEMA_VERSION

CONTENT_IDENTITY_SCHEMA = "univai.agent.content_identity"
CONTENT_IDENTITY_VERSION = "1.0.0"
PIPELINE_FINGERPRINT_VERSION = "1.0.0"

# Bump whenever the parsing logic in ``document_processing.loaders`` changes the
# text a file becomes, so an artifact built by an older parser is not reused.
PARSER_VERSION = "1.0.0"
# Bump whenever the chunking logic in ``document_processing.chunking`` changes.
SPLITTER_VERSION = "1.0.0"

_STREAM_BLOCK_BYTES = 1 << 20


def content_hash(data: bytes) -> str:
    """SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def file_content_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes, read in 1 MiB blocks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_STREAM_BLOCK_BYTES), b""):
            hasher.update(block)
    return hasher.hexdigest()


def file_size(path: str | Path) -> int:
    """Byte length of the file on disk."""
    return Path(path).stat().st_size


class PipelineComponents(BaseModel):
    """Every knuckle that shapes the indexed vectors, in one fingerprint.

    A change to any field must change :meth:`fingerprint`, which changes the
    artifact key and therefore forces a fresh build instead of a stale reuse.
    """

    schema_version: str = PIPELINE_FINGERPRINT_VERSION
    parser_version: str = PARSER_VERSION
    splitter_version: str = SPLITTER_VERSION
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    dense_embedding_model: str = Field(min_length=1)
    sparse_embedding_model: str = Field(min_length=1)
    chunk_metadata_schema_version: str = Field(min_length=1)

    def fingerprint(self, *, document_type: str | None = None) -> str:
        """Stable SHA-256 over the components plus the document's format.

        ``document_type`` is the file format (from the extension) because the
        parser/splitter selection depends on it; two byte-identical files with
        different formats are different artifacts.
        """
        components = [
            ("parser", self.parser_version),
            ("splitter", self.splitter_version),
            ("chunk_size", str(self.chunk_size)),
            ("chunk_overlap", str(self.chunk_overlap)),
            ("dense_embedding", self.dense_embedding_model),
            ("sparse_embedding", self.sparse_embedding_model),
            ("chunk_metadata_schema", self.chunk_metadata_schema_version),
        ]
        if document_type:
            components.append(("document_type", document_type))
        seed = "\n".join(f"{name}:{value}" for name, value in components)
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def default_pipeline_components() -> PipelineComponents:
    """The live pipeline components, read from configuration at call time."""
    return PipelineComponents(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        dense_embedding_model=config.DENSE_EMBEDDING_MODEL,
        sparse_embedding_model=config.SPARSE_EMBEDDING_MODEL,
        chunk_metadata_schema_version=CHUNK_METADATA_SCHEMA_VERSION,
    )


def artifact_key(content_hash: str, fingerprint: str) -> str:
    """Bind the byte identity and the pipeline identity into one addressable key."""
    if not content_hash or not fingerprint:
        raise ValueError("content_hash and fingerprint are both required")
    combined = f"{content_hash}:{fingerprint}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class ContentIdentity(BaseModel):
    """Immutable identity of one file under one pipeline, read from disk.

    ``artifact_key`` is the cache key; ``content_hash``, ``byte_size`` and
    ``pipeline_fingerprint`` are kept alongside so a reuse can be re-verified
    rather than trusted.
    """

    model_config = {"frozen": True}

    schema_name: str = CONTENT_IDENTITY_SCHEMA
    schema_version: str = CONTENT_IDENTITY_VERSION
    content_hash: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    document_type: str = Field(min_length=1)
    pipeline_fingerprint: str = Field(min_length=1)
    artifact_key: str = Field(min_length=1)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        pipeline: PipelineComponents | None = None,
    ) -> "ContentIdentity":
        """Compute the identity from the file's own bytes and the pipeline."""
        source = Path(path)
        pipeline = pipeline or default_pipeline_components()
        content_hash = file_content_hash(source)
        size = file_size(source)
        document_type = source.suffix.lstrip(".") or "unknown"
        fingerprint = pipeline.fingerprint(document_type=document_type)
        return cls(
            content_hash=content_hash,
            byte_size=size,
            document_type=document_type,
            pipeline_fingerprint=fingerprint,
            artifact_key=artifact_key(content_hash, fingerprint),
        )


__all__ = [
    "CONTENT_IDENTITY_SCHEMA",
    "CONTENT_IDENTITY_VERSION",
    "PARSER_VERSION",
    "PIPELINE_FINGERPRINT_VERSION",
    "SPLITTER_VERSION",
    "ContentIdentity",
    "PipelineComponents",
    "artifact_key",
    "content_hash",
    "default_pipeline_components",
    "file_content_hash",
    "file_size",
]
