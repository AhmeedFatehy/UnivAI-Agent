"""Content identity: SHA-256 of bytes + pipeline fingerprint → one safe key.

These tests pin the two invariants that make a content-addressed cache safe:

* the key is derived from the file's *bytes* — never its filename or a client
  hash — so identical bytes always collide and different bytes never do;
* a change to any pipeline component (chunking, embedding model, schema)
  changes the fingerprint and therefore forces a new artifact instead of a
  stale reuse.
"""

from __future__ import annotations

import hashlib

import pytest

from cache.content_identity import (
    ContentIdentity,
    PipelineComponents,
    artifact_key,
    content_hash,
    default_pipeline_components,
    file_content_hash,
    file_size,
)


def _pipeline(**overrides) -> PipelineComponents:
    kwargs = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "dense_embedding_model": "dense-model-a",
        "sparse_embedding_model": "sparse-model-a",
        "chunk_metadata_schema_version": "1.0.0",
        "parser_version": "1.0.0",
        "splitter_version": "1.0.0",
    }
    kwargs.update(overrides)
    return PipelineComponents(**kwargs)


# ── Bytes, not names ──────────────────────────────────────────────────


def test_the_hash_is_sha256_of_the_bytes():
    assert content_hash(b"hello") == hashlib.sha256(b"hello").hexdigest()
    assert content_hash(b"hello") != content_hash(b"hello!")


def test_file_content_hash_matches_the_raw_bytes(tmp_path):
    book = tmp_path / "book.md"
    book.write_bytes(b"alpha\nbeta\n")
    assert file_content_hash(book) == hashlib.sha256(b"alpha\nbeta\n").hexdigest()
    assert file_size(book) == len(b"alpha\nbeta\n")


def test_identical_bytes_in_different_filenames_share_a_key(tmp_path):
    first = tmp_path / "book_a.md"
    second = tmp_path / "totally_different_name.md"
    first.write_text("the same content", encoding="utf-8")
    second.write_text("the same content", encoding="utf-8")

    left = ContentIdentity.from_file(first, pipeline=_pipeline())
    right = ContentIdentity.from_file(second, pipeline=_pipeline())

    assert left.content_hash == right.content_hash
    assert left.artifact_key == right.artifact_key


def test_different_bytes_never_collide(tmp_path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("one paragraph of content", encoding="utf-8")
    b.write_text("one paragraph of content, and more", encoding="utf-8")

    assert ContentIdentity.from_file(a, pipeline=_pipeline()).artifact_key != (
        ContentIdentity.from_file(b, pipeline=_pipeline()).artifact_key
    )


def test_the_filename_extension_still_picks_the_parser(tmp_path):
    """Byte-identical files with different formats are different artifacts."""
    a = tmp_path / "book.md"
    b = tmp_path / "book.txt"
    a.write_bytes(b"identical bytes")
    b.write_bytes(b"identical bytes")

    assert ContentIdentity.from_file(a, pipeline=_pipeline()).artifact_key != (
        ContentIdentity.from_file(b, pipeline=_pipeline()).artifact_key
    )


def test_a_truncated_file_has_a_different_hash(tmp_path):
    full = tmp_path / "full.md"
    partial = tmp_path / "partial.md"
    full.write_text("a longer book body", encoding="utf-8")
    partial.write_text("a longer book", encoding="utf-8")

    assert ContentIdentity.from_file(full, pipeline=_pipeline()).artifact_key != (
        ContentIdentity.from_file(partial, pipeline=_pipeline()).artifact_key
    )


# ── Pipeline fingerprint ──────────────────────────────────────────────


def test_each_pipeline_component_changes_the_fingerprint():
    baseline = _pipeline()
    changes = {
        "chunk_size": 700,
        "chunk_overlap": 100,
        "dense_embedding_model": "dense-model-b",
        "sparse_embedding_model": "sparse-model-b",
        "chunk_metadata_schema_version": "2.0.0",
        "parser_version": "2.0.0",
        "splitter_version": "2.0.0",
    }
    for field, value in changes.items():
        changed = _pipeline(**{field: value})
        assert changed.fingerprint(document_type="md") != baseline.fingerprint(
            document_type="md"
        ), f"changing {field} must change the fingerprint"


def test_the_document_type_is_part_of_the_fingerprint():
    assert _pipeline().fingerprint(document_type="pdf") != _pipeline().fingerprint(
        document_type="md"
    )


def test_the_fingerprint_is_deterministic():
    assert _pipeline().fingerprint(document_type="md") == _pipeline().fingerprint(
        document_type="md"
    )


def test_a_changed_embedding_model_forces_a_new_artifact_key():
    key_before = artifact_key("ab" * 32, _pipeline().fingerprint(document_type="md"))
    changed = _pipeline(dense_embedding_model="a-different-model")
    key_after = artifact_key("ab" * 32, changed.fingerprint(document_type="md"))
    assert key_before != key_after


# ── The default pipeline is the live one ──────────────────────────────


def test_the_default_pipeline_reads_the_live_config(monkeypatch):
    import config

    monkeypatch.setattr(config, "CHUNK_SIZE", 1234)
    components = default_pipeline_components()
    assert components.chunk_size == 1234
    assert components.dense_embedding_model == config.DENSE_EMBEDDING_MODEL


def test_artifact_key_requires_both_parts():
    with pytest.raises(ValueError):
        artifact_key("", "fingerprint")
    with pytest.raises(ValueError):
        artifact_key("hash", "")
