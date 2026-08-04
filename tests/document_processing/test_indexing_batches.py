"""The production indexer must never embed a complete textbook at once."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

import vector_store.indexing as indexing


class DenseEmbedder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, texts):
        values = list(texts)
        self.batch_sizes.append(len(values))
        return [[float(index), 0.0] for index, _ in enumerate(values)]


class SparseEmbedder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, texts):
        values = list(texts)
        self.batch_sizes.append(len(values))
        return [
            SimpleNamespace(indices=ArrayValue([0]), values=ArrayValue([1.0]))
            for _ in values
        ]


class ArrayValue(list):
    def tolist(self):
        return list(self)


class FakeClient:
    def __init__(self) -> None:
        self.upload_sizes: list[int] = []
        self.deletes = []

    def upload_points(self, *, points, **_kwargs):
        self.upload_sizes.append(len(points))

    def delete(self, **kwargs):
        self.deletes.append(kwargs)


def test_indexer_embeds_and_uploads_bounded_batches(monkeypatch):
    dense = DenseEmbedder()
    sparse = SparseEmbedder()
    client = FakeClient()
    monkeypatch.setattr(indexing, "EMBEDDING_BATCH_SIZE", 3)
    monkeypatch.setattr(indexing, "ensure_collection", lambda _name: None)
    monkeypatch.setattr(indexing, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(indexing, "get_dense_embedder", lambda: dense)
    monkeypatch.setattr(indexing, "get_sparse_embedder", lambda: sparse)

    chunks = [Document(page_content=f"chunk {index}", metadata={}) for index in range(8)]
    result = indexing.index_chunks(chunks, "student-a")

    assert result["chunks_indexed"] == 8
    assert dense.batch_sizes == [3, 3, 2]
    assert sparse.batch_sizes == [3, 3, 2]
    assert client.upload_sizes == [3, 3, 2]
    assert client.deletes == []


def test_later_batch_failure_rolls_back_only_the_current_indexing_run(monkeypatch):
    dense = DenseEmbedder()
    sparse = SparseEmbedder()
    client = FakeClient()
    uploads = 0

    def fail_second_upload(*, points, **_kwargs):
        nonlocal uploads
        uploads += 1
        client.upload_sizes.append(len(points))
        if uploads == 2:
            raise ConnectionError("qdrant unavailable")

    client.upload_points = fail_second_upload
    monkeypatch.setattr(indexing, "EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(indexing, "ensure_collection", lambda _name: None)
    monkeypatch.setattr(indexing, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(indexing, "get_dense_embedder", lambda: dense)
    monkeypatch.setattr(indexing, "get_sparse_embedder", lambda: sparse)
    chunks = [Document(page_content=f"chunk {index}", metadata={}) for index in range(5)]

    try:
        indexing.index_chunks(chunks, "student-a", document_id="stable-document")
    except ConnectionError:
        pass
    else:
        raise AssertionError("the simulated Qdrant failure was swallowed")

    assert client.upload_sizes == [2, 2]
    assert len(client.deletes) == 1
    selector = client.deletes[0]["points_selector"]
    conditions = selector.filter.must
    assert len(conditions) == 1
    assert conditions[0].key == "indexing_run_id"
    assert conditions[0].match.value
