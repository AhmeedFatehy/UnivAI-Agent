from unittest.mock import Mock

from qdrant_client import models

from vector_store import collection_manager


class FakeQdrantClient:
    def __init__(self, count: int = 3, exists: bool = True) -> None:
        self.count_value = count
        self.exists = exists
        self.count_filter = None
        self.delete_kwargs = None

    def collection_exists(self, *, collection_name):
        return self.exists

    def count(self, *, collection_name, count_filter):
        self.count_filter = count_filter
        return Mock(count=self.count_value)

    def delete(self, **kwargs):
        self.delete_kwargs = kwargs


def _matches(filter_value: models.Filter) -> dict[str, object]:
    return {
        condition.key: condition.match.value
        for condition in filter_value.must
        if isinstance(condition, models.FieldCondition)
    }


def test_delete_document_is_tenant_and_document_scoped(monkeypatch) -> None:
    client = FakeQdrantClient()
    monkeypatch.setattr(collection_manager, "get_qdrant_client", lambda: client)
    monkeypatch.setattr("cache.authorization.revoke_document", Mock())

    deleted = collection_manager.delete_document(
        "student-a",
        "document-123",
        grants=Mock(),
        registry=Mock(),
    )

    assert deleted == 3
    assert _matches(client.count_filter) == {
        "user_id": "student-a",
        "document_id": "document-123",
    }
    selector = client.delete_kwargs["points_selector"]
    assert _matches(selector.filter) == {
        "user_id": "student-a",
        "document_id": "document-123",
    }
    assert client.delete_kwargs["wait"] is True


def test_one_tenant_cannot_delete_another_tenants_same_document_id(monkeypatch) -> None:
    client = FakeQdrantClient(count=0)
    monkeypatch.setattr(collection_manager, "get_qdrant_client", lambda: client)
    monkeypatch.setattr("cache.authorization.revoke_document", Mock())

    collection_manager.delete_document(
        "student-a",
        "shared-deterministic-id",
        grants=Mock(),
        registry=Mock(),
    )

    selector = client.delete_kwargs["points_selector"]
    matches = _matches(selector.filter)
    assert matches["user_id"] == "student-a"
    assert matches["document_id"] == "shared-deterministic-id"
    assert "student-b" not in matches.values()


def test_delete_is_idempotent_before_qdrant_collection_exists(monkeypatch) -> None:
    client = FakeQdrantClient(exists=False)
    monkeypatch.setattr(collection_manager, "get_qdrant_client", lambda: client)
    monkeypatch.setattr("cache.authorization.revoke_document", Mock())

    deleted = collection_manager.delete_document(
        "student-a",
        "pending-document",
        grants=Mock(),
        registry=Mock(),
    )

    assert deleted == 0
    assert client.count_filter is None
    assert client.delete_kwargs is None
