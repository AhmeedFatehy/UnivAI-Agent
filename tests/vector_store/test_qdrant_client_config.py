from unittest.mock import Mock

import vector_store.qdrant_client as qdrant_module


def setup_function() -> None:
    qdrant_module.get_qdrant_client.cache_clear()


def teardown_function() -> None:
    qdrant_module.get_qdrant_client.cache_clear()


def test_configured_api_key_is_passed_to_qdrant(monkeypatch) -> None:
    constructor = Mock(return_value=object())
    monkeypatch.setattr(qdrant_module, "QdrantClient", constructor)
    monkeypatch.setattr(qdrant_module, "QDRANT_URL", "https://qdrant.example")
    monkeypatch.setattr(qdrant_module, "QDRANT_API_KEY", "secret-key")

    first = qdrant_module.get_qdrant_client()
    second = qdrant_module.get_qdrant_client()

    assert first is second
    constructor.assert_called_once_with(
        url="https://qdrant.example",
        api_key="secret-key",
    )


def test_missing_api_key_is_explicitly_passed_as_none(monkeypatch) -> None:
    constructor = Mock(return_value=object())
    monkeypatch.setattr(qdrant_module, "QdrantClient", constructor)
    monkeypatch.setattr(qdrant_module, "QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(qdrant_module, "QDRANT_API_KEY", None)

    qdrant_module.get_qdrant_client()

    constructor.assert_called_once_with(
        url="http://localhost:6333",
        api_key=None,
    )
